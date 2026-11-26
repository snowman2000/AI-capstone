import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

SYSTEM_PROMPT_NO_DOC = "You are a helpful assistant."
CHROMA_DB_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
ACTIVE_INDEX_FILE = CHROMA_DB_DIR / "active_index.txt"


def load_and_split(uploaded_file, chunk_size: int = 800, chunk_overlap: int = 100):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in {".pdf", ".txt"}:
        raise ValueError(f"Unsupported file type: {file_extension}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=file_extension,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getvalue())

        if file_extension == ".pdf":
            documents = PyPDFLoader(temp_path).load()
        else:
            documents = TextLoader(temp_path, encoding="utf-8").load()
        for document in documents:
            document.metadata["source"] = uploaded_file.name
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    chunks_key = hashlib.sha256(
        json.dumps(
            [
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in chunks
            ],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    vectorstore = _cached_build_vectorstore(chunks, chunks_key)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_INDEX_FILE.write_text(chunks_key, encoding="utf-8")
    return vectorstore


@st.cache_resource
def _cached_build_vectorstore(chunks, chunks_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    persist_directory = CHROMA_DB_DIR / chunks_key
    if (persist_directory / "chroma.sqlite3").is_file():
        return Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain",
        persist_directory=str(persist_directory),
    )


def load_persisted_vectorstore() -> Chroma | None:
    if not ACTIVE_INDEX_FILE.is_file():
        return None

    chunks_key = ACTIVE_INDEX_FILE.read_text(encoding="utf-8").strip()
    if len(chunks_key) != 64 or any(
        character not in "0123456789abcdef" for character in chunks_key
    ):
        raise ValueError("The persisted Chroma index key is invalid.")

    persist_directory = CHROMA_DB_DIR / chunks_key
    if not (persist_directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"The persisted Chroma index was not found at {persist_directory}."
        )
    return _cached_build_vectorstore([], chunks_key)


def retrieve_context(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
) -> tuple[str, list[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    documents = retriever.invoke(query)
    context = "\n\n---\n\n".join(document.page_content for document in documents)
    return context, documents


def rewrite_query(
    client: OpenAI,
    query: str,
    history: list[dict[str, str]],
    model: str,
    system_prompt: str = SYSTEM_PROMPT_NO_DOC,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    "Rewrite the latest user question as a concise, standalone "
                    "search query using the conversation history for context. "
                    "Do not answer the question. Return only the rewritten query."
                ),
            },
            *history,
            {"role": "user", "content": query},
        ],
    )
    rewritten_query = response.choices[0].message.content
    if rewritten_query is None or not rewritten_query.strip():
        raise ValueError("The model returned an empty search query.")
    return rewritten_query.strip()


def build_rag_system_prompt(context: str) -> str:
    return (
        "Answer the user's question using ONLY the information in the provided "
        "context. If the answer is not in the context, say exactly: "
        "\"I couldn't find that information in the uploaded document.\""
        f"\n\nContext:\n{context}"
    )


@st.cache_data(show_spinner=False)
def summarize_document(_document_text: str, document_key: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Summarize the provided document in one paragraph.",
            },
            {"role": "user", "content": _document_text},
        ],
    )
    summary = response.choices[0].message.content
    if summary is None:
        raise ValueError("The model returned an empty document summary.")
    return summary
