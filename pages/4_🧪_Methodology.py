import streamlit as st

from helper_functions.Utility import check_password

st.set_page_config(
    layout="centered",
    page_title="Methodology",
    page_icon="🧪",
)

if not check_password():
    st.stop()

st.title("🧪 Methodology")
st.write(
    """
This application uses retrieval-augmented generation (RAG) to answer questions
from administrator-provided documents. Uploaded content is split into passages,
indexed locally for semantic search, and supplied to the language model only
when a document-backed question is asked.
"""
)

st.header("Data flow and implementation")
st.subheader("1. Document upload and parsing")
st.write(
    """
An administrator uploads one or more PDF or TXT files on the Admin page.
PDFs are read with `PyPDFLoader`; TXT files are read as UTF-8 with
`TextLoader`. Each upload is temporarily written to disk for parsing and then
removed. The uploaded filename is attached to each parsed document as its
`source` metadata.
"""
)

st.subheader("2. Chunking")
st.write(
    """
Parsed documents are split with LangChain's `RecursiveCharacterTextSplitter`.
The Admin page defaults to 800 characters per chunk and 100 characters of
overlap; these values are configurable. Overlap helps preserve context across
chunk boundaries. The resulting chunks, including their metadata, are the units
used for embedding and retrieval.
"""
)

st.subheader("3. Embeddings and vector storage")
st.write(
    """
Each chunk is embedded with OpenAI's `text-embedding-3-small` model. A SHA-256
key derived from the chunk text and metadata identifies that index. Chroma
stores the vectors persistently under `chroma_db/<index-key>/`; the active key
is written to `chroma_db/active_index.txt`. On a later app run, the chatbot
loads that active index into session state when available. This lets the
document index persist independently of an individual chat session.
"""
)

st.subheader("4. Query processing and retrieval")
st.write(
    """
When a user submits a question, the chatbot checks whether a vector store is
available. If the conversation has history, it first asks the selected chat
model to rewrite the latest question as a concise, standalone search query,
using the prior conversation for context. If rewriting raises an OpenAI or
validation error, the app reports the issue and searches with the original
question. The retriever returns the configured number of matching chunks
(default 4; adjustable from 1 to 10), and joins their text into the context
sent to the model.
"""
)

st.subheader("5. Answer generation")
st.write(
    """
For document-backed questions, the app adds a system instruction restricting
the answer to retrieved context and specifying a fixed response when that
context does not contain the answer. It sends this instruction, the configured
system prompt, and the conversation messages to the selected chat model
(`gpt-4o-mini` by default). The response is streamed to the chat interface and
saved in session state.

If there is no vector store, retrieval is skipped and the assistant answers
without document context, using the configured system prompt. In this mode,
answers are not constrained to uploaded sources.
"""
)

st.header("Use-case flowcharts")

st.subheader("Document ingestion and indexing")
st.graphviz_chart(
    """
digraph ingestion {
    graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4472C4", fontname="Arial"];
    edge [color="#52677D", fontname="Arial"];
    start [label="Admin uploads PDF/TXT"];
    parse [label="Parse file\\nPDF loader or UTF-8 text loader"];
    metadata [label="Attach source filename\\nto document metadata"];
    split [label="Split into chunks\\n(default: 800 chars, 100 overlap)"];
    embed [label="Create embeddings\\ntext-embedding-3-small"];
    store [label="Store vectors in persistent Chroma\\nchroma_db/<index-key>/"];
    active [label="Write active index key"];
    summary [label="Summarize uploaded text\\nGPT-4o-mini; cached by content key"];
    done [label="Index and summary ready"];
    start -> parse -> metadata -> split -> embed -> store -> active -> summary -> done;
}
""",
    use_container_width=True,
)

st.subheader("Question answering with uploaded documents")
st.graphviz_chart(
    """
digraph rag_chat {
    graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4472C4", fontname="Arial"];
    edge [color="#52677D", fontname="Arial"];
    start [label="User submits a question"];
    load [label="Load active Chroma index\\nif not already in session"];
    history [label="Conversation history available?"];
    rewrite [label="Rewrite question as standalone\\nsearch query with chat model"];
    original [label="Use original question\\n(no history to resolve)"];
    fallback [label="If rewrite fails, report error\\nand search original question"];
    search [label="Retrieve top-k matching chunks\\n(default k = 4)"];
    context [label="Join retrieved text into context"];
    prompt [label="Add context-only instruction\\nand conversation/system prompts"];
    generate [label="Generate selected-model response\\n(stream tokens to chat)"];
    save [label="Save assistant reply\\nin session state"];
    start -> load -> history;
    history -> rewrite [label="Yes"];
    history -> original [label="No"];
    rewrite -> search [label="Success"];
    rewrite -> fallback [label="OpenAI/validation error"];
    fallback -> search;
    original -> search;
    search -> context -> prompt -> generate -> save;
}
""",
    use_container_width=True,
)

st.subheader("Conversation without an indexed document")
st.graphviz_chart(
    """
digraph general_chat {
    graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4472C4", fontname="Arial"];
    edge [color="#52677D", fontname="Arial"];
    start [label="User submits a question"];
    check [label="Vector store available?"];
    no [label="No"];
    skip [label="Skip query rewrite and retrieval"];
    prompt [label="Use configured system prompt\\nwithout document context"];
    generate [label="Generate response with selected model\\n(stream tokens to chat)"];
    save [label="Save assistant reply\\nin session state"];
    start -> check -> no -> skip -> prompt -> generate -> save;
}
""",
    use_container_width=True,
)

st.subheader("Administrator settings and document summary")
st.graphviz_chart(
    """
digraph admin {
    graph [rankdir=TB, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#4472C4", fontname="Arial"];
    edge [color="#52677D", fontname="Arial"];
    start [label="Administrator opens Admin page"];
    auth [label="Validate administrator password"];
    allowed [label="Access granted"];
    configure [label="Set system prompt, chat model,\\nchunk settings, and retrieval k"];
    state [label="Save active controls in\\nStreamlit session state"];
    chat [label="Chat page reads model, prompt,\\nand k when processing questions"];
    upload [label="Document uploaded and indexed"];
    summarize [label="Send extracted chunk text to GPT-4o-mini\\nfor a one-paragraph summary"];
    cache [label="Cache summary using document content key"];
    end [label="Show summary on Admin page"];
    start -> auth -> allowed -> configure -> state -> chat;
    allowed -> upload -> summarize -> cache -> end;
}
""",
    use_container_width=True,
)

st.caption(
    "The regular app password protects the chatbot and informational pages. "
    "The Admin page uses a separate administrator password."
)
