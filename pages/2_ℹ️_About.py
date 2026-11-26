import streamlit as st
from helper_functions.Utility import check_password

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="About",
    page_icon="ℹ️",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# endregion <--------- Streamlit Page Configuration --------->

st.title("ℹ️ About This Project")
st.write(
    """
This project is an educational chatbot that demonstrates how retrieval-
augmented generation (RAG) can connect a conversational language model to
documents supplied by an administrator. The pre-loaded document is on Python
programming, but the app can be configured to use any PDF or UTF-8 TXT files by administrator.
"""
)

with st.expander("Project Scope"):
    st.write(
        """
The application provides a password-protected Streamlit interface for
conversational questions about Python and administrator-uploaded reference
documents. Administrators can build a persistent local vector index from PDF
and TXT files. Questions can then be answered using retrieved passages, or
without document retrieval when no index is available.

This is a prototype for learning and demonstration, not a production knowledge
management system. It does not provide guaranteed factual accuracy, source
citations, or professional advice.
"""
    )

with st.expander("Objectives"):
    st.markdown(
        """
- Demonstrate a complete RAG workflow, from document ingestion through
  retrieval and answer generation.
- Make uploaded document content searchable through semantic embeddings.
- Support conversational follow-up questions by rewriting queries using chat
  history.
- Provide a simple interface for trying different model, prompt, and retrieval
  settings.
"""
    )

with st.expander("Data Source"):
    st.markdown(
        """
- **Reference documents:** Administrators upload PDF or UTF-8 TXT files. The
  app parses and splits these files into chunks before indexing them. This application is pre-loaded with some simple Python programming reference material.
- **Conversation:** The current chat history is sent to the language model and
  may be used to rewrite a follow-up question into a standalone search query.
- **AI services:** OpenAI's `text-embedding-3-small` creates document
  embeddings. The chat model can be set to `gpt-4o-mini` or `gpt-4o`.
"""
    )

st.header("Features")
st.markdown(
    """
<span style="background-color: lightblue"><strong>Users:</strong></span>

- 💬 **Conversational Q&A** — Ask questions and continue with follow-ups.
- ⚡ **Streaming answers** — Responses appear as they are generated.
- 📥 **Chat management** — Clear the current conversation.
- 🔐 **Access controls** — The main app uses a shared password; administrator
  functions require a separate administrator password.

<span style="background-color: lightblue"><strong>Admin:</strong></span>

- 📄 **PDF and TXT ingestion** — Upload documents, split them into overlapping
  chunks, and create a persistent Chroma vector index.
- 🔎 **Semantic retrieval** — Select the number of document chunks to provide
  as context for an answer.
- ✍️ **Configurable prompts and models** — Administrators can set the system
  prompt and choose between supported chat models.
""",
    unsafe_allow_html=True,
)

st.caption(
    "AI-generated output may be inaccurate. Use this prototype for educational "
    "purposes only and consult qualified professionals for consequential decisions."
)
