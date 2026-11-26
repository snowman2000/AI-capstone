import hashlib
import json
import os

import streamlit as st

from helper_functions.Utility import check_admin_password
from helper_functions.app_helpers import (
    SYSTEM_PROMPT_NO_DOC,
    load_and_split,
    load_persisted_vectorstore,
    summarize_document,
    build_vectorstore,
)


st.title("🐼Admin")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    admin_password = st.secrets.get("ADMIN_PASSWORD") or os.getenv(
        "ADMIN_PASSWORD"
    )
    if not admin_password:
        st.error(
            "Admin access is not configured. Set ADMIN_PASSWORD in "
            "Streamlit secrets or environment variables."
        )
        st.stop()

    entered_password = st.text_input(
        "Enter admin password",
        type="password",
        key="admin_password",
    )
    login_clicked = st.button("Login")

    if login_clicked:
        if check_admin_password(entered_password, admin_password):
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.text_input(
                "Access status",
                value="This page is restricted to Administrator",
                disabled=True,
            )
    st.stop()

if st.button("Log out"):
    st.session_state.admin_authenticated = False
    st.rerun()

if "vectorstore" not in st.session_state:
    persisted_vectorstore = load_persisted_vectorstore()
    if persisted_vectorstore is not None:
        st.session_state.vectorstore = persisted_vectorstore

uploaded_files = st.file_uploader(
    "📄 Upload Documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)
st.header("⚙️ Settings")
chunk_size = st.number_input(
    "Chunk size",
    min_value=1,
    value=800,
    step=1,
)
chunk_overlap = st.number_input(
    "Chunk overlap",
    min_value=0,
    value=100,
    step=1,
)

if uploaded_files:
    chunks = []
    for uploaded_file in uploaded_files:
        chunks.extend(load_and_split(uploaded_file, chunk_size, chunk_overlap))
    st.session_state.vectorstore = build_vectorstore(chunks)
    st.success(
        f"Ready! Indexed {len(chunks)} chunks from "
        f"{len(uploaded_files)} files."
    )
    document_text = "\n\n".join(chunk.page_content for chunk in chunks)
    file_fingerprints = sorted(
        (
            uploaded_file.name,
            hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
        )
        for uploaded_file in uploaded_files
    )
    document_key = hashlib.sha256(
        json.dumps(file_fingerprints).encode("utf-8")
    ).hexdigest()
    with st.spinner("Summarizing document..."):
        document_summary = summarize_document(document_text, document_key)
    st.markdown("**Document summary**")
    st.write(document_summary)
elif "vectorstore" in st.session_state:
    st.info("Using the saved document index from chroma_db/.")
else:
    st.info("Upload PDF or TXT files to enable document Q&A.")

st.session_state.system_prompt = st.text_area(
    "System Prompt",
    value=st.session_state.get("system_prompt", SYSTEM_PROMPT_NO_DOC),
    key="admin_system_prompt",
)
model_options = ["gpt-4o-mini", "gpt-4o"]
current_model = st.session_state.get("selected_model", model_options[0])
st.session_state.selected_model = st.selectbox(
    "Model",
    options=model_options,
    index=model_options.index(current_model),
    key="admin_selected_model",
)
st.session_state.k_value = st.slider(
    "Chunks to retrieve (k)",
    min_value=1,
    max_value=10,
    value=st.session_state.get("k_value", 4),
    key="admin_k_value",
)
st.metric(
    "Conversation characters",
    sum(len(message["content"]) for message in st.session_state.messages),
)
st.download_button(
    "Download Chat",
    data="\n\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in st.session_state.messages
    ),
    file_name="conversation.txt",
    mime="text/plain",
)
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
