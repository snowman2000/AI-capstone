import streamlit as st
from openai import OpenAI, OpenAIError
from helper_functions.Utility import check_password

from helper_functions.app_helpers import (
    SYSTEM_PROMPT_NO_DOC,
    build_rag_system_prompt,
    load_persisted_vectorstore,
    retrieve_context,
    rewrite_query,
)

st.title("☁️ Ask Me about Python Programming basics")

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    persisted_vectorstore = load_persisted_vectorstore()
    if persisted_vectorstore is not None:
        st.session_state.vectorstore = persisted_vectorstore

st.sidebar.header("⚙️ Settings")
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful assistant.",
    key="system_prompt",
)
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

selected_model = st.session_state.get("selected_model", "gpt-4o-mini")
k_value = st.session_state.get("k_value", 4)

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if "vectorstore" in st.session_state:
        retrieval_query = prompt
        history = st.session_state.messages[:-1]
        if history:
            try:
                with st.spinner("Rewriting search query..."):
                    retrieval_query = rewrite_query(
                        client,
                        prompt,
                        history,
                        selected_model,
                        system_prompt,
                    )
            except (OpenAIError, ValueError):
                import traceback

                traceback.print_exc()
                st.error(
                    "I couldn't refine the search query, so I'll search using "
                    "your original message."
                )

        source_context, _ = retrieve_context(
            st.session_state.vectorstore,
            retrieval_query,
            k_value,
        )
        active_system_prompt = build_rag_system_prompt(source_context)
    else:
        active_system_prompt = SYSTEM_PROMPT_NO_DOC

    def generate_reply():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": active_system_prompt},
                    *st.session_state.messages,
                ],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_reply += token
                    yield token
        except Exception:
            import traceback

            traceback.print_exc()
            st.error("Sorry, I couldn't process your message. Please try again.")
            full_reply = "Sorry, I couldn't get a response. Please try again."
            yield full_reply

        st.session_state.messages.append(
            {"role": "assistant", "content": full_reply}
        )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.write_stream(generate_reply)
