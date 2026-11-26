from pathlib import Path

import streamlit as st
from helper_functions.Utility import check_password

st.set_page_config(page_title="AI Chatbot on Python", page_icon="🤖")

# Do not continue if check_password is not True.
if not check_password():  
    st.stop()


st.title("🤖 AI Chatbot")
st.write("""
Welcome to the AI Chatbot! Navigate using the **sidebar** to explore:

- 💬 **Chatbot** — Have a conversation with the AI assistant to udnerstand more about Python
- ℹ️ **About** — Learn more about this app
- 😎 **Admin** — Admin page for managing the app (password protected)
""")

with st.expander("⚠️ Important Notice"):
    st.markdown("""
**IMPORTANT NOTICE:** This web application is a prototype developed for **educational purposes only.** The information provided here is **NOT intended for real-world usage** and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

**Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.**

Always consult with qualified professionals for accurate and personalised advice.
""")
