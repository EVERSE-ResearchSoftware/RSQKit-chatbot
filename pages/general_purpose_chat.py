import streamlit as st
from llm_provider_tools import get_default_llm
from llms.openai_interface import get_chat_response_stream
from ui.header import sidebar, ICONS
from ui.custom_styles import inject_page_styles

GENERAL_PURPOSE_CHAT_SESSION_KEY = "general"


st.set_page_config(
    page_title="General Chatbot",
    page_icon=ICONS["chat"],
    layout="wide",
    initial_sidebar_state="expanded",
)


sidebar(page_key=GENERAL_PURPOSE_CHAT_SESSION_KEY)
provider_key = f"provider_{GENERAL_PURPOSE_CHAT_SESSION_KEY}"
selected_provider = st.session_state[provider_key]
llm_model = get_default_llm(selected_provider=selected_provider)

inject_page_styles()

st.markdown(f'<h1 class="main-title">{ICONS["chat"]} Chat</h1>', unsafe_allow_html=True)


# Chat history
def init_general_bot_messages():
    if GENERAL_PURPOSE_CHAT_SESSION_KEY not in st.session_state:
        st.session_state[GENERAL_PURPOSE_CHAT_SESSION_KEY] = {"messages": []}


# Display chat history
def display_chat_history():
    for message in st.session_state[GENERAL_PURPOSE_CHAT_SESSION_KEY]["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_conversation_with_memory(prompt: str):
    st.session_state[GENERAL_PURPOSE_CHAT_SESSION_KEY]["messages"].append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ollama response streaming
    with st.chat_message("assistant"):
        full_resp = ""
        # with chat_win:
        placeholder = st.empty()

        for token in get_chat_response_stream(
            provider=selected_provider,
            model_name=llm_model,
            messages=st.session_state[GENERAL_PURPOSE_CHAT_SESSION_KEY]["messages"],
        ):
            full_resp += token
            placeholder.write(full_resp)

    st.session_state[GENERAL_PURPOSE_CHAT_SESSION_KEY]["messages"].append(
        {"role": "assistant", "content": full_resp}
    )


init_general_bot_messages()

display_chat_history()
# Chat input
prompt = st.chat_input("Ask me anything...")

if prompt:
    handle_conversation_with_memory(prompt=prompt)
