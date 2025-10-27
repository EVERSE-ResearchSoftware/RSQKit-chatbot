import streamlit as st
import logging
from pathlib import Path
from llm_provider_tools import get_default_llm, _get_provider_config
from llms.openai_interface import get_chat_response_stream
from ui.header import sidebar
from app_config import ICONS
from ui.custom_styles import inject_page_styles
from app_config import get_selected_llm
from dotenv import load_dotenv
from provider_persistence import ProviderPersistence

load_dotenv()

# Import the MCP classes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
current_page_key = "general_chat"

# Initialize sidebar FIRST
sidebar(page_key=current_page_key)

# Now safely get the provider with persistence
selected_provider = ProviderPersistence.get_provider(current_page_key)

# Only proceed with LLM model selection if we have a provider
if selected_provider:
    try:
        llm_model = get_selected_llm(
            current_page_key, selected_provider
        ) or get_default_llm(selected_provider)
    except Exception as e:
        st.error(f"Error getting LLM model: {e}")
        llm_model = None
        # Clear problematic provider
        ProviderPersistence.set_provider(current_page_key, "")
        selected_provider = None
else:
    llm_model = None

inject_page_styles()
st.markdown(f'<h1 class="main-title">{ICONS["chat"]} Chat</h1>', unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = {"messages": []}


def display_history():
    """Display chat message history."""
    for msg in st.session_state[current_page_key]["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_streaming_conversation(prompt: str):
    """Handle conversation with streaming (no tools)."""
    if not selected_provider or not llm_model:
        st.error(
            "Provider or model not properly configured. Please check your settings."
        )
        return

    # Add user message
    st.session_state[current_page_key]["messages"].append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in get_chat_response_stream(
                provider=selected_provider,
                messages=st.session_state[current_page_key]["messages"],
                model_name=llm_model,
                temperature=st.session_state.get("temperature", 0.0),
            ):
                full_response += chunk
                placeholder.markdown(full_response)

            placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ **Error during conversation:** {str(e)}"
            full_response = error_msg
            placeholder.markdown(full_response)

    st.session_state[current_page_key]["messages"].append(
        {"role": "assistant", "content": full_response}
    )


# Initialize everything
init_session_state()

# Show configuration status
if not selected_provider:
    st.warning("🔄 Please select a provider from the sidebar to start chatting.")
    st.info("Your selection will be remembered across page refreshes!")
elif not llm_model:
    st.warning("🔄 Please select a model from the sidebar to start chatting.")
else:
    # Display UI components
    display_history()

    # Chat input
    prompt = st.chat_input("Ask me anything...")
    if prompt:
        handle_streaming_conversation(prompt=prompt)
