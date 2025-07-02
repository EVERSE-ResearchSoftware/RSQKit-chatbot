import streamlit as st
from llm_provider_tools import (
    get_default_llm,
    get_default_vison_model,
    resolve_vision_function,
)
from llms.openai_interface import get_chat_response_stream
from ui.header import sidebar, ICONS
from ui.custom_styles import inject_page_styles
from dotenv import load_dotenv

from task_modules.code_review_task import CodeReviewTask
from task_modules.ai_ocr_task import AIOCRTask


load_dotenv()

# Constants
TASK_SESSION_KEY = "TASK_SESSION"
CODE_REVIEW = "Code Reviewer"
AI_OCR = "AI-OCR"
README_GENERATOR = "README Generator"


# Task configurations
TASK_CONFIG = {
    CODE_REVIEW: {
        "title": "Code Review",
        "prompt": """CODE:
```
{content}
```
Improve the above code to make it robust, maintainable and easy to read. Explain each modification and why.""".strip(),
    },
    AI_OCR: {"title": "Image to Text with AI", "prompt": "Perform OCR on this image"},
}


TASK_REGISTRY = {
    CODE_REVIEW: CodeReviewTask,
    AI_OCR: AIOCRTask,
}


###
def initialize_page_config():
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title="TASK Chatbot",
        page_icon=ICONS["chat"],
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_session_state():
    """Initialize session state variables in the correct order."""
    # Initialize main task session if not exists
    if TASK_SESSION_KEY not in st.session_state:
        st.session_state[TASK_SESSION_KEY] = {}


def get_task_selection():
    """Handle task selection and return selected task."""
    with st.container():
        selected_task = st.selectbox(
            label="Choose a task",
            label_visibility="hidden",
            options=[task for task in TASK_REGISTRY],
            index=0,
            key="task_selector",
        )
    return selected_task


def setup_provider_and_llm(selected_task_session):
    """Setup provider and LLM configuration."""
    # Call sidebar to ensure provider is set in session state
    sidebar(page_key=selected_task_session)

    provider_key = f"provider_{selected_task_session}"

    # Ensure provider key exists in session state
    if provider_key not in st.session_state:
        st.error("Provider configuration not found. Please check sidebar setup.")
        st.stop()

    selected_provider = st.session_state[provider_key]
    llm_model = get_default_llm(selected_provider=selected_provider)
    chat_function = get_chat_response_stream

    return selected_provider, llm_model, chat_function


def get_provider(selected_task_session):
    sidebar(page_key=selected_task_session)

    provider_key = f"provider_{selected_task_session}"

    # Ensure provider key exists in session state
    if provider_key not in st.session_state:
        st.error("Provider configuration not found. Please check sidebar setup.")
        st.stop()

    return st.session_state[provider_key]


def setup_provider_and_vision(selected_task_session):
    """Setup provider and LLM configuration."""
    # Call sidebar to ensure provider is set in session state
    sidebar(page_key=selected_task_session)

    provider_key = f"provider_{selected_task_session}"

    # Ensure provider key exists in session state
    if provider_key not in st.session_state:
        st.error("Provider configuration not found. Please check sidebar setup.")
        st.stop()

    selected_provider = st.session_state[provider_key]
    vision_model = get_default_vison_model(selected_provider=selected_provider)
    vision_function = resolve_vision_function(provider=selected_provider)

    return selected_provider, vision_model, vision_function


def get_resource_by_task(selected_task_session):
    """
    Returns tuple: provider, ai_model, chat or ocr function
    """
    if selected_task_session == AI_OCR:
        return setup_provider_and_vision(selected_task_session=selected_task_session)
    return setup_provider_and_llm(selected_task_session=selected_task_session)


def initialize_task_messages(task_session_key):
    """Initialize messages for the selected task."""
    if task_session_key not in st.session_state:
        st.session_state[task_session_key] = {"messages": []}


def display_chat_history(task_session_key):
    """Display chat history for the current task."""
    if (
        task_session_key in st.session_state
        and "messages" in st.session_state[task_session_key]
    ):
        for message in st.session_state[task_session_key]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def render_page_header(selected_task):
    """Render the page header with title."""
    inject_page_styles()

    task_title = TASK_CONFIG[selected_task]["title"]
    st.markdown(f'<h1 class="main-title">{task_title}</h1>', unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize page configuration
    initialize_page_config()

    # Initialize session state
    initialize_session_state()

    # Get task selection
    selected_task = get_task_selection()
    # Initialize task-specific messages
    selected_task_session = f"{TASK_SESSION_KEY}_{selected_task}"
    initialize_task_messages(selected_task_session)

    selected_provider = get_provider(selected_task_session=selected_task_session)

    # Display chat history
    display_chat_history(selected_task_session)

    ###
    # Insert task ui
    task_class = TASK_REGISTRY[selected_task]
    task_instance = task_class(
        task_name=selected_task,
        config=TASK_CONFIG[selected_task],
        provider=selected_provider,
    )

    task_instance.render_ui()
    ###

    # Handle user input
    prompt = st.chat_input("Ask me anything...")
    if prompt:
        task_instance.process_input(prompt, selected_task_session)


if __name__ == "__main__":
    main()
