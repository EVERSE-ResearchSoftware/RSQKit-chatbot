import streamlit as st
from functools import lru_cache
from settings import StreamlitKeys, PROVIDER_TO_RESOURCE_KEY


# Move set_page_config to a separate function that's called conditionally
def configure_page():
    """Configure page settings - only call this in the main app entry point"""
    st.set_page_config(
        page_title="Generative AI Services",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    


# CSS sidebar
SIDEBAR_STYLE = """
<style>
.nav-item {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 8px;
    transition: all 0.2s ease;
    text-decoration: none;
    color: inherit;
}

.nav-item:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.nav-icon {
    font-size: 1.2em;
    margin-right: 10px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

.provider-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
}

.divider {
    background: linear-gradient(90deg, transparent, #ddd, transparent);
    height: 1px;
    margin: 15px 0;
}

.new-chat-btn {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: pointer;
    width: 100%;
    box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
}

.new-chat-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(67, 233, 123, 0.4);
}
</style>
"""

ICONS = {
  "home": "🏡",
  "chat": "💬",
  "rsqkit_chat": "📖",
  "document_chat": "📋",
  "collections": "🗂",
  "rag_collections": "📑",
  "provider" : "⚙️",
  "rag_settings": "🛠️",
  "new_chat": "➕",
  "remote": "🚀",
  "local": "💻",
  "tasks": "🔀",
  "eosc": "🚀",
  "email": "📬",
  "api_down": "❌",
  "api_up": "✅",
}

pages_config = {
    "home": {
        "title": "Home",
        "page": "pages/home.py",
        "icon_key": "home",
        "session_key": "home",
    },
    "general_chat": {
        "title": "Chat",
        "page": "pages/general_purpose_chat.py",
        "icon_key": "chat",
        "session_key": "general_chat",
    },
    "rsqkit_chat": {
        "title": "RSQKit Chat",
        "page": "pages/rsqkit_rag_chat.py",
        "icon_key": "rsqkit_chat",
        "session_key": "rsqkit_chat",
    },
    "document_chat": {
        "title": "Document Chat",
        "page": "pages/document_chat.py",
        "icon_key": "document_chat",
        "session_key": "document_chat",
        "collection": "collection_document_chat_temp"
    },
    "collections": {
        "title": "Collections",
        "page": "pages/collections.py",
        "icon_key": "collections",
        "session_key": "collections",
    },
    "rag_collections": {
        "title": "RAG on a collection",
        "page": "pages/rag_on_collection.py",
        "icon_key": "rag_collections",
        "session_key": "rag_collections",
    },
    "tasks": {
        "title": "Tasks",
        "page": "pages/tasks.py",
        "icon_key": "tasks",
        "session_key": "tasks",
    },
    # "email": {
    #     "title": "Email Classifier",
    #     "page": "pages/email_classifier.py",
    #     "icon_key": "email",
    #     "session_key": "email",
    # },
}

DOCUMENT_CHAT_COLLECTION = pages_config["document_chat"]["collection"]

# Cache page icon mapping
@lru_cache(maxsize=128)
def get_page_icon_mapping():
    """Cache page icons to avoid repeated lookups"""
    return {
        page["session_key"]: ICONS[page["icon_key"]] for _, page in pages_config.items()
    }


def display_navigation_links():
    """Navigation sidebar sans provider selection"""
    with st.sidebar:
        # Inject CSS
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        # Use cached icon mapping
        page_icons = get_page_icon_mapping()
        for _, page in pages_config.items():
            icon = page_icons[page["session_key"]]
            st.page_link(pages[page["session_key"]], label=f"{icon} {page['title']}")


# Global variable to store pages once generated
_pages_cache = None


def generate_pages():
    """Generate pages with lazy initialization"""
    global _pages_cache
    if _pages_cache is None:
        _pages_cache = {
            page["session_key"]: st.Page(page["page"], title=page["title"])
            for _, page in pages_config.items()
        }
    return _pages_cache


pages = generate_pages()


def get_selected_llm_key(page_key, provider):
    """Generate LLM key for page and provider"""
    provider_id = PROVIDER_TO_RESOURCE_KEY.get(provider, "")
    return StreamlitKeys.SELECTED_LLM_PROVIDER + f"_{page_key}_{provider_id}"


def get_selected_llm(page_key: str, provider):
    """Get selected LLM for page and provider"""
    selected_llm_provider_key = get_selected_llm_key(
        page_key=page_key, provider=provider
    )
    # Ensure key exists before accessing
    ensure_session_state_key(selected_llm_provider_key, "")
    return st.session_state[selected_llm_provider_key]


def get_provider_key_page(page_key: str) -> str:
    """Get provider key for a specific page"""
    return f"provider_{page_key}"


def ensure_session_state_key(key: str, default_value=None):
    """Ensure session state key exists with default value"""
    if key not in st.session_state:
        st.session_state[key] = default_value


def get_page_keys():
    """Get all page keys"""
    return [page["session_key"] for _, page in pages_config.items()]


def get_provider_keys():
    """Get all provider keys"""
    return [get_provider_key_page(page_key=page_key) for page_key in get_page_keys()]


def get_selected_llm_keys():
    """Get all selected LLM keys"""
    keys = []
    for page_key in get_page_keys():
        for provider in PROVIDER_TO_RESOURCE_KEY:
            keys.append(get_selected_llm_key(page_key=page_key, provider=provider))
    return keys


def get_all_session_keys():
    """Get all session keys that need to be initialized"""
    return get_provider_keys() + get_selected_llm_keys()


# Initialize session state for global settings
def init_global_session_state():
    """Initialize session state variables that persist across pages"""
    try:
        # RAG Parameters
        ensure_session_state_key("retrieval_k", 5)
        ensure_session_state_key("top_rerank", 3)
        ensure_session_state_key("temperature", 0.0)

        # Initialize all page-specific keys
        for key in get_all_session_keys():
            ensure_session_state_key(key, "")

        # Initialize page-specific message storage
        for page_key in get_page_keys():
            ensure_session_state_key(page_key, {"messages": []})

    except Exception as e:
        st.error(f"Error initializing session state: {e}")


def init_page_session_state(page_key: str):
    """Initialize session state for a specific page"""
    try:
        # Ensure page exists
        ensure_session_state_key(page_key, {"messages": []})

        # Initialize retrieval_history for RAG-related pages
        if page_key in ["rag_chat", "document_chat", "rag_collections"]:
            if "retrieval_history" not in st.session_state[page_key]:
                st.session_state[page_key]["retrieval_history"] = []

        # Ensure provider key exists
        provider_key = get_provider_key_page(page_key)
        ensure_session_state_key(provider_key, "")

        # Ensure LLM keys exist for all providers
        for provider in PROVIDER_TO_RESOURCE_KEY:
            llm_key = get_selected_llm_key(page_key, provider)
            ensure_session_state_key(llm_key, "")

    except Exception as e:
        st.error(f"Error initializing page session state for {page_key}: {e}")


def safe_get_session_state(key: str, default=None):
    """Safely get session state value"""
    try:
        return st.session_state.get(key, default)
    except KeyError:
        ensure_session_state_key(key, default)
        return default


# Call global initialization when module loads, but make it safe
def _safe_init():
    """Safe initialization that doesn't fail if Streamlit isn't ready"""
    try:
        # Only initialize if we're in a Streamlit context
        if hasattr(st, "session_state"):
            init_global_session_state()
    except Exception:
        # If initialization fails, it will be called later
        pass


# Attempt safe initialization
_safe_init()


def main():
    """Main application entry point"""
    # Configure page settings - this should be the very first Streamlit command
    configure_page()

    # Ensure session state is initialized
    init_global_session_state()

    # Create navigation
    pg = st.navigation([value for _, value in pages.items()])

    # Run the selected page
    pg.run()


# Only run main if this file is executed directly
if __name__ == "__main__":
    main()