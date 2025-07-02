import streamlit as st
from settings import COLLECTIONS_SESSION, PROVIDER_ID_TO_NAME
from core_utils.health_api import check_api_key

# Configuration des icônes modernes et professionnelles
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

# Define icons
ICONS = {
    "home": "🏡",
    "chat": "💬",
    "rsqkit_chat": "📖",
    "document_chat": "📋",
    "collections": "🗂",
    "rag_collections": "📑",
    "provider": "⚙️",
    "rag_settings": "🛠️",
    "new_chat": "➕",
    "remote": "🚀",
    "local": "💻",
    "tasks": "🔀",
    "eosc": "🚀",
    "email": "📬",
}

# Centralized page configuration
pages_config = [
    {"key": "home", "title": "Home", "page": "app.py", "icon_key": "home"},
    {
        "key": "general_chat",
        "title": "Chat",
        "page": "pages/general_purpose_chat.py",
        "icon_key": "chat",
    },
    {
        "key": "rsqkit_chat",
        "title": "RSQKit Chat",
        "page": "pages/rsqkit_rag_chat.py",
        "icon_key": "rsqkit_chat",
    },
    {
        "key": "document_chat",
        "title": "Document Chat",
        "page": "pages/document_chat.py",
        "icon_key": "document_chat",
    },
    {
        "key": "collections",
        "title": "Collections",
        "page": "pages/collections.py",
        "icon_key": "collections",
    },
    {
        "key": "rag_collections",
        "title": "RAG on a collection",
        "page": "pages/rag_on_collection.py",
        "icon_key": "rag_collections",
    },
    {"key": "tasks", "title": "Tasks", "page": "pages/tasks.py", "icon_key": "tasks"},
]


# Generate the pages dictionary from the configuration
pages = {
    page["key"]: st.Page(page["page"], title=page["title"]) for page in pages_config
}


def display_navigation_links():
    """Navigation sidebar sans provider selection"""
    with st.sidebar:
        # Injection du CSS
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)
        for page in pages_config:
            icon = ICONS[page["icon_key"]]
            st.page_link(pages[page["key"]], label=f"{icon} {page['title']}")


# Helper function to get the appropriate icon for AI provider
def get_icon(provider_key):
    return ICONS["local"] if provider_key == "ollama" else ICONS["remote"]


# Function to reset chat messages for a given page
def reset_chat(page_key: str):
    """Reset chat messages for a given page"""
    st.session_state[page_key]["messages"] = []


# Function to render provider selection
def render_provider_selection(page_key: str):
    with st.expander(f"{ICONS['provider']} AI Provider", expanded=True):
        provider_options = {
            display_name: f"{get_icon(key)} {display_name}"
            for key, display_name in PROVIDER_ID_TO_NAME.items()
        }
        provider = st.radio(
            label="Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0,
            label_visibility="hidden",
        )
        # Update session state
        if "provider" not in st.session_state:
            st.session_state[f"provider_{page_key}"] = provider
        if not check_api_key(provider=provider):
            st.error(f"Missing API_KEY for {provider}")


def render_navigation_links():
    for page in pages_config:
        icon = ICONS[page["icon_key"]]
        st.page_link(pages[page["key"]], label=f"{icon} {page['title']}")


# Main sidebar rendering function
def sidebar(page_key: str):
    display_navigation_links()
    with st.sidebar:
        # Divider
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # New chat button
        if page_key != COLLECTIONS_SESSION:
            if st.button(
                f"{ICONS['new_chat']} New Chat",
                help="Start a new chat",
                use_container_width=True,
            ):
                reset_chat(page_key)
                st.success("✅ New chat started!")

        # Provider selection
        render_provider_selection(page_key)
