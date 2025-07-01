import streamlit as st
from settings import COLLECTIONS_SESSION, PROVIDER_ID_TO_NAME

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
}

pages = {
    "home": st.Page("app.py", title="Home"),
    "rsqkit_chat": st.Page("pages/rsqkit_rag_chat.py", title="RSQKit Chat"),
    "general_chat": st.Page("pages/general_purpose_chat.py", title="Chat"),
    "document_chat": st.Page("pages/document_chat.py", title="Document Chat"),
    "collections": st.Page("pages/collections.py", title="Collections"),
    "rag_collections": st.Page(
        "pages/rag_on_collection.py", title="RAG on a collection"
    ),
    "tasks": st.Page("pages/tasks.py", title="Tasks"),
}


# Helper function to get the appropriate icon
def get_icon(key):
    return ICONS["local"] if key == "ollama" else ICONS["remote"]


def reset_chat(page_key: str):
    """Reset chat messages for a given page"""
    st.session_state[page_key]["messages"] = []


def create_nav_link(icon_key, label):
    """Crée un lien de navigation stylisé"""
    return f"""
    <div class="nav-item">
        <span class="nav-icon">{ICONS[icon_key]}</span>
        <span>{label}</span>
    </div>
    """


def sidebar_items():
    """Navigation sidebar sans provider selection"""
    with st.sidebar:
        # Injection du CSS
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        # Navigation links avec style amélioré
        st.page_link(pages["home"], label=f"{ICONS['home']} Home")
        st.page_link(pages["general_chat"], label=f"{ICONS['chat']} Chat")
        st.page_link(pages["rsqkit_chat"], label=f"{ICONS['rsqkit_chat']} RSQKit Chat")
        st.page_link(
            pages["document_chat"], label=f"{ICONS['document_chat']} Document Chat"
        )
        st.page_link(
            pages["rag_collections"],
            label=f"{ICONS['rag_collections']} RAG on Collection",
        )
        st.page_link(pages["collections"], label=f"{ICONS['collections']} Collections")
        st.page_link(pages["tasks"], label=f"{ICONS['tasks']} Tasks")

        # Séparateur stylisé
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def sidebar(page_key: str):
    """Sidebar complète avec provider selection"""
    with st.sidebar:
        # Injection du CSS
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        # Navigation links
        st.page_link(pages["home"], label=f"{ICONS['home']} Home")
        st.page_link(pages["general_chat"], label=f"{ICONS['chat']} Chat")
        st.page_link(pages["rsqkit_chat"], label=f"{ICONS['rsqkit_chat']} RSQKit Chat")
        st.page_link(
            pages["document_chat"], label=f"{ICONS['document_chat']} Document Chat"
        )
        st.page_link(
            pages["rag_collections"],
            label=f"{ICONS['rag_collections']} RAG on Collection",
        )
        st.page_link(pages["collections"], label=f"{ICONS['collections']} Collections")
        st.page_link(pages["tasks"], label=f"{ICONS['tasks']} Tasks")
        # Séparateur
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        # Bouton nouvelle discussion avec style amélioré
        if page_key != COLLECTIONS_SESSION:
            if st.button(
                f"{ICONS['new_chat']} New Chat",
                help="Start a new chat",
                use_container_width=True,
            ):
                reset_chat(page_key)
                st.success("✅ New chat started!")

        with st.expander(f"{ICONS['provider']} AI Provider", expanded=True):

            # Generate formatted provider options
            provider_options = {
                display_name: f"{get_icon(key)} {display_name}"
                for key, display_name in PROVIDER_ID_TO_NAME.items()
            }

            # Streamlit radio widget
            provider = st.radio(
                label="Provider",
                options=list(provider_options.keys()),
                format_func=lambda x: provider_options[x],
                index=0,
                label_visibility="hidden",
            )
        # Mise à jour du state
        if "provider" not in st.session_state:
            st.session_state[f"provider_{page_key}"] = provider
