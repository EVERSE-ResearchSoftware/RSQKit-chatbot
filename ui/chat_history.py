import streamlit as st


@st.cache_data(show_spinner=False, ttl=3600)
def _wrap_sources_html(sources_html: str) -> str:
    # Lightweight toggle with native HTML. No Streamlit widget = fewer elements.
    # style kept minimal; adjust to your theme if needed.
    return f"""
<details style="margin:0.25rem 0 0.75rem 0;">
  <summary style="cursor:pointer; font-weight:600;">View sources</summary>
  <div class="sources-container" style="margin-top:0.5rem;">{sources_html}</div>
</details>
"""


def check_for_assistant_message(current_page_key: str):
    last_message_entry = st.session_state[current_page_key]["messages"][-1] if st.session_state[current_page_key]["messages"] else {}
    # print(last_message_entry)
    if last_message_entry:
        is_assistant  = last_message_entry.get('role', "") == "assistant"
        print(is_assistant)
        if is_assistant and "sources_html" in last_message_entry:
            return  True
        else:
            return False
    return False
def display_chat_history(current_page_key: str, view: bool = False, persist_last_view: bool = True):
    """
    Displays chat history with pre-computed HTML sources
    """
    for message in st.session_state[current_page_key]["messages"]:
        if message["role"] != "system":
            # Display message content
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

            # Display sources OUTSIDE chat_message context to avoid HTML escaping
            if message["role"] == "assistant" and "sources_html" in message and view:
                st.markdown(
                _wrap_sources_html(message["sources_html"]),
                unsafe_allow_html=True
                )
                
    if persist_last_view:
        if not view and check_for_assistant_message(current_page_key):
            st.markdown(
                _wrap_sources_html(message["sources_html"]),
                unsafe_allow_html=True
                )
