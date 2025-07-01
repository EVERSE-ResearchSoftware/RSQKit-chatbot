# styles.py
import streamlit as st

CSS_CONTENT = """
    <style>
    /* outer frame of each source */
    .source-card {
        background: #f7f7f9;           /* light‑grey on white */
        border: 1px solid #e1e1e8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.04);
    }
    /* heading "Source #" */
    .source-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: .5rem;
        color: #333;
    }
    /* coloured divider line */
    .source-divider {
        height: 1px;
        background: linear-gradient(90deg,#ff6b6b 0%,#f06595 100%);
        border: 0;
        margin: .8rem 0 1rem 0;
    }
    /* key‑value pairs */
    .meta-key   { font-weight: 500; color:#444; }
    .meta-value { color:#222;       }
    </style>
    """


def inject_page_styles():
    st.markdown(
        """
    <style>
        .main-title {
            text-align: center;
            color: #2E4053;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
            margin-top: 1rem;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
