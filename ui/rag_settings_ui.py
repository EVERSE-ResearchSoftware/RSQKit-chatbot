# rag_settings.py

import streamlit as st
from ui.header import ICONS


def display_rag_settings():
    with st.sidebar:
        with st.expander(f"{ICONS['rag_settings']} RAG Settings", expanded=False):
            st.subheader("Multi-Retrieval Settings")
            enable_multi_retrieval = st.checkbox("Enable Multi-Retrieval", value=True)
            show_retrieval_details = st.checkbox("Show Retrieval Details", value=False)
            show_decomposition = st.checkbox("Show Query Decomposition", value=True)

            st.subheader("Retrieval Strategy Settings")
            default_strategy = st.selectbox(
                "Default Strategy",
                ["hybrid", "semantic", "keyword", "multi_step"],
                index=0,
            )

            max_subqueries = st.slider("Max Subqueries", 1, 10, 5)

            # Return the variables if needed
            return {
                "enable_multi_retrieval": enable_multi_retrieval,
                "show_retrieval_details": show_retrieval_details,
                "show_decomposition": show_decomposition,
                "default_strategy": default_strategy,
                "max_subqueries": max_subqueries,
            }


# # rag_settings.py

# import streamlit as st
# from ui.header import ICONS


# def display_rag_settings():
#     """
#     Displays the RAG settings in the sidebar and returns the current settings.
#     The settings are stored in st.session_state for persistence across reruns.

#     Returns:
#         dict: A dictionary containing the current RAG settings.
#     """
#     with st.sidebar:
#         with st.expander(f"{ICONS['rag_settings']} RAG Settings", expanded=False):
#             st.subheader("Multi-Retrieval Settings")

#             # Enable Multi-Retrieval checkbox
#             enable_multi_retrieval = st.checkbox(
#                 "Enable Multi-Retrieval",
#                 value=True,
#                 key="enable_multi_retrieval"
#             )

#             # Show Retrieval Details checkbox
#             show_retrieval_details = st.checkbox(
#                 "Show Retrieval Details",
#                 value=False,
#                 key="show_retrieval_details"
#             )

#             # Show Query Decomposition checkbox
#             show_decomposition = st.checkbox(
#                 "Show Query Decomposition",
#                 value=True,
#                 key="show_decomposition"
#             )

#             st.subheader("Retrieval Strategy Settings")

#             # Default Strategy selectbox
#             default_strategy = st.selectbox(
#                 "Default Strategy",
#                 ["hybrid", "semantic", "keyword", "multi_step"],
#                 index=0,
#                 key="default_strategy"
#             )

#             # Max Subqueries slider
#             max_subqueries = st.slider(
#                 "Max Subqueries",
#                 min_value=1,
#                 max_value=10,
#                 value=5,
#                 key="max_subqueries"
#             )

#     # Return the current settings from session state
#     return {
#         "enable_multi_retrieval": st.session_state.enable_multi_retrieval,
#         "show_retrieval_details": st.session_state.show_retrieval_details,
#         "show_decomposition": st.session_state.show_decomposition,
#         "default_strategy": st.session_state.default_strategy,
#         "max_subqueries": st.session_state.max_subqueries,
#     }
