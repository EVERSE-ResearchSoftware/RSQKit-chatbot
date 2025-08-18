import streamlit as st
from app_config import init_global_session_state


def display_llm_settings():
    # init_global_session_state()  # set keys for retrieval_k, temperature, top_rerank
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        key="temperature_slider",
    )
    st.session_state.temperature = temperature
