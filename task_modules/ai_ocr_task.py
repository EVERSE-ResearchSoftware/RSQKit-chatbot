import streamlit as st
from streamlit import file_uploader, session_state, chat_message
from PIL import Image
from llm_provider_tools import resolve_vision_function, _get_provider_config
from task_modules.base_task import BaseTask
import os


class AIOCRTask(BaseTask):
    def __init__(self, task_name, task_config, provider):
        super().__init__(task_name, task_config, provider)
        self.provider_config = _get_provider_config(provider=provider)

    def render_ui(self):
        uploaded_file = file_uploader(
            "Choose an image...", type=["png", "jpg", "jpeg", "gif", "bmp"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            session_state["ocr_uploaded_file"] = uploaded_file

    def process_input(self, prompt, session_key):
        uploaded_file = session_state.get("ocr_uploaded_file")
        if uploaded_file is None:
            st.error("Please upload an image first.")
            return

        with chat_message("user"):
            st.markdown(prompt)

        with chat_message("assistant"):
            vision_function = resolve_vision_function(provider=self.provider)
            response = vision_function(
                image=uploaded_file,
                question=prompt,
                model_name=self.provider_config.get("default_vision"),
                base_url=self.provider_config.get("base_url_vision"),
                api_key=os.environ.get(self.provider_config["api_env_var_vision"]),
            )
            st.markdown(response)
