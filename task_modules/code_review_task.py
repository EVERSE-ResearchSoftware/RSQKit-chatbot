
import streamlit as st
from streamlit import session_state
from llms.openai_interface import get_chat_response_stream
from task_modules.base_task import BaseTask
from llm_provider_tools import _get_provider_config


class CodeReviewTask(BaseTask):
    def __init__(self, task_name, task_config, provider):
        """
        Initialize the CodeReviewTask with the given configuration and provider.

        Args:
            task_name (str): Name of the task.
            config (dict): Configuration dictionary containing prompt and other settings.
            provider (str): Name of the AI provider to use.
        """
        super().__init__(task_name, task_config, provider)
        self.provider_config = _get_provider_config(provider=provider)
        self.session_key = "code_review"  # Fixed session key for this task

    def render_ui(self):
        """
        Render the user interface for the code review task.
        Includes a text area for code input and a button to trigger the review.
        """
        st.markdown("### 📝 Paste your code for review:")
        code_input = st.text_area("Paste your code here", key="code_input", height=300)
        if st.button("Submit for Review"):
            self.process_input(code_input, session_key=self.session_key)

    def process_input(self, prompt, session_key):
        """
        Process the user input and generate a response.

        Args:
            prompt (str): The user's code input.
            session_key (str): Key to store the conversation in session state.
        """
        try:
            # Format the prompt using the task-specific configuration
            task_prompt = self.config["prompt"].format(content=prompt)

            # Store the user's message in session state
            self._store_message(session_key, "user", task_prompt)

            # Display the user message
            self._display_user_message(task_prompt)

            # Generate and display the assistant response
            self._generate_and_display_assistant_response(session_key)

        except KeyError as ke:
            st.error(f"Missing configuration: {ke}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    def _store_message(self, session_key, role, content):
        """
        Store a message (user or assistant) in the session state.

        Args:
            session_key (str): Key to store the conversation.
            role (str): Role of the message ("user" or "assistant").
            content (str): Content of the message.
        """
        if session_key not in session_state:
            session_state[session_key] = {"messages": []}
        session_state[session_key]["messages"].append(
            {"role": role, "content": content}
        )

    def _display_user_message(self, prompt):
        """
        Display the user's message in the chat interface.

        Args:
            prompt (str): The message to display.
        """
        with st.chat_message("user"):
            st.markdown(prompt)

    def _generate_and_display_assistant_response(self, session_key):
        """
        Generate and display the assistant's response using a streaming interface.

        Args:
            session_key (str): Key to retrieve the conversation from session state.
        """
        full_response = ""
        placeholder = st.empty()

        try:
            # Validate that the model is available
            if not self.provider_config or "default_llm" not in self.provider_config:
                raise ValueError(
                    "No default model available for the selected provider."
                )

            # Stream the response from the model
            for token in get_chat_response_stream(
                model_name=self.provider_config["default_llm"],
                messages=session_state[session_key]["messages"],
                provider=self.provider,
            ):
                full_response += token
                placeholder.markdown(full_response)

            # Store the assistant's response in session state
            self._store_message(session_key, "assistant", full_response)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return


