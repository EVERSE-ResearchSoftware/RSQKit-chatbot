

# Implementing a New Task Class in the Task Management System
## Creating the New Task Class
### Create a New Python File:

- Navigate to the tasks/ directory.
- Create a new Python file named new_task_task.py.
### Define the New Task Class:

- Implement the NewTaskTask class by extending the BaseTask class.

- Add necessary UI elements and task processing logic.
```python
from streamlit import chat_message, session_state
from llms.openai_interface import get_chat_response_stream
from tasks.base_task import BaseTask

class NewTaskTask(BaseTask):
    def render_ui(self):
        # Add any UI elements specific to your task (e.g., sliders, file uploaders)
        st.text_input("Enter your input here", key="new_task_input")

    def process_input(self, prompt, session_key):
        # Implement your task logic here
        task_prompt = self.config["prompt"].format(content=prompt)
        session_state[session_key]["messages"].append({"role": "user", "content": task_prompt})

        with chat_message("user"):
            st.markdown(prompt)

        with chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            try:
                for token in get_chat_response_stream(
                    model_name=self.model,
                    messages=session_state[session_key]["messages"],
                    provider=self.provider,
                ):
                    full_response += token
                    placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

            session_state[session_key]["messages"].append({"role": "assistant", "content": full_response})
```
## Updating the Task Registry
### Navigate to the Task Registry File:

Open the pages/tasks.py file.
### Register the New Task:

- Import the NewTaskTask class.
- Add the new task to the TASK_REGISTRY dictionary.
```python
from tasks.new_task_task import NewTaskTask

TASK_REGISTRY = {
   
    "Code Reviewer": CodeReviewTask,
    "AI-OCR": AIOCRTask,
    "New Task": NewTaskTask,  # <-- Add your new task here
}
```
## Updating the Task Configuration
- In the file pages/tasks.py.

- Update the TASK_CONFIG to add the configuration for the New Task:

- Include the title and prompt for the new task in the TASK_CONFIG dictionary.
```python
TASK_CONFIG = {
    "New Task": {
        "title": "🔍 New Task Title",
        "prompt": "Process the following input: {content}",
    },
    # ... other tasks
}
```

By following these steps, you will successfully integrate a new task class into the task management system.