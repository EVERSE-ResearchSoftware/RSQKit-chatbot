from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(
        self,
        task_name,
        config,
        provider,
        # , model, function, provider_resources
    ):
        self.task_name = task_name
        self.config = config
        self.provider = provider
        # self.model = model
        # self.function = function
        # self.provider_resources = provider_resources

    @abstractmethod
    def render_ui(self):
        """Render task-specific UI elements."""
        pass

    @abstractmethod
    def process_input(self, prompt, session_key):
        """Process user input and generate a response."""
        pass
