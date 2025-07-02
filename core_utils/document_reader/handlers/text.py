import os
import json
from typing import Union, List
from .base import BaseFileHandler
from ..exceptions import FileProcessingError
import io


class TextHandler(BaseFileHandler):
    """Handler for text-based files (Python, JSON, TXT, Markdown)."""

    def __init__(self):
        self.supported_extensions = {".py", ".json", ".txt", ".md"}

    def can_handle(self, file_extension: str) -> bool:
        return file_extension in self.supported_extensions

    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        try:
            content = self._read_file_content(file)

            # Special handling for JSON files
            if self._get_file_extension(file) == ".json":
                content = self._format_json(content)

            return [content] if return_pages else content
        except Exception as e:
            raise FileProcessingError(f"Error reading text file: {e}")

    def _get_file_extension(self, file: Union[str, io.IOBase]) -> str:
        """Get file extension from file object or path."""
        if hasattr(file, "name") and isinstance(file.name, str):
            return os.path.splitext(file.name)[1].lower()
        elif isinstance(file, str):
            return os.path.splitext(file)[1].lower()
        return ""

    def _format_json(self, content: str) -> str:
        """Format JSON content for better readability."""
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return content
