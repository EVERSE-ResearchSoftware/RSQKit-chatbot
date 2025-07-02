import io
from abc import ABC, abstractmethod
from typing import Union, List

DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODING = "latin-1"


class BaseFileHandler(ABC):
    """Abstract base class for file handlers."""

    @abstractmethod
    def can_handle(self, file_extension: str) -> bool:
        """Check if this handler can process the given file extension."""
        pass

    @abstractmethod
    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        """Read and extract content from the file."""
        pass

    def _decode_bytes(self, content: bytes) -> str:
        """Decode bytes with fallback encoding."""
        try:
            return content.decode(DEFAULT_ENCODING)
        except UnicodeDecodeError:
            return content.decode(FALLBACK_ENCODING, errors="ignore")

    def _read_text_file(self, filepath: str) -> str:
        """Read text file with encoding detection."""
        try:
            with open(filepath, "r", encoding=DEFAULT_ENCODING) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding=FALLBACK_ENCODING, errors="ignore") as f:
                return f.read()

    def _read_file_content(self, file: Union[str, io.IOBase]) -> str:
        """Helper method to read file content with proper encoding handling."""
        if hasattr(file, "read"):
            file.seek(0)
            content = file.read()
            if isinstance(content, bytes):
                return self._decode_bytes(content)
            return content
        else:
            return self._read_text_file(file)
