import logging
import io
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from .handlers.base import BaseFileHandler
from .exceptions import UnsupportedFileTypeError
import os
from .handlers import HANDLER_CLASSES

# Constants
SUPPORTED_FORMATS = {
    ".pdf": "Portable Document Format",
    ".docx": "Microsoft Word Document",
    ".xlsx": "Microsoft Excel Spreadsheet",
    ".ppt": "Microsoft PowerPoint Presentation",
    ".pptx": "Microsoft PowerPoint Presentation",
    ".html": "HyperText Markup Language",
    ".py": "Python Script",
    ".json": "JavaScript Object Notation",
    ".txt": "Plain Text",
    ".ipynb": "Jupyter Notebook",
    ".tex": "LaTeX Document",
    ".md": "Markdown Document",
}


class FileReader:
    """
    A robust toolkit class to read and extract content from various file types.

    This class uses a plugin-based architecture with specialized handlers for each file type,
    making it easy to extend and maintain.
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the FileReader with handlers and optional logging.

        Args:
            enable_logging: Whether to enable logging for debugging
        """
        self._current_content: Optional[str] = None
        self._current_file_type: Optional[str] = None
        self.supported_formats = SUPPORTED_FORMATS.copy()

        # Initialize handlers from the registry
        self.handlers = [handler_class() for handler_class in HANDLER_CLASSES]

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging and not self.logger.handlers:
            self._setup_logging()

    def _setup_logging(self):
        """Setup basic logging configuration."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @property
    def current_content(self) -> Optional[str]:
        """Get the current extracted content of the file."""
        return self._current_content

    @property
    def current_file_type(self) -> Optional[str]:
        """Get the file type of the current file being processed."""
        return self._current_file_type

    def get_supported_extensions(self) -> List[str]:
        """Get a list of all supported file extensions."""
        return list(self.supported_formats.keys())

    def is_supported_file(self, filename: str) -> bool:
        """
        Check if a file is supported based on its extension.

        Args:
            filename: The filename to check

        Returns:
            True if the file type is supported, False otherwise
        """
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)

    def _detect_file_type(self, file: Union[str, io.IOBase]) -> str:
        """
        Detect the file type by extracting the file extension.

        Args:
            file: The file path or file-like object

        Returns:
            The file extension (e.g., '.pdf', '.docx')

        Raises:
            UnsupportedFileTypeError: If the file type cannot be determined or is unsupported
        """
        extension = None

        if hasattr(file, "name") and isinstance(file.name, str):
            _, extension = os.path.splitext(file.name)
        elif isinstance(file, str):
            _, extension = os.path.splitext(file)

        if not extension:
            raise UnsupportedFileTypeError(
                "Unable to detect file type: file must have a valid name or path with an extension"
            )

        extension = extension.lower()
        if extension not in self.supported_formats:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {extension}. "
                f"Supported formats: {', '.join(self.supported_formats.keys())}"
            )

        self._current_file_type = extension
        return extension

    def _get_handler(self, file_extension: str) -> BaseFileHandler:
        """
        Get the appropriate handler for a file extension.

        Args:
            file_extension: The file extension

        Returns:
            The handler that can process the file

        Raises:
            UnsupportedFileTypeError: If no handler is found
        """
        for handler in self.handlers:
            if handler.can_handle(file_extension):
                return handler

        raise UnsupportedFileTypeError(
            f"No handler found for file type: {file_extension}"
        )

    @contextmanager
    def _error_context(self, operation: str, file_info: str):
        """Context manager for consistent error handling and logging."""
        try:
            self.logger.info(f"Starting {operation} for {file_info}")
            yield
            self.logger.info(f"Successfully completed {operation} for {file_info}")
        except Exception as e:
            self.logger.error(f"Error during {operation} for {file_info}: {e}")
            raise

    def get_content(self, file: Union[str, io.IOBase]) -> str:
        """
        Get the full content of the file as a single string.

        Args:
            file: The file path or file-like object

        Returns:
            The content of the file as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFileTypeError: If the file type is not supported
            FileProcessingError: If there's an error processing the file
            MissingDependencyError: If a required dependency is missing
        """
        file_info = getattr(file, "name", str(file))

        with self._error_context("content extraction", file_info):
            file_type = self._detect_file_type(file)
            handler = self._get_handler(file_type)
            content = handler.read_content(file, return_pages=False)
            self._current_content = content
            return content

    def get_pages(self, file: Union[str, io.IOBase]) -> List[str]:
        """
        Get the content of the file as a list of pages or sections.

        Args:
            file: The file path or file-like object

        Returns:
            A list of strings, each representing a page or section

        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFileTypeError: If the file type is not supported
            FileProcessingError: If there's an error processing the file
            MissingDependencyError: If a required dependency is missing
        """
        file_info = getattr(file, "name", str(file))

        with self._error_context("page extraction", file_info):
            file_type = self._detect_file_type(file)
            handler = self._get_handler(file_type)
            pages = handler.read_content(file, return_pages=True)

            # Ensure we always return a list
            if isinstance(pages, str):
                pages = [pages]

            return pages

    def read_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        include_metadata: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Read all supported files in a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories recursively
            include_metadata: Whether to include file metadata

        Returns:
            List of tuples containing (file_path, content) or
            (file_path, content, metadata) if include_metadata is True
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        documents = []

        with self._error_context("directory reading", str(directory_path)):
            if recursive:
                file_iterator = directory_path.rglob("*")
            else:
                file_iterator = directory_path.iterdir()

            for file_path in file_iterator:
                if file_path.is_file() and self.is_supported_file(file_path.name):
                    try:
                        content = self.get_content(file_path)

                        if include_metadata:
                            metadata = self._get_file_metadata(file_path)
                            documents.append((str(file_path), content, metadata))
                        else:
                            documents.append((str(file_path), content))

                    except Exception as e:
                        self.logger.warning(
                            f"Skipping file {file_path} due to error: {e}"
                        )
                        continue

        return documents

    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata for a file."""
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "extension": file_path.suffix.lower(),
            "name": file_path.name,
        }

    def get_file_info(self, file: Union[str, io.IOBase]) -> Dict[str, Any]:
        """
        Get information about a file without reading its content.

        Args:
            file: The file path or file-like object

        Returns:
            Dictionary containing file information
        """
        try:
            file_type = self._detect_file_type(file)
            info = {
                "extension": file_type,
                "supported": True,
                "description": self.supported_formats.get(file_type, "Unknown"),
            }

            if isinstance(file, str):
                file_path = Path(file)
                if file_path.exists():
                    info.update(self._get_file_metadata(file_path))

            return info

        except UnsupportedFileTypeError:
            return {
                "extension": "unknown",
                "supported": False,
                "description": "Unsupported file type",
            }
