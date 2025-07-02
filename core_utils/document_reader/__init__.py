"""
Document Reader Package

A robust toolkit for reading and extracting content from various file formats
including PDFs, Office documents, text files, HTML, Jupyter notebooks, and more.

This package provides a unified interface for processing different document types
in RAG (Retrieval-Augmented Generation) applications and other document processing
pipelines.

Key Features:
    - Support for 12+ file formats
    - Robust error handling with custom exceptions
    - Plugin-based handler architecture
    - Memory-efficient processing
    - Comprehensive logging
    - Encoding detection and fallback

Supported Formats:
    - PDF files (.pdf)
    - Microsoft Office documents (.docx, .xlsx, .ppt, .pptx)
    - Text files (.txt, .py, .json, .md)
    - Web formats (.html)
    - Jupyter Notebooks (.ipynb)
    - LaTeX documents (.tex)

Basic Usage:
    >>> from my_app.document_reader import FileReader
    >>> reader = FileReader()
    >>> content = reader.get_content("document.pdf")
    >>> pages = reader.get_pages("presentation.pptx")

Advanced Usage:
    >>> # Process entire directories
    >>> documents = reader.read_directory("./docs", recursive=True)
    >>>
    >>> # Handle errors gracefully
    >>> try:
    ...     content = reader.get_content("file.docx")
    >>> except FileReaderError as e:
    ...     print(f"Error: {e}")

Dependencies:
    The package has optional dependencies for different file types:
    - PyMuPDF: for PDF files
    - python-docx: for Word documents
    - openpyxl: for Excel files
    - python-pptx: for PowerPoint files
    - beautifulsoup4: for HTML files
    - nbformat: for Jupyter notebooks
    - pylatexenc: for LaTeX files
"""

# Import main classes and functions
from .reader import FileReader
from .exceptions import (
    FileReaderError,
    UnsupportedFileTypeError,
    FileProcessingError,
    MissingDependencyError,
)

# Import constants that users might need
from .reader import SUPPORTED_FORMATS

# Import handler base class for extensions
from .handlers.base import BaseFileHandler

# Import all handlers for direct access if needed
from .handlers import (
    PDFHandler,
    DocxHandler,
    ExcelHandler,
    PowerPointHandler,
    HTMLHandler,
    TextHandler,
    JupyterHandler,
    LaTeXHandler,
    HANDLER_CLASSES,
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "A robust document reader for RAG applications"

# Public API
__all__ = [
    # Main class
    "FileReader",
    # Exceptions
    "FileReaderError",
    "UnsupportedFileTypeError",
    "FileProcessingError",
    "MissingDependencyError",
    # Constants
    "SUPPORTED_FORMATS",
    # Base class for extensions
    "BaseFileHandler",
    # Individual handlers (for advanced usage)
    "PDFHandler",
    "DocxHandler",
    "ExcelHandler",
    "PowerPointHandler",
    "HTMLHandler",
    "TextHandler",
    "JupyterHandler",
    "LaTeXHandler",
    "HANDLER_CLASSES",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]


# Convenience functions for quick access
def get_supported_formats():
    """
    Get a dictionary of supported file formats.

    Returns:
        Dict[str, str]: Mapping of file extensions to descriptions
    """
    return SUPPORTED_FORMATS.copy()


def is_supported_file(filename: str) -> bool:
    """
    Check if a file is supported based on its extension.

    Args:
        filename: The filename to check

    Returns:
        bool: True if the file type is supported, False otherwise
    """
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


def create_reader(enable_logging: bool = True) -> FileReader:
    """
    Create a new FileReader instance with sensible defaults.

    Args:
        enable_logging: Whether to enable logging

    Returns:
        FileReader: A configured FileReader instance
    """
    return FileReader(enable_logging=enable_logging)


# Add convenience functions to __all__
__all__.extend(
    [
        "get_supported_formats",
        "is_supported_file",
        "create_reader",
    ]
)


# Optional: Lazy loading of handlers for better startup performance
def _get_available_handlers():
    """
    Get information about available handlers and their dependencies.

    Returns:
        Dict[str, Dict]: Handler information including availability
    """
    from .handlers import HANDLER_CLASSES

    handler_info = {}
    for handler_class in HANDLER_CLASSES:
        handler = handler_class()
        handler_name = handler_class.__name__

        # Try to determine if handler dependencies are available
        # This is a simple check - handlers will do proper checking
        try:
            # Most handlers check dependencies in their methods
            # so we'll just note the handler is available
            handler_info[handler_name] = {
                "class": handler_class,
                "available": True,  # Will be checked at runtime
                "formats": [
                    ext for ext in SUPPORTED_FORMATS.keys() if handler.can_handle(ext)
                ],
            }
        except Exception:
            handler_info[handler_name] = {
                "class": handler_class,
                "available": False,
                "formats": [],
            }

    return handler_info


# Expose handler information function
__all__.append("_get_available_handlers")


# Package-level configuration
class Config:
    """Package configuration settings."""

    # Default settings
    DEFAULT_ENCODING = "utf-8"
    FALLBACK_ENCODING = "latin-1"
    ENABLE_LOGGING = True
    LOG_LEVEL = "INFO"

    # Performance settings
    MAX_FILE_SIZE_MB = 100  # Maximum file size to process
    CHUNK_SIZE = 8192  # Reading chunk size for large files

    @classmethod
    def get_max_file_size_bytes(cls) -> int:
        """Get maximum file size in bytes."""
        return cls.MAX_FILE_SIZE_MB * 1024 * 1024


# Expose config
__all__.append("Config")


# Optional: Package-level logger setup
def setup_package_logging(level: str = "INFO"):
    """
    Setup package-wide logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging

    logger = logging.getLogger(__name__.split(".")[0])  # Get package root logger

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))


__all__.append("setup_package_logging")


# Package initialization
def _initialize_package():
    """Initialize the package with default settings."""
    if Config.ENABLE_LOGGING:
        setup_package_logging(Config.LOG_LEVEL)


# Run initialization
_initialize_package()
