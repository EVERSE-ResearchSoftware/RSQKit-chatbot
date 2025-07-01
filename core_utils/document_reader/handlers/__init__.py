"""
File format handlers for the document reader.

Each handler is responsible for reading a specific type of file format.
All handlers inherit from BaseFileHandler and implement the same interface.
"""

from .base import BaseFileHandler
from .pdf import PDFHandler
from .office import DocxHandler, ExcelHandler, PowerPointHandler
from .text import TextHandler
from .web import HTMLHandler
from .notebook import JupyterHandler, LaTeXHandler

# Registry of all available handlers
HANDLER_CLASSES = [
    PDFHandler,
    DocxHandler,
    ExcelHandler,
    PowerPointHandler,
    HTMLHandler,
    TextHandler,
    JupyterHandler,
    LaTeXHandler,
]

__all__ = [
    "BaseFileHandler",
    "PDFHandler",
    "DocxHandler",
    "ExcelHandler",
    "PowerPointHandler",
    "HTMLHandler",
    "TextHandler",
    "JupyterHandler",
    "LaTeXHandler",
    "HANDLER_CLASSES",
]
