from typing import Union, List
import io
from .base import BaseFileHandler
from ..exceptions import MissingDependencyError, FileProcessingError

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFHandler(BaseFileHandler):
    """Handler for PDF files using PyMuPDF."""

    def can_handle(self, file_extension: str) -> bool:
        return file_extension == ".pdf"

    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        if fitz is None:
            raise MissingDependencyError("PyMuPDF (fitz) is required to read PDF files")

        content = []
        try:
            if hasattr(file, "read"):
                file.seek(0)
                with fitz.open(stream=file.read(), filetype="pdf") as doc:
                    content = [page.get_text() for page in doc]
            else:
                with fitz.open(file) as doc:
                    content = [page.get_text() for page in doc]
        except Exception as e:
            raise FileProcessingError(f"Error reading PDF file: {e}")

        return content if return_pages else "\n".join(content)
