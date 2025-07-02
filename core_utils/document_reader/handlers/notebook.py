from typing import Union, List
from .base import BaseFileHandler
from ..exceptions import MissingDependencyError, FileProcessingError
import io

try:
    import nbformat
except ImportError:
    nbformat = None

try:
    import pylatexenc.latex2text
except ImportError:
    pylatexenc = None


class JupyterHandler(BaseFileHandler):
    """Handler for Jupyter Notebook files."""

    def can_handle(self, file_extension: str) -> bool:
        return file_extension == ".ipynb"

    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        if nbformat is None:
            raise MissingDependencyError(
                "nbformat is required to read Jupyter Notebook files"
            )

        try:
            if hasattr(file, "read"):
                file.seek(0)
                if isinstance(file, io.TextIOBase):
                    notebook_content = nbformat.read(file, as_version=4)
                else:
                    content = file.read().decode("utf-8", errors="ignore")
                    notebook_content = nbformat.reads(content, as_version=4)
            else:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    notebook_content = nbformat.read(f, as_version=4)

            # Extract text content from cells
            content_parts = []
            for cell in notebook_content.cells:
                if cell.cell_type == "markdown":
                    content_parts.append(cell.source)
                elif cell.cell_type == "code":
                    content_parts.append(f"```python\n{cell.source}\n```")

            content = "\n\n".join(content_parts)
            return [content] if return_pages else content
        except Exception as e:
            raise FileProcessingError(f"Error reading Jupyter Notebook file: {e}")


class LaTeXHandler(BaseFileHandler):
    """Handler for LaTeX files."""

    def can_handle(self, file_extension: str) -> bool:
        return file_extension == ".tex"

    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        if pylatexenc is None:
            raise MissingDependencyError("pylatexenc is required to read LaTeX files")

        try:
            content = self._read_file_content(file)

            # Convert LaTeX to plain text
            latex_converter = pylatexenc.latex2text.LatexNodes2Text()
            plain_text = latex_converter.latex_to_text(content)
            return [plain_text] if return_pages else plain_text
        except Exception as e:
            raise FileProcessingError(f"Error reading LaTeX file: {e}")
