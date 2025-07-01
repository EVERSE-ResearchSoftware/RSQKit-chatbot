from typing import Union, List
from .base import BaseFileHandler
from ..exceptions import MissingDependencyError, FileProcessingError
import io

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class HTMLHandler(BaseFileHandler):
    """Handler for HTML files."""

    def can_handle(self, file_extension: str) -> bool:
        return file_extension == ".html"

    def read_content(
        self, file: Union[str, io.IOBase], return_pages: bool = False
    ) -> Union[str, List[str]]:
        if BeautifulSoup is None:
            raise MissingDependencyError(
                "beautifulsoup4 is required to read HTML files"
            )

        try:
            content = self._read_file_content(file)
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return [text] if return_pages else text
        except Exception as e:
            raise FileProcessingError(f"Error reading HTML file: {e}")
