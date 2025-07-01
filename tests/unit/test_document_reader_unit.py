"""
Unit tests for the document_reader package.

These tests focus on testing individual components in isolation,
using mocks to avoid dependencies on external libraries and file system.
"""

import pytest
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the classes we're testing
from core_utils.document_reader.exceptions import (
    FileReaderError,
    UnsupportedFileTypeError,
    FileProcessingError,
    MissingDependencyError,
)
from core_utils.document_reader.handlers.pdf import PDFHandler
from core_utils.document_reader.handlers.office import (
    DocxHandler,
    ExcelHandler,
    PowerPointHandler,
)
from core_utils.document_reader.handlers.text import TextHandler
from core_utils.document_reader.handlers.web import HTMLHandler
from core_utils.document_reader.handlers.notebook import JupyterHandler, LaTeXHandler
from core_utils.document_reader.handlers.base import BaseFileHandler


class TestBaseFileHandler:
    """Unit tests for BaseFileHandler."""

    def test_is_abstract_class(self):
        """Test that BaseFileHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFileHandler()

    def test_decode_bytes_utf8(self):
        """Test successful UTF-8 decoding."""
        # Create a concrete handler to test the base methods
        handler = TextHandler()

        content_bytes = "Hello World".encode("utf-8")
        result = handler._decode_bytes(content_bytes)
        assert result == "Hello World"

    def test_decode_bytes_fallback(self):
        """Test fallback encoding when UTF-8 fails."""
        handler = TextHandler()

        # Create bytes that will fail UTF-8 decoding
        content_bytes = b"\xff\xfe\x48\x00\x65\x00\x6c\x00\x6c\x00\x6f\x00"
        result = handler._decode_bytes(content_bytes)
        assert isinstance(result, str)  # Should return string, even if garbled


class TestTextHandler:
    """Unit tests for TextHandler."""

    @pytest.fixture
    def text_handler(self):
        return TextHandler()

    def test_can_handle_supported_extensions(self, text_handler):
        """Test that handler recognizes supported extensions."""
        assert text_handler.can_handle(".txt")
        assert text_handler.can_handle(".py")
        assert text_handler.can_handle(".json")
        assert text_handler.can_handle(".md")

    def test_can_handle_unsupported_extensions(self, text_handler):
        """Test that handler rejects unsupported extensions."""
        assert not text_handler.can_handle(".pdf")
        assert not text_handler.can_handle(".docx")
        assert not text_handler.can_handle(".xlsx")

    def test_get_file_extension_from_string(self, text_handler):
        """Test extracting extension from string path."""
        assert text_handler._get_file_extension("test.json") == ".json"
        assert text_handler._get_file_extension("path/to/file.py") == ".py"
        assert text_handler._get_file_extension("FILE.TXT") == ".txt"

    def test_get_file_extension_from_file_object(self, text_handler):
        """Test extracting extension from file object."""
        mock_file = Mock()
        mock_file.name = "test.json"
        assert text_handler._get_file_extension(mock_file) == ".json"

    def test_get_file_extension_no_extension(self, text_handler):
        """Test handling files without extensions."""
        assert text_handler._get_file_extension("noextension") == ""

        mock_file = Mock()
        mock_file.name = "noextension"
        assert text_handler._get_file_extension(mock_file) == ""

    def test_format_json_valid(self, text_handler):
        """Test JSON formatting with valid JSON."""
        json_string = '{"name":"John","age":30}'
        result = text_handler._format_json(json_string)

        # Should be formatted (indented)
        assert "{\n" in result
        assert '"name"' in result
        assert '"John"' in result

    def test_format_json_invalid(self, text_handler):
        """Test JSON formatting with invalid JSON."""
        invalid_json = '{"name": invalid}'
        result = text_handler._format_json(invalid_json)

        # Should return original content when JSON is invalid
        assert result == invalid_json

    def test_read_content_string_io(self, text_handler):
        """Test reading from StringIO object."""
        content = "Hello from StringIO"
        file_obj = io.StringIO(content)
        file_obj.name = "test.txt"

        result = text_handler.read_content(file_obj)
        assert result == content

    def test_read_content_bytes_io(self, text_handler):
        """Test reading from BytesIO object."""
        content = "Hello from BytesIO"
        file_obj = io.BytesIO(content.encode("utf-8"))
        file_obj.name = "test.txt"

        result = text_handler.read_content(file_obj)
        assert result == content

    def test_read_content_json_formatting(self, text_handler):
        """Test that JSON files get formatted."""
        json_data = {"test": "data"}
        file_obj = io.StringIO(json.dumps(json_data))
        file_obj.name = "test.json"

        result = text_handler.read_content(file_obj)
        # Should be formatted, not the original compact JSON
        assert result != json.dumps(json_data)
        assert "test" in result
        assert "data" in result

    def test_read_content_return_pages(self, text_handler):
        """Test reading content with return_pages=True."""
        content = "Test content"
        file_obj = io.StringIO(content)
        file_obj.name = "test.txt"

        result = text_handler.read_content(file_obj, return_pages=True)
        assert result == [content]


class TestHTMLHandler:
    """Unit tests for HTMLHandler."""

    @pytest.fixture
    def html_handler(self):
        return HTMLHandler()

    def test_can_handle(self, html_handler):
        """Test HTML file type detection."""
        assert html_handler.can_handle(".html")
        assert not html_handler.can_handle(".txt")
        assert not html_handler.can_handle(".htm")  # Only .html supported

    @patch("core_utils.document_reader.handlers.web.BeautifulSoup")
    def test_read_content_with_beautifulsoup(self, mock_bs, html_handler):
        """Test reading HTML content with BeautifulSoup available."""
        # Setup mock
        mock_soup = Mock()
        mock_soup.get_text.return_value = "Extracted text"
        mock_bs.return_value = mock_soup

        # Test with StringIO
        html_content = "<html><body><h1>Title</h1></body></html>"
        file_obj = io.StringIO(html_content)
        file_obj.name = "test.html"

        result = html_handler.read_content(file_obj)

        assert result == "Extracted text"
        mock_bs.assert_called_once_with(html_content, "html.parser")
        mock_soup.get_text.assert_called_once_with(separator="\n", strip=True)

    @patch("core_utils.document_reader.handlers.web.BeautifulSoup")
    def test_read_content_return_pages(self, mock_bs, html_handler):
        """Test reading HTML content with return_pages=True."""
        mock_soup = Mock()
        mock_soup.get_text.return_value = "Extracted text"
        mock_bs.return_value = mock_soup

        file_obj = io.StringIO("<html><body>Test</body></html>")
        file_obj.name = "test.html"

        result = html_handler.read_content(file_obj, return_pages=True)
        assert result == ["Extracted text"]

    @patch("core_utils.document_reader.handlers.web.BeautifulSoup", None)
    def test_missing_beautifulsoup_dependency(self, html_handler):
        """Test behavior when BeautifulSoup is not available."""
        file_obj = io.StringIO("<html><body>Test</body></html>")
        file_obj.name = "test.html"

        with pytest.raises(MissingDependencyError, match="beautifulsoup4 is required"):
            html_handler.read_content(file_obj)


class TestPDFHandler:
    """Unit tests for PDFHandler."""

    @pytest.fixture
    def pdf_handler(self):
        return PDFHandler()

    def test_can_handle(self, pdf_handler):
        """Test PDF file type detection."""
        assert pdf_handler.can_handle(".pdf")
        assert not pdf_handler.can_handle(".txt")
        assert not pdf_handler.can_handle(".PDF")  # Case sensitive in handler

    @patch("core_utils.document_reader.handlers.pdf.fitz")
    def test_read_content_from_file_object(self, mock_fitz, pdf_handler):
        """Test reading PDF from file object."""
        # Setup mocks
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2 content"

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz.open.return_value = mock_doc

        # Test with file object
        file_obj = io.BytesIO(b"fake pdf content")
        file_obj.name = "test.pdf"

        result = pdf_handler.read_content(file_obj)

        assert result == "Page 1 content\nPage 2 content"
        mock_fitz.open.assert_called_once()

        # Verify stream was used
        call_args = mock_fitz.open.call_args
        assert "stream" in call_args.kwargs
        assert call_args.kwargs["filetype"] == "pdf"

    @patch("core_utils.document_reader.handlers.pdf.fitz")
    def test_read_content_from_file_path(self, mock_fitz, pdf_handler):
        """Test reading PDF from file path."""
        # Setup mocks
        mock_page = Mock()
        mock_page.get_text.return_value = "Page content"

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz.open.return_value = mock_doc

        result = pdf_handler.read_content("test.pdf")

        assert result == "Page content"
        mock_fitz.open.assert_called_once_with("test.pdf")

    @patch("core_utils.document_reader.handlers.pdf.fitz")
    def test_read_content_return_pages(self, mock_fitz, pdf_handler):
        """Test reading PDF with return_pages=True."""
        # Setup mocks
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1"
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "Page 2"

        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)

        mock_fitz.open.return_value = mock_doc

        result = pdf_handler.read_content("test.pdf", return_pages=True)

        assert result == ["Page 1", "Page 2"]

    @patch("core_utils.document_reader.handlers.pdf.fitz")
    def test_read_content_fitz_exception(self, mock_fitz, pdf_handler):
        """Test handling of fitz exceptions."""
        mock_fitz.open.side_effect = Exception("PDF error")

        with pytest.raises(FileProcessingError, match="Error reading PDF file"):
            pdf_handler.read_content("test.pdf")

    @patch("core_utils.document_reader.handlers.pdf.fitz", None)
    def test_missing_fitz_dependency(self, pdf_handler):
        """Test behavior when PyMuPDF is not available."""
        with pytest.raises(MissingDependencyError, match="PyMuPDF.*is required"):
            pdf_handler.read_content("test.pdf")


class TestOfficeHandlers:
    """Unit tests for Office document handlers."""

    def test_docx_handler_can_handle(self):
        """Test DocxHandler file type detection."""
        handler = DocxHandler()
        assert handler.can_handle(".docx")
        assert not handler.can_handle(".doc")
        assert not handler.can_handle(".txt")

    def test_excel_handler_can_handle(self):
        """Test ExcelHandler file type detection."""
        handler = ExcelHandler()
        assert handler.can_handle(".xlsx")
        assert not handler.can_handle(".xls")
        assert not handler.can_handle(".csv")

    def test_powerpoint_handler_can_handle(self):
        """Test PowerPointHandler file type detection."""
        handler = PowerPointHandler()
        assert handler.can_handle(".ppt")
        assert handler.can_handle(".pptx")
        assert not handler.can_handle(".pdf")

    @patch("core_utils.document_reader.handlers.office.Document", None)
    def test_docx_missing_dependency(self):
        """Test DocxHandler with missing dependency."""
        handler = DocxHandler()

        with pytest.raises(MissingDependencyError, match="python-docx is required"):
            handler.read_content("test.docx")

    @patch("core_utils.document_reader.handlers.office.load_workbook", None)
    def test_excel_missing_dependency(self):
        """Test ExcelHandler with missing dependency."""
        handler = ExcelHandler()

        with pytest.raises(MissingDependencyError, match="openpyxl is required"):
            handler.read_content("test.xlsx")

    @patch("core_utils.document_reader.handlers.office.Presentation", None)
    def test_powerpoint_missing_dependency(self):
        """Test PowerPointHandler with missing dependency."""
        handler = PowerPointHandler()

        with pytest.raises(MissingDependencyError, match="python-pptx is required"):
            handler.read_content("test.pptx")

    @patch("core_utils.document_reader.handlers.office.Document")
    def test_docx_read_content(self, mock_document_class):
        """Test DocxHandler reading content."""
        # Setup mock
        mock_para1 = Mock()
        mock_para1.text = "First paragraph"
        mock_para2 = Mock()
        mock_para2.text = "Second paragraph"
        mock_para3 = Mock()
        mock_para3.text = ""  # Empty paragraph - should be filtered

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_document_class.return_value = mock_doc

        handler = DocxHandler()
        result = handler.read_content("test.docx")

        assert result == "First paragraph\nSecond paragraph"
        mock_document_class.assert_called_once_with("test.docx")

    @patch("core_utils.document_reader.handlers.office.Document")
    def test_docx_read_content_return_pages(self, mock_document_class):
        """Test DocxHandler reading content with return_pages=True."""
        mock_para = Mock()
        mock_para.text = "Paragraph content"

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        mock_document_class.return_value = mock_doc

        handler = DocxHandler()
        result = handler.read_content("test.docx", return_pages=True)

        assert result == ["Paragraph content"]


class TestNotebookHandlers:
    """Unit tests for Jupyter and LaTeX handlers."""

    def test_jupyter_handler_can_handle(self):
        """Test JupyterHandler file type detection."""
        handler = JupyterHandler()
        assert handler.can_handle(".ipynb")
        assert not handler.can_handle(".py")
        assert not handler.can_handle(".json")

    def test_latex_handler_can_handle(self):
        """Test LaTeXHandler file type detection."""
        handler = LaTeXHandler()
        assert handler.can_handle(".tex")
        assert not handler.can_handle(".txt")
        assert not handler.can_handle(".latex")

    @patch("core_utils.document_reader.handlers.notebook.nbformat", None)
    def test_jupyter_missing_dependency(self):
        """Test JupyterHandler with missing dependency."""
        handler = JupyterHandler()

        with pytest.raises(MissingDependencyError, match="nbformat is required"):
            handler.read_content("test.ipynb")

    @patch("core_utils.document_reader.handlers.notebook.pylatexenc", None)
    def test_latex_missing_dependency(self):
        """Test LaTeXHandler with missing dependency."""
        handler = LaTeXHandler()

        with pytest.raises(MissingDependencyError, match="pylatexenc is required"):
            handler.read_content("test.tex")

    @patch("core_utils.document_reader.handlers.notebook.nbformat")
    def test_jupyter_read_content(self, mock_nbformat):
        """Test JupyterHandler reading content."""
        # Setup mock cells
        mock_cell1 = Mock()
        mock_cell1.cell_type = "markdown"
        mock_cell1.source = "# Title\nMarkdown content"

        mock_cell2 = Mock()
        mock_cell2.cell_type = "code"
        mock_cell2.source = 'print("hello")\nx = 42'

        mock_cell3 = Mock()
        mock_cell3.cell_type = "raw"  # Should be ignored
        mock_cell3.source = "raw content"

        mock_notebook = Mock()
        mock_notebook.cells = [mock_cell1, mock_cell2, mock_cell3]
        mock_nbformat.read.return_value = mock_notebook

        with patch("builtins.open", return_value=mock_nbformat):
            handler = JupyterHandler()
            result = handler.read_content("test.ipynb")

        # Check that markdown and code cells are included
        assert "# Title" in result
        assert "Markdown content" in result
        assert "```python" in result
        assert 'print("hello")' in result
        assert "x = 42" in result
        # Raw cell should not be included
        assert "raw content" not in result

    @patch("core_utils.document_reader.handlers.notebook.pylatexenc")
    def test_latex_read_content(self, mock_pylatexenc):
        """Test LaTeXHandler reading content."""
        # Setup mock converter
        mock_converter = Mock()
        mock_converter.latex_to_text.return_value = "Converted plain text"
        mock_pylatexenc.latex2text.LatexNodes2Text.return_value = mock_converter

        # Test with file object
        latex_content = r"\documentclass{article}\begin{document}Hello\end{document}"
        file_obj = io.StringIO(latex_content)
        file_obj.name = "test.tex"

        handler = LaTeXHandler()
        result = handler.read_content(file_obj)

        assert result == "Converted plain text"
        mock_converter.latex_to_text.assert_called_once_with(latex_content)


class TestExceptionClasses:
    """Unit tests for custom exception classes."""

    def test_file_reader_error_inheritance(self):
        """Test FileReaderError is base exception."""
        error = FileReaderError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_unsupported_file_type_error_inheritance(self):
        """Test UnsupportedFileTypeError inherits from FileReaderError."""
        error = UnsupportedFileTypeError("Unsupported type")
        assert isinstance(error, FileReaderError)
        assert isinstance(error, Exception)

    def test_file_processing_error_inheritance(self):
        """Test FileProcessingError inherits from FileReaderError."""
        error = FileProcessingError("Processing failed")
        assert isinstance(error, FileReaderError)
        assert isinstance(error, Exception)

    def test_missing_dependency_error_inheritance(self):
        """Test MissingDependencyError inherits from FileReaderError."""
        error = MissingDependencyError("Missing dependency")
        assert isinstance(error, FileReaderError)
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
