"""
Integration tests for the document_reader package.

These tests focus on testing the complete workflow with real files and
the full FileReader class functionality, ensuring all components work
together correctly.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Import the main class we're testing
from core_utils.document_reader.reader import FileReader
from core_utils.document_reader.exceptions import (
    FileReaderError,
    UnsupportedFileTypeError,
    FileProcessingError,
    MissingDependencyError,
)


class TestFileReaderInitialization:
    """Integration tests for FileReader initialization and configuration."""

    def test_file_reader_initialization_default(self):
        """Test FileReader initialization with default settings."""
        reader = FileReader()

        assert reader.current_content is None
        assert reader.current_file_type is None
        assert len(reader.handlers) == 8  # All handlers loaded
        assert len(reader.supported_formats) == 12  # All formats supported

    def test_file_reader_initialization_no_logging(self):
        """Test FileReader initialization with logging disabled."""
        reader = FileReader(enable_logging=False)

        assert reader.current_content is None
        assert reader.current_file_type is None
        assert len(reader.handlers) == 8

    def test_supported_formats_complete(self):
        """Test that all expected formats are supported."""
        reader = FileReader(enable_logging=False)

        expected_formats = {
            ".pdf",
            ".docx",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".html",
            ".py",
            ".json",
            ".txt",
            ".ipynb",
            ".tex",
            ".md",
        }
        actual_formats = set(reader.supported_formats.keys())

        assert actual_formats == expected_formats

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        reader = FileReader(enable_logging=False)
        extensions = reader.get_supported_extensions()

        assert isinstance(extensions, list)
        assert len(extensions) == 12
        assert ".pdf" in extensions
        assert ".txt" in extensions


class TestFileTypeDetectionIntegration:
    """Integration tests for file type detection and validation."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    def test_is_supported_file_various_cases(self, file_reader):
        """Test file support detection with various cases."""
        # Supported files
        assert file_reader.is_supported_file("document.pdf")
        assert file_reader.is_supported_file("DOCUMENT.PDF")  # Case insensitive
        assert file_reader.is_supported_file("path/to/file.docx")
        assert file_reader.is_supported_file("file.json")

        # Unsupported files
        assert not file_reader.is_supported_file("image.png")
        assert not file_reader.is_supported_file("archive.zip")
        assert not file_reader.is_supported_file("video.mp4")

    def test_detect_file_type_integration(self, file_reader):
        """Test file type detection integration."""
        # Test with various file paths
        assert file_reader._detect_file_type("test.pdf") == ".pdf"
        assert file_reader._detect_file_type("TEST.DOCX") == ".docx"
        assert file_reader._detect_file_type("/path/to/file.json") == ".json"

        # Check that current_file_type is updated
        file_reader._detect_file_type("example.txt")
        assert file_reader.current_file_type == ".txt"

    def test_detect_unsupported_file_types(self, file_reader):
        """Test detection of unsupported file types."""
        unsupported_files = [
            "image.png",
            "video.mp4",
            "archive.zip",
            "database.db",
            "executable.exe",
            "unknown.xyz",
        ]

        for file_path in unsupported_files:
            with pytest.raises(UnsupportedFileTypeError):
                file_reader._detect_file_type(file_path)

    def test_detect_file_type_no_extension(self, file_reader):
        """Test handling of files without extensions."""
        files_without_ext = ["README", "Makefile", "dockerfile", "file_no_ext"]

        for file_path in files_without_ext:
            with pytest.raises(UnsupportedFileTypeError):
                file_reader._detect_file_type(file_path)


class TestTextFileIntegration:
    """Integration tests for text file processing."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_read_plain_text_file(self, file_reader, temp_dir):
        """Test reading plain text files end-to-end."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello World\nThis is a test file.\nWith multiple lines."
        test_file.write_text(test_content, encoding="utf-8")

        content = file_reader.get_content(str(test_file))

        assert content == test_content
        assert file_reader.current_content == test_content
        assert file_reader.current_file_type == ".txt"

    def test_read_python_file(self, file_reader, temp_dir):
        """Test reading Python files end-to-end."""
        test_file = temp_dir / "script.py"
        test_content = '''def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
'''
        test_file.write_text(test_content, encoding="utf-8")

        content = file_reader.get_content(str(test_file))

        assert content == test_content
        assert "def hello_world():" in content
        assert file_reader.current_file_type == ".py"

    def test_read_json_file_with_formatting(self, file_reader, temp_dir):
        """Test reading and formatting JSON files."""
        test_file = temp_dir / "data.json"
        test_data = {
            "users": [
                {"name": "John Doe", "age": 30, "city": "New York"},
                {"name": "Jane Smith", "age": 25, "city": "San Francisco"},
            ],
            "metadata": {"version": "1.0", "created": "2024-01-01"},
        }
        # Write compact JSON
        test_file.write_text(json.dumps(test_data), encoding="utf-8")

        content = file_reader.get_content(str(test_file))

        # Content should be formatted (not compact)
        assert content != json.dumps(test_data)
        assert "John Doe" in content
        assert "Jane Smith" in content
        assert "metadata" in content
        # Should be indented
        assert "{\n" in content
        assert file_reader.current_file_type == ".json"

    def test_read_markdown_file(self, file_reader, temp_dir):
        """Test reading Markdown files."""
        test_file = temp_dir / "README.md"
        test_content = """# My Project

This is a sample **markdown** file with:

- List items
- *Italic text*
- `code snippets`

## Section 2

More content here.
"""
        test_file.write_text(test_content, encoding="utf-8")

        content = file_reader.get_content(str(test_file))

        assert content == test_content
        assert "# My Project" in content
        assert "**markdown**" in content
        assert file_reader.current_file_type == ".md"

    def test_read_text_files_with_pages(self, file_reader, temp_dir):
        """Test reading text files with page separation."""
        test_file = temp_dir / "test.txt"
        test_content = "Single page content for text file"
        test_file.write_text(test_content, encoding="utf-8")

        pages = file_reader.get_pages(str(test_file))

        # Text files return content as single page
        assert pages == [test_content]

    def test_read_files_with_special_encoding(self, file_reader, temp_dir):
        """Test reading files with special characters."""
        test_file = temp_dir / "special.txt"
        test_content = "Café résumé naïve fiancé 中文 العربية 🚀"
        test_file.write_text(test_content, encoding="utf-8")

        content = file_reader.get_content(str(test_file))

        assert content == test_content
        assert "Café" in content
        assert "中文" in content
        assert "🚀" in content


class TestFileReaderErrorHandling:
    """Integration tests for error handling scenarios."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_file_not_found_error(self, file_reader):
        """Test handling of non-existent files."""
        # Fix: Expect FileProcessingError instead of FileNotFoundError
        with pytest.raises(FileProcessingError):
            file_reader.get_content("nonexistent_file.txt")

        with pytest.raises(FileProcessingError):
            file_reader.get_pages("another_nonexistent.json")

    def test_unsupported_file_type_error(self, file_reader, temp_dir):
        """Test handling of unsupported file types."""
        # Create files with unsupported extensions
        unsupported_files = ["image.png", "video.mp4", "archive.zip", "database.db"]

        for filename in unsupported_files:
            test_file = temp_dir / filename
            test_file.write_text("Some content")

            with pytest.raises(UnsupportedFileTypeError):
                file_reader.get_content(str(test_file))

    def test_corrupted_json_handling(self, file_reader, temp_dir):
        """Test handling of corrupted JSON files."""
        test_file = temp_dir / "corrupted.json"
        # Write invalid JSON
        test_file.write_text('{"name": "John", "age": invalid_value}', encoding="utf-8")

        # Should not raise an exception, just return original content
        content = file_reader.get_content(str(test_file))
        assert "invalid_value" in content

    def test_empty_file_handling(self, file_reader, temp_dir):
        """Test handling of empty files."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        content = file_reader.get_content(str(test_file))
        assert content == ""

    def test_very_large_content_handling(self, file_reader, temp_dir):
        """Test handling of large file content."""
        test_file = temp_dir / "large.txt"
        # Create a large content string (1MB)
        large_content = "A" * (1024 * 1024)
        test_file.write_text(large_content, encoding="utf-8")

        content = file_reader.get_content(str(test_file))
        assert len(content) == 1024 * 1024
        assert content == large_content


class TestDirectoryProcessingIntegration:
    """Integration tests for directory processing functionality."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    @pytest.fixture
    def sample_directory(self, temp_dir):
        """Create a comprehensive directory structure for testing."""
        # Root level files
        (temp_dir / "readme.txt").write_text("Project README", encoding="utf-8")
        (temp_dir / "config.json").write_text('{"debug": true}', encoding="utf-8")
        (temp_dir / "script.py").write_text("print('hello')", encoding="utf-8")
        (temp_dir / "notes.md").write_text("# Notes\nSome notes", encoding="utf-8")

        # Unsupported files (should be ignored)
        (temp_dir / "image.png").write_bytes(b"fake png data")
        (temp_dir / "archive.zip").write_bytes(b"fake zip data")

        # Subdirectory with more files
        subdir = temp_dir / "documents"
        subdir.mkdir()
        (subdir / "report.txt").write_text("Report content", encoding="utf-8")
        (subdir / "data.json").write_text('{"results": [1,2,3]}', encoding="utf-8")

        # Nested subdirectory
        nested_dir = subdir / "archived"
        nested_dir.mkdir()
        (nested_dir / "old_notes.md").write_text("Old notes", encoding="utf-8")

        # Empty subdirectory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        return temp_dir

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_read_directory_recursive(self, file_reader, sample_directory):
        """Test reading directory recursively."""
        documents = file_reader.read_directory(sample_directory, recursive=True)

        # Should find 7 supported files total
        assert len(documents) == 7

        # Extract file paths and contents
        file_paths = [doc[0] for doc in documents]
        file_contents = [doc[1] for doc in documents]

        # Check that all supported files are found
        expected_files = [
            "readme.txt",
            "config.json",
            "script.py",
            "notes.md",
            "report.txt",
            "data.json",
            "old_notes.md",
        ]

        for expected_file in expected_files:
            assert any(expected_file in path for path in file_paths)

        # Check that unsupported files are not included
        assert not any("image.png" in path for path in file_paths)
        assert not any("archive.zip" in path for path in file_paths)

        # Verify some content
        assert "Project README" in file_contents
        assert "print('hello')" in file_contents
        assert "Old notes" in file_contents

    def test_read_directory_non_recursive(self, file_reader, sample_directory):
        """Test reading directory non-recursively."""
        documents = file_reader.read_directory(sample_directory, recursive=False)

        # Should find only 4 files in root directory
        assert len(documents) == 4

        file_paths = [doc[0] for doc in documents]

        # Should include root level files
        root_files = ["readme.txt", "config.json", "script.py", "notes.md"]
        for root_file in root_files:
            assert any(root_file in path for path in file_paths)

        # Should not include subdirectory files
        subdirectory_files = ["report.txt", "data.json", "old_notes.md"]
        for subdir_file in subdirectory_files:
            assert not any(subdir_file in path for path in file_paths)

    def test_read_directory_with_metadata(self, file_reader, sample_directory):
        """Test reading directory with metadata included."""
        documents = file_reader.read_directory(
            sample_directory, recursive=True, include_metadata=True
        )

        assert len(documents) == 7

        # Each document should have 3 elements: (path, content, metadata)
        for doc in documents:
            assert len(doc) == 3
            file_path, content, metadata = doc

            # Check metadata structure
            assert isinstance(metadata, dict)
            assert "size" in metadata
            assert "modified" in metadata
            assert "created" in metadata
            assert "extension" in metadata
            assert "name" in metadata

            # Verify metadata values
            assert metadata["size"] >= 0
            assert metadata["extension"] in [".txt", ".json", ".py", ".md"]
            assert metadata["name"] == Path(file_path).name

    def test_read_directory_error_handling(self, file_reader):
        """Test directory reading error scenarios."""
        # Non-existent directory
        with pytest.raises(FileNotFoundError):
            file_reader.read_directory("non_existent_directory")

        # Try to read a file as directory
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            with pytest.raises(ValueError, match="Path is not a directory"):
                file_reader.read_directory(temp_file_path)
        finally:
            Path(temp_file_path).unlink()  # Clean up

    def test_read_directory_with_processing_errors(self, file_reader, temp_dir):
        """Test directory reading when some files cause processing errors."""
        # Create valid files
        (temp_dir / "valid.txt").write_text("Valid content", encoding="utf-8")
        (temp_dir / "also_valid.json").write_text('{"valid": true}', encoding="utf-8")

        # Create a file that might cause issues (binary content with text extension)
        problematic_file = temp_dir / "problematic.txt"
        with open(problematic_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")  # Binary data

        # Should handle gracefully and continue processing other files
        documents = file_reader.read_directory(temp_dir)

        # Should get at least the valid files
        assert len(documents) >= 2

        file_contents = [doc[1] for doc in documents]
        assert "Valid content" in file_contents


class TestFileInfoIntegration:
    """Integration tests for file information functionality."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_get_file_info_existing_supported_file(self, file_reader, temp_dir):
        """Test getting file info for existing supported files."""
        test_file = temp_dir / "test_document.txt"
        test_content = "This is a test document with some content."
        test_file.write_text(test_content, encoding="utf-8")

        info = file_reader.get_file_info(str(test_file))

        assert info["extension"] == ".txt"
        assert info["supported"] is True
        assert info["description"] == "Plain Text"
        assert info["size"] == len(test_content.encode("utf-8"))
        assert info["name"] == "test_document.txt"
        assert "modified" in info
        assert "created" in info

    def test_get_file_info_different_file_types(self, file_reader, temp_dir):
        """Test getting file info for different supported file types."""
        file_types = {
            "document.pdf": (".pdf", "Portable Document Format"),
            "spreadsheet.xlsx": (".xlsx", "Microsoft Excel Spreadsheet"),
            "presentation.pptx": (".pptx", "Microsoft PowerPoint Presentation"),
            "webpage.html": (".html", "HyperText Markup Language"),
            "notebook.ipynb": (".ipynb", "Jupyter Notebook"),
        }

        for filename, (expected_ext, expected_desc) in file_types.items():
            test_file = temp_dir / filename
            test_file.write_text("test content", encoding="utf-8")

            info = file_reader.get_file_info(str(test_file))

            assert info["extension"] == expected_ext
            assert info["supported"] is True
            assert info["description"] == expected_desc

    def test_get_file_info_unsupported_file(self, file_reader):
        """Test getting file info for unsupported files."""
        unsupported_files = ["image.png", "video.mp4", "archive.zip", "executable.exe"]

        for filename in unsupported_files:
            info = file_reader.get_file_info(filename)

            assert info["extension"] == "unknown"
            assert info["supported"] is False
            assert info["description"] == "Unsupported file type"

    def test_get_file_info_nonexistent_file(self, file_reader):
        """Test getting file info for non-existent files."""
        info = file_reader.get_file_info("nonexistent.txt")

        # Should still detect the type based on extension
        assert info["extension"] == ".txt"
        assert info["supported"] is True
        assert info["description"] == "Plain Text"
        # But should not have file system metadata
        assert "size" not in info
        assert "modified" not in info


class TestWorkflowIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def file_reader(self):
        return FileReader(enable_logging=False)

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_rag_document_processing_workflow(self, file_reader, temp_dir):
        """Test a complete RAG document processing workflow."""
        # Simulate a typical RAG ingestion scenario

        # 1. Create various document types
        documents = {
            "research_paper.txt": "Abstract: This paper discusses...\n\nIntroduction: The field of...",
            "data_analysis.py": "import pandas as pd\n\ndef analyze_data():\n    return results",
            "config.json": '{"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}',
            "notes.md": "# Research Notes\n\n## Key Findings\n- Finding 1\n- Finding 2",
        }

        document_paths = []
        for filename, content in documents.items():
            file_path = temp_dir / filename
            file_path.write_text(content, encoding="utf-8")
            document_paths.append(str(file_path))

        # 2. Process each document
        processed_documents = []
        for doc_path in document_paths:
            # Check if supported
            if file_reader.is_supported_file(doc_path):
                # Get file info
                file_info = file_reader.get_file_info(doc_path)

                # Extract content
                content = file_reader.get_content(doc_path)

                processed_documents.append(
                    {
                        "path": doc_path,
                        "type": file_info["extension"],
                        "description": file_info["description"],
                        "content": content,
                        "size": len(content),
                    }
                )

        # 3. Verify processing results
        assert len(processed_documents) == 4

        # Check each document type was processed correctly
        types_found = {doc["type"] for doc in processed_documents}
        assert types_found == {".txt", ".py", ".json", ".md"}

        # Verify content extraction
        contents = [doc["content"] for doc in processed_documents]
        assert any("This paper discusses" in content for content in contents)
        assert any("import pandas" in content for content in contents)
        assert any("gpt-4" in content for content in contents)
        assert any("# Research Notes" in content for content in contents)

    def test_batch_processing_workflow(self, file_reader, temp_dir):
        """Test batch processing multiple files with error handling."""
        # Create a mix of valid and problematic files
        files_to_create = [
            ("valid1.txt", "Valid content 1"),
            ("valid2.json", '{"status": "ok"}'),
            ("valid3.py", "print('hello')"),
            ("unsupported.png", "fake image data"),  # Will be skipped
            ("empty.txt", ""),  # Valid but empty
        ]

        created_files = []
        for filename, content in files_to_create:
            file_path = temp_dir / filename
            file_path.write_text(content, encoding="utf-8")
            created_files.append(str(file_path))

        # Process files with error handling
        successful_docs = []
        failed_docs = []

        for file_path in created_files:
            try:
                if file_reader.is_supported_file(file_path):
                    content = file_reader.get_content(file_path)
                    successful_docs.append({"path": file_path, "content": content})
                else:
                    failed_docs.append({"path": file_path, "reason": "unsupported"})
            except Exception as e:
                failed_docs.append({"path": file_path, "reason": str(e)})

        # Verify results
        assert len(successful_docs) == 4  # 4 supported files
        assert len(failed_docs) == 1  # 1 unsupported file

        # Check successful processing
        successful_contents = [doc["content"] for doc in successful_docs]
        assert "Valid content 1" in successful_contents
        # Fix: Check for formatted JSON string instead of exact match
        assert any(
            '"status": "ok"' in content for content in successful_contents
        )  # Should be formatted
        assert "print('hello')" in successful_contents
        assert "" in successful_contents  # Empty file

    def test_memory_efficient_large_directory_processing(self, file_reader, temp_dir):
        """Test processing a large number of files efficiently."""
        # Create many small files
        num_files = 50

        for i in range(num_files):
            file_path = temp_dir / f"file_{i:03d}.txt"
            file_path.write_text(f"Content of file {i}", encoding="utf-8")

        # Process directory
        documents = file_reader.read_directory(temp_dir, recursive=True)

        assert len(documents) == num_files

        # Verify content - Fix: Check for any file content instead of specific file
        contents = [doc[1] for doc in documents]
        assert any("Content of file" in content for content in contents)
        # Check that we have files with different numbers
        content_numbers = []
        for content in contents:
            if "Content of file" in content:
                # Extract the number from the content
                parts = content.split("Content of file ")
                if len(parts) > 1:
                    try:
                        num = int(parts[1])
                        content_numbers.append(num)
                    except ValueError:
                        pass

        # Should have files numbered 0 through num_files-1
        assert len(content_numbers) == num_files
        assert min(content_numbers) == 0
        assert max(content_numbers) == num_files - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
