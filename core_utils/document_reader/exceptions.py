class FileReaderError(Exception):
    """Base exception for FileReader errors."""

    pass


class UnsupportedFileTypeError(FileReaderError):
    """Raised when attempting to read an unsupported file type."""

    pass


class FileProcessingError(FileReaderError):
    """Raised when there's an error processing a file."""

    pass


class MissingDependencyError(FileReaderError):
    """Raised when a required dependency is missing."""

    pass
