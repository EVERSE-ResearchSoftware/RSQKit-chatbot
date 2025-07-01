from dataclasses import dataclass
import re
from typing import Set, List, Union, Dict, Optional
from functools import lru_cache
from ..data_types.node import Node
import hashlib
import uuid


@dataclass
class ChunkConfig:
    """Configuration for text chunking parameters."""

    chunk_size: int = 1000
    chunk_overlap: int = 100
    num_page_after: int = 1
    page_num_shift: int = 1


try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


class TextChunker:
    """An optimized class for efficiently chunking text documents with configurable parameters."""

    def __init__(self, config: Optional[ChunkConfig] = None, **kwargs):
        """
        Initialize the TextChunker with configuration.

        Args:
            config (ChunkConfig, optional): Chunking configuration parameters.
        """
        if config is None and not kwargs:
            self.config = ChunkConfig()
        elif isinstance(config, ChunkConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = ChunkConfig(**config)
        else:
            self.config = ChunkConfig(**kwargs)
        self._sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    @lru_cache(maxsize=1024)
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing line breaks.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        return text  # text.replace("-\n", "")

    def _find_overlap(self, chunk: str, overlap: int, tokens: List[str]) -> str:
        chunk_words = set(chunk.split())
        tokens_words = [token for token in tokens if token in chunk_words]
        if tokens_words:
            res = []
            current_len = 0
            for word in tokens_words:
                if len(word) + current_len > overlap:
                    break
                else:
                    res.append(word)
                    current_len += len(word)
            return " ".join(res)
        else:
            return ""

    def create_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into overlapping chunks while preserving sentence boundaries.

        Args:
            text (str): Input text to chunk.
            chunk_size (int, optional): Override default chunk size.
            chunk_overlap (int, optional): Override default chunk overlap.

        Returns:
            List[str]: List of text chunks.
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap

        sentences = self._sentence_pattern.split(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                if chunk_overlap > 0:
                    overlap_tokens = current_chunk[
                        len(current_chunk) - chunk_overlap :
                    ].split()
                    overlap_text = self._find_overlap(
                        chunk=current_chunk,
                        overlap=self.config.chunk_overlap,
                        tokens=overlap_tokens,
                    )
                    current_chunk = (
                        f"{overlap_text} {sentence}"
                        if len(f"{overlap_text} {sentence}") <= chunk_size
                        else sentence
                    )
                else:
                    current_chunk = sentence
            else:
                current_chunk = f"{current_chunk}\n{sentence}".strip()

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def _is_subtext(self, chunk_words: Set[str], search_window_words: Set[str]) -> bool:
        """
        Efficiently check if all words in chunk are present in search_window_text.

        Args:
            chunk_words (Set[str]): Words in the chunk
            search_window_words (Set[str]): Words in the search window

        Returns:
            bool: True if all words in chunk are in search_window_text, False otherwise
        """
        return len(chunk_words - search_window_words) == 0

    def create_chunks_with_source_page(
        self,
        pages: List[str],
        return_nodes: bool = True,
        file_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Union[Dict[str, str], Node]]:
        """
        Create chunks with their corresponding source page ranges using an optimized approach.

        Args:
            pages (List[str]): List of text pages to chunk.
            return_nodes (bool, optional): If True, returns Node objects; otherwise, returns dictionaries.
            file_path (str, optional): Path to the source file.
            show_progress (bool, optional): If True, shows a progress bar during processing.

        Returns:
            List[Union[Dict[str, str], Node]]: List of chunks with page range information.
        """
        # Preprocess pages to reduce redundant operations
        all_pages = "\n\n".join(pages)
        chunks = self.create_chunks(text=all_pages)

        # Precompute page word sets for faster lookup
        page_word_sets = [set(page.split()) for page in pages]
        page_length = len(pages)

        final_chunks = []
        processed_chunks = set()

        # Use tqdm for progress tracking if show_progress is True
        chunks_iterator = tqdm(
            chunks, desc="Processing Chunks", disable=not show_progress
        )

        for chunk_index, chunk in enumerate(chunks_iterator):
            chunk_words = set(chunk.split())
            chunk_mapped = False

            # Sliding window approach with precomputed page word sets
            for page_num in range(page_length):
                # Create windows of 1-4 pages
                for window_size in range(1, min(5, page_length - page_num + 1)):
                    # Combine word sets of window pages
                    window_words = set().union(
                        *page_word_sets[page_num : page_num + window_size]
                    )

                    # Fast subtext check
                    if self._is_subtext(chunk_words, window_words):
                        # Generate unique ID using SHA-256 hash of chunk content and metadata
                        chunk_content = self.clean_text(chunk)
                        chunk_id = hashlib.sha256(
                            f"{chunk_content}_{page_num}_{page_num + window_size - 1}".encode()
                        ).hexdigest()

                        # Create chunk dict
                        chunk_dict = {
                            "id": chunk_id,
                            "content": chunk_content,
                            "page-start": page_num,
                            "page-end": page_num + window_size - 1,
                            "page-range": f"{page_num}-{page_num + window_size - 1}",
                            "chunkSource": f"localfile://{file_path}",
                            "description": "Unknown",
                            "docAuthor": "Unknown",
                            "docSource": "a text file uploaded by the user.",
                            "published": "27/03/2025 13:43:37",
                            "title": file_path.split("/")[-1],
                            "token_count_estimate": 1.5 * len(chunk_content.split()),
                            "url": "",
                            "wordCount": len(chunk_content.split()),
                            ## add metadata to match anything llm metadatas
                        }

                        # Add file path if provided
                        if file_path:
                            chunk_dict["file-path"] = file_path

                        # Add next and previous chunk references
                        if final_chunks:
                            chunk_dict["previous_id"] = final_chunks[-1]["id"]
                            final_chunks[-1]["next_id"] = chunk_id

                        # Use a tuple of dict values to check for uniqueness
                        chunk_key = (
                            chunk_dict["content"],
                            chunk_dict["page-start"],
                            chunk_dict["page-end"],
                        )

                        # Add only if not already processed
                        if chunk_key not in processed_chunks:
                            final_chunks.append(chunk_dict)
                            processed_chunks.add(chunk_key)
                            chunk_mapped = True
                            break

                # Stop searching if chunk is mapped
                if chunk_mapped:
                    break

        # Convert to Nodes if required
        if return_nodes:
            return [
                Node(
                    content=d["content"],
                    id=d["id"],
                    metadata={
                        key: item
                        for key, item in d.items()
                        if key not in ["content", "id"]
                    },
                )
                for d in final_chunks
            ]

        return final_chunks


if __name__ == "__main__":
    # Create with default configuration
    chunker = TextChunker()

    # Or with custom configuration
    custom_config = ChunkConfig(
        chunk_size=1024, chunk_overlap=100, num_page_after=2, page_num_shift=1
    )
    custom_chunker = TextChunker(custom_config)

    # Example usage
    sample_pages = [
        "This is page 1. It contains some text.",
        "This is page 2. More text here.",
        "This is page 3. Final page of text.",
    ]

    # Get chunks with default settings
    chunks = chunker.create_chunks_with_source_page(sample_pages)

    # Print results
    for chunk in chunks:
        print(f"Page range: {chunk.metadata['page-range']}")
        print(f"Content: {chunk.content}\n")
