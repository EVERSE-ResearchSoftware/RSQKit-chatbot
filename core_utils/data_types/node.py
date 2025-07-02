from dataclasses import dataclass
from typing import Dict, Optional, Any, TypeVar
import numpy as np


NodeType = TypeVar("NodeType", bound="Node")


@dataclass
class Node:
    """Represents a document node containing content and metadata."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Node instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the Node.
        """
        node_dict = {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": (
                self.embedding.tolist()
                if (
                    self.embedding is not None
                    and isinstance(self.embedding, np.ndarray)
                )
                else self.embedding if self.embedding else None
            ),
            "id": self.id,
        }
        return node_dict
