import hashlib
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class randomEmbedder:
    """
    A configurable Random embedding generator for prototyping and testing.
    """
    def __init__(self, dim: int = 128, seed: Optional[int] = None,
                 deterministic: bool = True, normalize: bool = True):
        """
        Initialize the RandomEmbedder.

        Args:
        dim (int): Embedding dimension. Must be > 0.
        seed (int, optional): Seed for random generation.
        deterministic (bool): Use deterministic hashing or random.
        normalize (bool): Normalize output to unit norm.
        """
        if dim <= 0:
            raise ValueError("Embedding dimension must be > 0")

        self.dim = dim
        self.seed = seed
        self.deterministic = deterministic
        self.normalize = normalize

        self.rng = np.random.default_rng(seed)

    def hash_to_vector(self, text: str) -> np.ndarray:
        """
        Deterministically convert a text string into a vector using SHA256 hashing.
        """
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

      
        hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)

        
        repeats = int(np.ceil(self.dim / len(hash_array)))
        full_array = np.tile(hash_array, repeats)[: self.dim]

        
        vector = full_array.astype(np.float32) / 255.0

        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def generate_vector(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector, either deterministically or randomly.
        """
        if self.deterministic:
            return self.hash_to_vector(text)

        # Random vector generation
        vector = self.rng.random(self.dim).astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of input texts.
        """
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings")

        vectors = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Each item in texts must be a string")

            vec = self.generate_vector(text)
            vectors.append(vec)

        return np.vstack(vectors)

