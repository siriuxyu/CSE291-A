"""
Embedding Service Implementation

Storage team can choose: OpenAI, Cohere, sentence-transformers, etc.
"""

from typing import List

from agent.interfaces import EmbeddingError


class OpenAIEmbeddingService:
    """Embedding service using OpenAI"""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536
    ):
        """Initialize OpenAI embedding service"""
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.client = None

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        raise NotImplementedError("TODO: Implement embed_text")

    async def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        raise NotImplementedError("TODO: Implement embed_texts_batch")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class SentenceTransformerEmbeddingService:
    """Embedding service using sentence-transformers (local)"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize sentence-transformers embedding service"""
        self.model_name = model_name
        self.model = None
        self.dimension = 384

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        raise NotImplementedError("TODO: Implement embed_text")

    async def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        raise NotImplementedError("TODO: Implement embed_texts_batch")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


def create_embedding_service(provider: str = "openai", **kwargs):
    """
    Factory function to create embedding service

    Args:
        provider: "openai" or "sentence-transformers"
        **kwargs: Provider-specific arguments

    Example:
        service = create_embedding_service("openai", api_key="sk-...")
        service = create_embedding_service("sentence-transformers")
    """
    if provider == "openai":
        return OpenAIEmbeddingService(**kwargs)
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingService(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
