"""
Defining the Embedder class and the OpenAIEmbedder class.
"""
from abc import ABC, abstractmethod
import os
from typing import List, Literal, Union
from functools import lru_cache
import numpy as np
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from openai import OpenAI


class Embedder(ABC, BaseModel):
    """Abstract Class for all Embedders"""

    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def are_embeddings_different(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> bool:
        pass


class OpenAIEmbedder(Embedder):
    """Wrapper around OpenAI APIs to get embeddings."""

    type: Literal["openaiembedder"] = "openaiembedder"  # Unique discriminator value
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL")
    # Define a private attributes
    _embedding_client: OpenAI = PrivateAttr()
    _get_embedding_cached: callable = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Instantiate the OpenAI client as a private attribute.
        self._embedding_client = OpenAI()
        # Wrap the internal _get_embedding method with lru_cache.
        self._get_embedding_cached = lru_cache(maxsize=128)(self._get_embedding)

    def _get_embedding(self, text: str) -> List[float]:
        """
        Internal method that retrieves the embedding for the given text from OpenAI's API.
        This method is intended to be wrapped with lru_cache.
        """
        response = self._embedding_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def get_embedding(self, text: str) -> List[float]:
        """
        Returns the embedding for the provided text using a cached call.
        """
        return self._get_embedding_cached(text)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Computes the cosine similarity between two vectors using NumPy.
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def are_embeddings_different(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ):
        return self._cosine_similarity(embedding1, embedding2) < self.similarity_threshold

    def is_embedding_different_from_list(
            self,
            embedding: List[float],
            list_embedding: List[List[float]],
        ) -> bool:
        """
        Checks if the embedding is different from all embeddings in the list.
        """
        return all(
            self.are_embeddings_different(embedding, other_embedding)
            for other_embedding in list_embedding
        )


EmbedderType = Union[OpenAIEmbedder]
