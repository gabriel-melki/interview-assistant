"""
Integration tests for the OpenAI embedding.
"""

import os
import pytest
from dotenv import load_dotenv
from interview_assistant.core.ai.embeddings import OpenAIEmbedder

load_dotenv(override=True)


def test_openai_embedding():
    """
    Integration test that makes a real call to the OpenAI API to get an embedding.

    This test requires:
      - OPENAI_API_KEY to be set in your environment.
      - OPENAI_EMBEDDING_MODEL to be set in your environment. Otherwise use default.
    """
    # Check for the required API key and skip the test if not present.
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set. Skipping integration test.")

    # Use a known embedding model or fallback to a default.
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Instantiate the embedder with the desired model.
    embedder = OpenAIEmbedder(embedding_model=embedding_model)

    # Provide a sample text for embedding.
    text1 = "This is a real integration test for OpenAI embedding."
    text2 = "This is a real integration test for OpenAI embedding system."
    text3 = "Totally different sentence."

    # Make the call to get an embedding.
    embedding1 = embedder.get_embedding(text1)
    embedding2 = embedder.get_embedding(text2)
    embedding3 = embedder.get_embedding(text3)

    assert embedder.similarity_threshold is not None

    # Verify that the embedding is a list of floats.
    assert isinstance(embedding1, list), "Expected embedding to be a list."
    assert all(
        isinstance(x, float) for x in embedding1
    ), "All elements in the embedding should be floats."

    assert not embedder.are_embeddings_different(
        embedding1, embedding2
    ), "Embeddings should be the same."
    assert embedder.are_embeddings_different(
        embedding1, embedding3
    ), "Embeddings should be different."
    assert embedder.are_embeddings_different(
        embedding2, embedding3
    ), "Embeddings should be different."
