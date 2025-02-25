"""
Integration tests for the QuestionService and TipService.
"""

import os
import uuid
import pytest
from uuid import UUID
from dotenv import load_dotenv

# Load environment variables from .env (if available)
load_dotenv(override=True)

# Import the service classes and storages.
from interview_assistant.core.services import QuestionService, TipService
from interview_assistant.storage import question_storage, tip_storage
from interview_assistant.core.schemas import (
    QuestionGenerationRequest,
    GeneratedQuestion,
    GeneratedTip,
    QuestionContent,
    TipContent,
    TipGenerationRequest,
    Metadata,
)
from interview_assistant.core.ai.embeddings import OpenAIEmbedder
from interview_assistant.core.ai.content_generation.question_content_generation import (
    OpenAIQuestionContentGenerator,
)
from interview_assistant.core.ai.content_generation.tip_content_generation import (
    OpenAITipContentGenerator,
)


# --- Global Autouse Fixture for Clearing Redis ---
@pytest.fixture(autouse=True)
def clear_storage():
    """
    Ensure that both QuestionStorage and TipStorage Redis instances are cleared
    before and after each test, to guarantee test isolation.
    """

    # Clear before test
    for storage_cls in (question_storage, tip_storage):
        storage_cls.conn.flushall()
    yield
    # Clear after test
    for storage_cls in (question_storage, tip_storage):
        storage_cls.conn.flushall()


@pytest.fixture
def embedder():
    return OpenAIEmbedder()


@pytest.fixture
def q_content_gen():
    return OpenAIQuestionContentGenerator()


@pytest.fixture
def t_content_gen():
    return OpenAITipContentGenerator()


@pytest.fixture
def sample_question_request():
    return QuestionGenerationRequest(
        user_id=uuid.uuid4(),
        question_type="exercise",
        job_title="data analyst",
        skill_to_test="SQL",
        n=1,
    )


# -----------------------------------------------------------------------------
# Integration Test for QuestionService.generate_questions
# -----------------------------------------------------------------------------
def test_generate_questions_integration(embedder, q_content_gen, sample_question_request):
    """
    Integration test for generating questions.

    This test calls OpenAI's chat API to generate questions. It requires:
      - OPENAI_API_KEY
      - OPENAI_CHAT_MODEL
      - OPENAI_CHAT_TEMPERATURE
    """
    # Skip if required environment variables are missing.
    required_vars = ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_CHAT_TEMPERATURE"]
    for var in required_vars:
        if not os.getenv(var):
            pytest.skip(f"{var} is not set. Skipping integration test.")

    # Instantiate the QuestionService.
    question_service = QuestionService(embedder=embedder, content_generator=q_content_gen)

    # Generate questions.
    generated_questions = question_service.generate_questions(sample_question_request)

    # Assertions:
    assert (
        len(generated_questions) == sample_question_request.n
    ), "Not all questions were generated."
    for q in generated_questions:
        # Verify that each generated question is an instance of GeneratedQuestion.
        assert isinstance(q, GeneratedQuestion), "Returned question is not a GeneratedQuestion."
        # Check that the question text is a non-empty string.
        assert hasattr(q, "question"), "GeneratedQuestion missing 'question' attribute."
        assert (
            isinstance(q.question, str) and q.question.strip()
        ), "Generated question text is empty."


# -----------------------------------------------------------------------------
# Integration Test for TipService.generate_tip
# -----------------------------------------------------------------------------
def test_generate_tip_integration(embedder, t_content_gen, sample_question_request):
    """
    Integration test for generating a tip for an existing question.

    This test calls OpenAI's chat API to generate a tip for a stored question.
    Requires:
      - OPENAI_API_KEY
      - OPENAI_CHAT_MODEL
      - OPENAI_CHAT_TEMPERATURE
    """
    # Skip if required environment variables are missing.
    required_vars = ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_CHAT_TEMPERATURE"]
    for var in required_vars:
        if not os.getenv(var):
            pytest.skip(f"{var} is not set. Skipping integration test.")

    # First, ensure there is a question in storage.
    # Create and store a question first
    question_content = QuestionContent(
        question="What is dependency injection?",
        expected_answer="A design pattern that decouples components.",
        evaluation_criteria="Clear and concise explanation",
        expected_duration="5 minutes",
    )

    # Store the question using QuestionStorage
    generated_question = question_storage.add_question(sample_question_request, question_content)

    # Instantiate the TipService.
    tip_service = TipService(embedder=embedder, content_generator=t_content_gen)

    # Generate a tip for the stored question.
    generated_tip = tip_service.generate_tip(generated_question.id)

    # Assertions:
    assert isinstance(generated_tip, GeneratedTip), "Returned tip is not a GeneratedTip."
    # Verify that the tip text is non-empty.
    assert hasattr(generated_tip, "tip"), "GeneratedTip missing 'tip' attribute."
    assert (
        isinstance(generated_tip.tip, str) and generated_tip.tip.strip()
    ), "Generated tip text is empty."
    print("Generated Tip:", generated_tip.tip)
