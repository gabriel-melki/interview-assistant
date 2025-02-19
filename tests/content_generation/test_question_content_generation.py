import os
from uuid import uuid1
import pytest
from dotenv import load_dotenv

from interview_assistant.core.schemas import QuestionGenerationRequest, QuestionContent
from interview_assistant.core.ai.content_generation.question_content_generation import (
    OpenAIQuestionContentGenerator,
)

# Load environment variables from .env if available.
load_dotenv(override=True)


def test_generate_question_content_integration():
    """
    Integration test that makes a real call to OpenAI's chat API to generate a question.

    Requires the following environment variables to be set:
      - OPENAI_API_KEY
      - OPENAI_CHAT_MODEL
      - OPENAI_CHAT_TEMPERATURE
    """
    # Skip the test if any required environment variable is missing.
    required_env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_CHAT_MODEL",
        "OPENAI_CHAT_TEMPERATURE",
    ]
    for var in required_env_vars:
        if not os.getenv(var):
            pytest.skip(f"{var} is not set. Skipping integration test.")

    # Create an instance of the generator.
    generator = OpenAIQuestionContentGenerator()

    # Create a sample request. (Make sure the field names match your actual schema.)
    request = QuestionGenerationRequest(
        user_id=uuid1(),
        question_type="knowledge question",
        job_title="data engineer",
        skill_to_test="Python programming",
        n=1,
    )

    # Optionally, provide any previous questions; here we use an empty list.
    previous_questions = []

    # Call the generator to produce a question.
    question_content = generator.generate_question_content(request, previous_questions)

    # Verify that the returned object is an instance of QuestionContent.
    assert isinstance(
        question_content, QuestionContent
    ), "Returned value is not of type QuestionContent."

    # Check that the generated question (or other relevant field) is a non-empty string.
    assert hasattr(
        question_content, "question"
    ), "The QuestionContent schema does not have a 'question' attribute to validate."

    generated_text = question_content.question
    assert isinstance(generated_text, str), "The generated question is not a string."
    assert generated_text.strip() != "", "The generated question is empty."
