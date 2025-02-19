import os
import pytest
from uuid import uuid1

from dotenv import load_dotenv

from interview_assistant.core.schemas import (
    TipContent,
    GeneratedQuestion,
    QuestionGenerationRequest,
)
from interview_assistant.core.ai.content_generation.tip_content_generation import (
    OpenAITipContentGenerator,
)

# Load environment variables from .env
load_dotenv(override=True)


def test_generate_tip_content_integration():
    """
    Integration test that makes a real call to OpenAI's chat API to generate a tip.

    Requirements:
      - OPENAI_API_KEY must be set.
      - OPENAI_CHAT_MODEL must be set.
      - OPENAI_CHAT_TEMPERATURE must be set.
    """
    # Verify that the necessary environment variables are present.
    required_vars = ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_CHAT_TEMPERATURE"]
    for var in required_vars:
        if not os.getenv(var):
            pytest.skip(f"{var} is not set. Skipping integration test.")

    # Instantiate the tip content generator.
    tip_generator = OpenAITipContentGenerator()

    # Create a dummy GeneratedQuestion.
    # Adjust the fields below based on your actual GeneratedQuestion schema.
    generated_question = GeneratedQuestion(
        expected_answer="N/A",
        evaluation_criteria="N/A",
        expected_duration="N/A",
        question="What are the benefits of using poetry in Python package?",
        id=uuid1(),
        request=QuestionGenerationRequest(
            user_id=uuid1(),
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="SQL",
            n=1,
        ),
    )

    # Provide an empty list or some previous tips.
    previous_tips = []

    # Call the generator to produce tip content.
    tip_content = tip_generator.generate_tip_content(generated_question, previous_tips)

    # Basic assertions:
    # 1. Verify that the returned object is an instance of TipContent.
    assert isinstance(tip_content, TipContent), "Returned object is not of type TipContent."

    # 2. Check that the generated tip (assuming an attribute 'tip') is a non-empty string.
    assert hasattr(
        tip_content, "tip"
    ), "The TipContent schema does not have a 'tip' attribute to validate."

    generated_tip = tip_content.tip
    assert isinstance(generated_tip, str), "Generated tip is not a string."
    assert generated_tip.strip() != "", "Generated tip is empty."
