"""
Defining the QuestionContentGenerator class and the OpenAIQuestionContentGenerator class.
"""

from abc import ABC, abstractmethod
import os
from typing import List, Literal, Union, Optional, AsyncGenerator
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from interview_assistant.core.schemas import (
    QuestionGenerationRequest,
    QuestionContent,
    GeneratedQuestion,
)

load_dotenv(override=True)


class QuestionContentGenerator(BaseModel, ABC):
    """Abstract Class for all Question Content Generators"""

    temperature: float = Field(default=0.8, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def generate_question_content(
        self, request: QuestionGenerationRequest, previous_questions: List[str]
    ) -> QuestionContent:
        pass


class OpenAIQuestionContentGenerator(QuestionContentGenerator):
    """Wrapper around OpenAI APIs to generate question content."""

    type: Literal["openaiquestioncontentgenerator"] = "openaiquestioncontentgenerator"
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL")
    # Define a private attributes
    _chat_client: OpenAI = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Instantiate the OpenAI client as a private attribute.
        self._chat_client = OpenAI()

    def generate_question_content(
        self, request: QuestionGenerationRequest, previous_questions: List[str]
    ) -> QuestionContent:
        """
        Generates question content using OpenAI's chat API.
        The prompt includes both the request details and any previously generated questions.
        """
        prompt = self._generate_prompt(request, previous_questions)
        try:
            response = self._chat_client.beta.chat.completions.parse(
                model=self.chat_model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert question generator in the context of interviews.",
                }, {"role": "user", "content": prompt}],
                temperature=float(os.getenv("OPENAI_CHAT_TEMPERATURE")),
                response_format=QuestionContent,
            )

            generated_text = response.choices[0].message

            return generated_text.parsed
        except Exception as e:
            raise ValueError(f"Failed to generate question content: {e}")

    @staticmethod
    def _generate_prompt(
        request: QuestionGenerationRequest,
        previous_questions: Optional[List[str]] = None,
    ) -> str:
        # Create a prompt for the question content generator
        prompt = (
            f"TASK:\n"
            f"Generate a {request.question_type} FINAL_QUESTION for a {request.job_title} "
            f"to test their skill in {request.skill_to_test}.\n\n "
            "CONSTRAINTS:\n"
            f"Make sure this FINAL_QUESTION includes the following elements:\n"
            "- the expected_answer of the FINAL_QUESTION \n"
            "- the expected_duration to solve answer the FINAL_QUESTION \n"
            "- A list of generic evaluation_criteria to assess the quality of the answer. "
            "The evaluation_criteria should be generic and not job-specific. "
            "The evaluation_criteria should be a list of disctinct skills "
            "and non overlapping criteria. "
            "Do not define the evaluation_criteria, just give keywords. \n\n"
        )
        if previous_questions:
            prompt += (
                "ADDITIONAL CONSTRAINTS:\n"
                "Ensure that this QUESTION is unique and different from the following "
                "previously generated questions:\n"
                + "\n".join(
                    f"- PREVIOUS QUESTION {i} was {question}"
                    for i, question in enumerate(previous_questions)
                )
                + "\n"
            )

        prompt += (
            "FINAL INSTRUCTIONS:\n"
            "Please internally think through your reasoning step-by-step to arrive "
            "at the best FINAL_QUESTION, "
            "but do not include any of your internal chain-of-thought in your final output. "
            "Only provide the final FINAL_QUESTION below.\n\n"
            "FINAL_QUESTION:\n"
        )
        # TODO: Provide few examples of tip for the different types of questions.
        return prompt


class AsyncOpenAIQuestionContentGenerator(OpenAIQuestionContentGenerator):
    """
    Async OpenAI-based question content generator.
    """

    chat_client: AsyncOpenAI = Field(default_factory=AsyncOpenAI)
    chat_model: str = Field(default="gpt-4")
    _current_question: QuestionContent = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _parse_response(self, response_text: str) -> QuestionContent:
        """
        Parse the raw response text into a QuestionContent object.

        Args:
            response_text: The raw text response from the API

        Returns:
            QuestionContent object containing the parsed question data
        """
        # Split the response into sections based on expected format
        lines = response_text.strip().split("\n")

        # Initialize variables
        question = ""
        expected_answer = ""
        evaluation_criteria = ""
        expected_duration = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if "Question:" in line:
                current_section = "question"
                question = line.split("Question:", 1)[1].strip()
            elif "Expected Answer:" in line:
                current_section = "answer"
                expected_answer = line.split("Expected Answer:", 1)[1].strip()
            elif "Evaluation Criteria:" in line:
                current_section = "criteria"
                evaluation_criteria = line.split("Evaluation Criteria:", 1)[1].strip()
            elif "Expected Duration:" in line:
                current_section = "duration"
                expected_duration = line.split("Expected Duration:", 1)[1].strip()
            elif line and current_section:
                # Append to current section if it's a continuation
                if current_section == "question":
                    question += " " + line
                elif current_section == "answer":
                    expected_answer += " " + line
                elif current_section == "criteria":
                    evaluation_criteria += " " + line
                elif current_section == "duration":
                    expected_duration += " " + line

        return QuestionContent(
            question=question,
            expected_answer=expected_answer,
            evaluation_criteria=evaluation_criteria,
            expected_duration=expected_duration,
        )

    async def generate_question_content_stream(
        self, request: QuestionGenerationRequest, previous_questions: List[str]
    ) -> AsyncGenerator[Union[str, GeneratedQuestion], None]:
        """
        Generate question content using OpenAI's API with streaming.
        """
        prompt = self._generate_prompt(request, previous_questions)
        self._current_question = None

        stream = await self.chat_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert question generator for technical interviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_CHAT_TEMPERATURE")),
            stream=True,
        )

        current_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                current_text += content
                yield content

        # Parse the complete response into QuestionContent
        self._current_question = self._parse_response(current_text)

    async def get_complete_question(self) -> QuestionContent:
        """
        Get the complete question after streaming is done.
        """
        if self._current_question is None:
            raise ValueError("No question has been generated yet")
        return self._current_question


QuestionContentGeneratorType = Union[
    OpenAIQuestionContentGenerator, AsyncOpenAIQuestionContentGenerator
]
