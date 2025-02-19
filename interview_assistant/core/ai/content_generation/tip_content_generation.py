from abc import ABC, abstractmethod
import os
from typing import List, Literal, Union, Optional, AsyncGenerator
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from interview_assistant.core.schemas import TipContent, GeneratedQuestion, GeneratedTip

load_dotenv(override=True)


class TipContentGenerator(BaseModel, ABC):
    """Abstract Class for all Embedders"""

    temperature: float = Field(default=0.8, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def generate_tip_content(
        self, generated_question: GeneratedQuestion, previous_tips: List[str]
    ) -> TipContent:
        """
        Function to generate a tip given generated question and optionnaly the previous generated tips asked to the user.
        """
        pass


class OpenAITipContentGenerator(TipContentGenerator):
    """Wrapper around OpenAI APIs to get embeddings."""

    type: Literal["openaitipcontentgenerator"] = "openaitipcontentgenerator"
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL")
    # Define a private attributes
    _chat_client: OpenAI = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Instantiate the OpenAI client as a private attribute.
        self._chat_client = OpenAI()

    def generate_tip_content(
        self, generated_question: GeneratedQuestion, previous_tips: List[str]
    ) -> TipContent:
        """
        Generates tip content using OpenAI's chat API.

        The prompt includes the question text (from the GeneratedQuestion) and any
        previously generated tips for that question to encourage a unique output.
        """
        prompt = self._generate_prompt(generated_question, previous_tips)

        # Call OpenAI's chat API.
        response = self._chat_client.beta.chat.completions.parse(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert tip generator in the context of technical interviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_CHAT_TEMPERATURE")),
            response_format=TipContent,
        )
        generated_text = response.choices[0].message
        # If the model refuses to respond, you will get a refusal message
        if generated_text.refusal:
            raise ValueError(generated_text.refusal)

        return generated_text.parsed

    @staticmethod
    def _generate_prompt(
        generated_question: GeneratedQuestion, previous_tips: Optional[List[str]] = None
    ) -> str:
        # Create a prompt for the tip content generator
        prompt = (
            "I need you to perform a TASK about this QUESTION:\n"
            f"{generated_question.question}\n\n"
            f"CONTEXT:\n"
            f"During a {generated_question.request.job_title} interview, "
            f"this {generated_question.request.question_type} QUESTION was asked "
            f"to test a candidate's {generated_question.request.skill_to_test} skill.\n"
            f"The ANSWER to the QUESTION is {generated_question.expected_answer} "
            f"and the candidate will be evaluated on the following criteria: {generated_question.evaluation_criteria}.\n\n"
            "TASK:\n"
            "Generate one short and concise FINAL_TIP to help the candidate give an ANSWER the QUESTION. "
            "Do not reveal the ANSWER to the question; only provide one short consideration to think about.\n\n"
        )

        if previous_tips:
            prompt += (
                "CONSTRAINTS:\n"
                "Ensure that the generated TIP is unique and different from the following tips:\n"
                + "\n".join(f"- PREVIOUS TIP {i}: {tip}" for i, tip in enumerate(previous_tips))
                + "\n"
            )

        prompt += (
            "FINAL INSTRUCTIONS:\n"
            "Please internally think through your reasoning step-by-step to arrive at the best TIP, "
            "but do not include any of your internal chain-of-thought in your final output. "
            "Only provide the final TIP below.\n\n"
            "FINAL_TIP:\n"
        )
        # TODO: Provide few examples of questions for the different types.

        return prompt

    async def generate_tip_content_stream(
        self, generated_question: GeneratedQuestion, previous_tips: List[str]
    ):
        """
        Streams tip content using OpenAI's chat API.
        """
        prompt = self._generate_prompt(generated_question, previous_tips)
        self._current_tip = ""

        # Call OpenAI's chat API with streaming
        stream = await self._chat_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert tip generator in the context of technical interviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_CHAT_TEMPERATURE")),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                self._current_tip += content
                yield content

    async def get_complete_tip(self) -> TipContent:
        """Returns the complete tip after streaming is finished."""
        return TipContent(tip=self._current_tip)


class AsyncOpenAITipContentGenerator(OpenAITipContentGenerator):
    """
    Async OpenAI-based tip content generator.
    """

    chat_client: AsyncOpenAI = Field(default_factory=AsyncOpenAI)
    chat_model: str = Field(default="gpt-4")
    _current_tip: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def generate_tip_content_stream(
        self, generated_question: GeneratedQuestion, previous_tips: List[str]
    ) -> AsyncGenerator[Union[str, GeneratedTip], None]:
        """
        Generate a tip content using OpenAI's API with streaming.
        """
        prompt = self._generate_prompt(generated_question, previous_tips)
        self._current_tip = ""

        stream = await self.chat_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert tip generator in the context of technical interviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_CHAT_TEMPERATURE")),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                self._current_tip += content
                yield content

    async def get_complete_tip(self) -> TipContent:
        """
        Get the complete tip after streaming is done.
        """
        return TipContent(tip=self._current_tip)


TipContentGeneratorType = Union[OpenAITipContentGenerator, AsyncOpenAITipContentGenerator]
