"""
Defining the QuestionService and TipService classes.
"""

from typing import List
import logging
from uuid import UUID
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from openai import AsyncOpenAI

from interview_assistant.storage.storage import question_storage, tip_storage
from interview_assistant.core.schemas import (
    QuestionGenerationRequest,
    QuestionContent,
    GeneratedQuestion,
    GeneratedTip,
    TipGenerationRequest,
)
from interview_assistant.core.ai.content_generation.question_content_generation import (
    QuestionContentGeneratorType,
    OpenAIQuestionContentGenerator,
    AsyncOpenAIQuestionContentGenerator,
)
from interview_assistant.core.ai.content_generation.tip_content_generation import (
    TipContentGeneratorType,
    OpenAITipContentGenerator,
    AsyncOpenAITipContentGenerator,
)
from interview_assistant.core.ai.embeddings import EmbedderType, OpenAIEmbedder
from interview_assistant.core.utils.retry import retry, RetryStrategy

load_dotenv(override=True)
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class QuestionService(BaseModel):
    """
    Service to generate questions.
    """

    embedder: EmbedderType = Field(default_factory=OpenAIEmbedder)
    content_generator: QuestionContentGeneratorType = Field(
        default_factory=OpenAIQuestionContentGenerator
    )
    max_attempts: int = Field(default=5, gt=0, le=15)
    retry_strategy: RetryStrategy = Field(default_factory=lambda: RetryStrategy(max_attempts=5))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def generate_questions(self, request: QuestionGenerationRequest) -> List[GeneratedQuestion]:
        """
        Generate `n` unique questions based on the QuestionGenerationRequest.
        """
        # Get stored questions once at the beginning
        stored_questions = question_storage.list_questions_by_user(request.user_id) or []
        stored_embeddings = [self.embedder.get_embedding(q.question) for q in stored_questions]
        previous_questions_text = [q.question for q in stored_questions]

        generated_questions = []
        for _ in range(request.n):
            candidate_content = self._generate_unique_question(
                request, previous_questions_text, stored_embeddings
            )
            generated_question = question_storage.add_question(request, candidate_content)
            # Update our lists with the new question for the next iteration
            previous_questions_text.append(generated_question.question)
            stored_embeddings.append(self.embedder.get_embedding(generated_question.question))
            generated_questions.append(generated_question)

        return generated_questions

    def _generate_unique_question(
        self,
        request: QuestionGenerationRequest,
        previous_questions_text: List[str],
        stored_embeddings: List[List[float]],
    ) -> QuestionContent:
        """
        Attempt to generate a candidate question that is sufficiently different from
        previously stored questions for the user.
        """

        @retry(
            max_attempts=self.retry_strategy.max_attempts,
            should_retry=self.retry_strategy.should_retry,
            error_message=self.retry_strategy.error_message,
        )
        def generate_and_validate():
            # If there are no stored questions, we can skip the embedding checks
            if not stored_embeddings:
                return self.content_generator.generate_question_content(
                    request=request, previous_questions=[]
                )

            candidate = self.content_generator.generate_question_content(
                request=request, previous_questions=previous_questions_text
            )
            candidate_embedding = self.embedder.get_embedding(candidate.question)

            if not self.embedder.is_embedding_different_from_list(
                candidate_embedding, stored_embeddings
            ):
                raise ValueError("Generated question is too similar to existing ones")

            return candidate

        return generate_and_validate()


class TipService(BaseModel):
    """
    Service to generate Tip.
    """

    embedder: EmbedderType = Field(default_factory=OpenAIEmbedder)
    content_generator: TipContentGeneratorType = Field(default_factory=OpenAITipContentGenerator)
    max_attempts: int = Field(default=5, gt=0)
    retry_strategy: RetryStrategy = Field(default_factory=lambda: RetryStrategy(max_attempts=5))
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def generate_tip(self, question_id: UUID) -> GeneratedTip:
        """
        Generate a unique tip for the given GeneratedQuestion.
        """
        # Make sure the question exists in the storage
        generated_question = question_storage.get_question(question_id)
        if generated_question is None:
            raise ValueError(f"Question for id {question_id} does not exist")

        # Create the tip generation request
        tip_request = TipGenerationRequest(question_id=question_id)

        # Retrieve stored tips once at the beginning
        stored_tips = tip_storage.list_tips_for_question(question_id)

        # If there are no stored tips, we can skip the embedding checks
        if not stored_tips:
            candidate_content = self.content_generator.generate_tip_content(generated_question, [])
            return tip_storage.add_tip(tip_request, candidate_content)

        # Prepare the data we'll reuse in the loop
        previous_tips_text = [t.tip for t in stored_tips]
        stored_embeddings = [self.embedder.get_embedding(t.tip) for t in stored_tips]

        return self.generate_tip_wrapper(
            generated_question, tip_request, previous_tips_text, stored_embeddings
        )

    def generate_tip_wrapper(
        self, generated_question, tip_request, previous_tips_text, stored_embeddings
    ):
        """
        Generate a unique tip for the given GeneratedQuestion.
        """
        @retry(
            max_attempts=self.retry_strategy.max_attempts,
            should_retry=self.retry_strategy.should_retry,
            error_message=self.retry_strategy.error_message,
        )
        def generate_and_validate():
            candidate_content = self.content_generator.generate_tip_content(
                generated_question, previous_tips_text
            )
            candidate_embedding = self.embedder.get_embedding(candidate_content.tip)

            if not all(
                self.embedder.are_embeddings_different(candidate_embedding, stored_emb)
                for stored_emb in stored_embeddings
            ):
                raise ValueError("Generated tip is too similar to existing ones")

            return tip_storage.add_tip(tip_request, candidate_content)

        return generate_and_validate()

    async def generate_tip_stream(self, question_id: UUID):
        """
        Generate a unique tip for the given GeneratedQuestion and stream the response.
        """
        # Make sure the question exists in the storage
        generated_question = question_storage.get_question(question_id)
        if generated_question is None:
            raise ValueError(f"Question for id {question_id} does not exist")

        # Create the tip generation request
        tip_request = TipGenerationRequest(question_id=question_id)

        # Retrieve stored tips once at the beginning
        stored_tips = tip_storage.list_tips_for_question(question_id)
        previous_tips_text = [t.tip for t in stored_tips]
        stored_embeddings = (
            [self.embedder.get_embedding(t.tip) for t in stored_tips] if stored_tips else []
        )

        attempt = 0
        while attempt < self.retry_strategy.max_attempts:
            try:
                async for chunk in self.content_generator.generate_tip_content_stream(
                    generated_question, previous_tips_text
                ):
                    yield chunk

                # After streaming is complete, validate and store the complete tip
                complete_tip = await self.content_generator.get_complete_tip()
                if stored_embeddings:
                    candidate_embedding = self.embedder.get_embedding(complete_tip.tip)
                    if not all(
                        self.embedder.are_embeddings_different(candidate_embedding, stored_emb)
                        for stored_emb in stored_embeddings
                    ):
                        raise ValueError("Generated tip is too similar to existing ones")

                stored_tip = tip_storage.add_tip(tip_request, complete_tip)
                yield stored_tip
                return  # Success - exit the retry loop

            except Exception as e:
                attempt += 1
                if (
                    attempt >= self.retry_strategy.max_attempts
                    or not self.retry_strategy.should_retry(e)
                ):
                    raise
                logger.warning("Attempt %d failed, retrying... Error: %s", attempt, str(e))


class AsyncTipService(BaseModel):
    """
    Async service to generate Tips.
    """

    embedder: EmbedderType = Field(default_factory=OpenAIEmbedder)
    content_generator: AsyncOpenAITipContentGenerator = Field(
        default_factory=AsyncOpenAITipContentGenerator
    )
    max_attempts: int = Field(default=5, gt=0)
    retry_strategy: RetryStrategy = Field(default_factory=lambda: RetryStrategy(max_attempts=5))
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def generate_tip_stream(self, question_id: UUID):
        """
        Generate a unique tip for the given GeneratedQuestion and stream the response.
        """
        # Make sure the question exists in the storage
        generated_question = question_storage.get_question(question_id)
        if generated_question is None:
            raise ValueError(f"Question for id {question_id} does not exist")

        # Create the tip generation request
        tip_request = TipGenerationRequest(question_id=question_id)

        # Retrieve stored tips once at the beginning
        stored_tips = tip_storage.list_tips_for_question(question_id)
        previous_tips_text = [t.tip for t in stored_tips]
        stored_embeddings = (
            [self.embedder.get_embedding(t.tip) for t in stored_tips] if stored_tips else []
        )

        attempt = 0
        while attempt < self.retry_strategy.max_attempts:
            try:
                async for chunk in self.content_generator.generate_tip_content_stream(
                    generated_question, previous_tips_text
                ):
                    yield chunk

                # After streaming is complete, validate and store the complete tip
                complete_tip = await self.content_generator.get_complete_tip()
                if stored_embeddings:
                    candidate_embedding = self.embedder.get_embedding(complete_tip.tip)
                    if not all(
                        self.embedder.are_embeddings_different(candidate_embedding, stored_emb)
                        for stored_emb in stored_embeddings
                    ):
                        raise ValueError("Generated tip is too similar to existing ones")

                stored_tip = tip_storage.add_tip(tip_request, complete_tip)
                yield stored_tip
                return  # Success - exit the retry loop

            except Exception as e:
                attempt += 1
                if (
                    attempt >= self.retry_strategy.max_attempts
                    or not self.retry_strategy.should_retry(e)
                ):
                    raise
                logger.warning("Attempt %d failed, retrying... Error: %s", attempt, str(e))


class AsyncQuestionService(BaseModel):
    """
    Async service to generate questions.
    """

    embedder: EmbedderType = Field(default_factory=OpenAIEmbedder)
    content_generator: AsyncOpenAIQuestionContentGenerator = Field(
        default_factory=AsyncOpenAIQuestionContentGenerator
    )
    max_attempts: int = Field(default=5, gt=0, le=15)
    retry_strategy: RetryStrategy = Field(default_factory=lambda: RetryStrategy(max_attempts=5))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def generate_question_stream(self, request: QuestionGenerationRequest):
        """
        Generate a question and stream the response.
        """
        # Get stored questions once at the beginning
        stored_questions = question_storage.list_questions_by_user(request.user_id) or []
        stored_embeddings = [self.embedder.get_embedding(q.question) for q in stored_questions]
        previous_questions_text = [q.question for q in stored_questions]

        attempt = 0
        while attempt < self.retry_strategy.max_attempts:
            try:
                async for chunk in self.content_generator.generate_question_content_stream(
                    request=request, previous_questions=previous_questions_text
                ):
                    yield chunk

                # After streaming is complete, validate and store the complete question
                complete_question = await self.content_generator.get_complete_question()

                if stored_embeddings:
                    candidate_embedding = self.embedder.get_embedding(complete_question.question)
                    if not self.embedder.is_embedding_different_from_list(
                        candidate_embedding, stored_embeddings
                    ):
                        raise ValueError("Generated question is too similar to existing ones")

                stored_question = question_storage.add_question(request, complete_question)
                yield stored_question
                return  # Success - exit the retry loop

            except Exception as e:
                attempt += 1
                if (
                    attempt >= self.retry_strategy.max_attempts
                    or not self.retry_strategy.should_retry(e)
                ):
                    raise
                logger.warning("Attempt %d failed, retrying... Error: %s", attempt, str(e))
