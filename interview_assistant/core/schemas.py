"""
Defining the schemas of the API input/output
"""

from datetime import datetime, UTC
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Common metadata for all generated entities"""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    app_version: str = Field(default="0.1.0")


class QuestionGenerationRequest(BaseModel):
    """
    Input for the QuestionGeneration API.
    """

    user_id: UUID
    question_type: Literal["exercise", "knowledge question"]
    job_title: Literal["data analyst", "data scientist", "data engineer"]
    skill_to_test: str
    n: int = Field(ge=1, le=10)


class QuestionContent(BaseModel):
    """
    Content generated QuestionGeneration API.
    """

    question: str
    expected_answer: str
    evaluation_criteria: str
    expected_duration: str


class GeneratedQuestion(QuestionContent, Metadata):
    """
    The question Generated
    """

    id: UUID
    request: QuestionGenerationRequest


class TipGenerationRequest(BaseModel):
    """
    Input for the TipGeneration API.
    """

    question_id: UUID


class TipContent(BaseModel):
    """
    Content generated TipGeneration API.
    """

    tip: str


class GeneratedTip(TipContent, Metadata):
    """
    Output for the TipGeneration API.
    """

    id: UUID
    request: TipGenerationRequest
