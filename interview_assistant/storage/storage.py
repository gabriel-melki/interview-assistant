"""
Defining the Storage class and the QuestionStorage and TipStorage classes.
"""

import os
import uuid
from uuid import UUID
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import fakeredis

from interview_assistant.core.schemas import (
    Metadata,
    QuestionGenerationRequest,
    QuestionContent,
    GeneratedQuestion,
    TipGenerationRequest,
    TipContent,
    GeneratedTip,
)

load_dotenv(override=True)


def singleton(cls):
    """Ensure all instances are pointing to the same instance"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class BaseStorage(BaseModel):
    """Base class for fake Redis storage operations."""

    EXPIRATION_SECONDS: int = os.getenv("EXPIRATION_SECONDS")
    _conn: Optional[fakeredis.FakeStrictRedis] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def conn(self) -> fakeredis.FakeStrictRedis:
        """
        Get the fakeredis connection.
        """
        if self._conn is None:
            self._conn = fakeredis.FakeStrictRedis(decode_responses=True, encoding="utf-8",)
        return self._conn

    def key_exists(self, key: str) -> bool:
        """
        Check if a key exists in the fakeredis database.
        """
        return self.conn.exists(key) == 1


@singleton
class QuestionStorage(BaseStorage):
    """Manages unique question storage with fakeredis."""

    def add_question(
        self, request: QuestionGenerationRequest, question_content: QuestionContent
    ) -> GeneratedQuestion:
        """
        Adds a unique question for a given user. 
        Raises an error if the question ID already exists.
        """

        question_id = uuid.uuid1()
        question_key = self._get_question_key(question_id)

        if self.key_exists(question_key):
            raise ValueError(f"Question with ID {question_id} already exists.")

        metadata = Metadata()
        question_data = {
            **question_content.model_dump(),
            **{
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in metadata.model_dump().items()
            },
            "id": str(question_id),
            "request": request.model_dump_json(),
        }

        user_questions_key = self._get_user_questions_key(request.user_id)

        pipe = self.conn.pipeline()
        pipe.hset(question_key, mapping=question_data)
        pipe.sadd(user_questions_key, str(question_id))
        pipe.expire(question_key, self.EXPIRATION_SECONDS)
        pipe.expire(user_questions_key, self.EXPIRATION_SECONDS)
        pipe.execute()

        return GeneratedQuestion(
            **question_content.model_dump(),
            **metadata.model_dump(),
            id=question_id,
            request=request,
        )

    def get_question(self, question_id: UUID) -> Optional[GeneratedQuestion]:
        """Retrieve a stored question by its ID."""
        question_key = self._get_question_key(question_id)
        data = self.conn.hgetall(question_key)
        if data:
            # Parse datetime strings back to datetime objects
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            return GeneratedQuestion(
                **{k: data[k] for k in data if k not in {"id", "request"}},
                id=UUID(data["id"]),
                request=QuestionGenerationRequest.model_validate_json(data["request"]),
            )
        return None

    def list_questions_by_user(self, user_id: UUID) -> List[GeneratedQuestion]:
        """List all questions generated for a specific user."""
        user_questions_key = self._get_user_questions_key(user_id)
        question_ids = self.conn.smembers(user_questions_key)
        return [
            self.get_question(UUID(qid)) for qid in question_ids if self.get_question(UUID(qid))
        ]

    def _get_question_key(self, question_id: UUID) -> str:
        """Generate the Redis key for a question."""
        return f"question:{question_id}"

    def _get_user_questions_key(self, user_id: UUID) -> str:
        """Generate the Redis key for a user's questions."""
        return f"user:{user_id}:questions"


@singleton
class TipStorage(BaseStorage):
    """Manages unique tip storage with fakeredis."""

    def add_tip(self, request: TipGenerationRequest, tip_content: TipContent) -> GeneratedTip:
        """Adds a unique tip for a given question. Raises an error if the tip ID already exists."""

        tip_id = uuid.uuid1()
        tip_key = self._get_tip_key(tip_id)

        if self.key_exists(tip_key):
            raise ValueError(f"Tip with ID {tip_id} already exists.")

        metadata = Metadata()
        tip_record = {
            **tip_content.model_dump(),
            **{
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in metadata.model_dump().items()
            },
            "id": str(tip_id),
            "request": request.model_dump_json(),
        }

        question_tips_key = self._get_question_tips_key(request.question_id)

        pipe = self.conn.pipeline()
        pipe.hset(tip_key, mapping=tip_record)
        pipe.sadd(question_tips_key, str(tip_id))
        pipe.expire(tip_key, self.EXPIRATION_SECONDS)
        pipe.expire(question_tips_key, self.EXPIRATION_SECONDS)
        pipe.execute()

        return GeneratedTip(
            **tip_content.model_dump(), **metadata.model_dump(), id=tip_id, request=request,
        )

    def get_tip(self, tip_id: UUID) -> Optional[GeneratedTip]:
        """Retrieve a stored tip by its ID."""
        tip_key = self._get_tip_key(tip_id)
        data = self.conn.hgetall(tip_key)
        if data:
            # Parse datetime strings back to datetime objects
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            return GeneratedTip(
                **{k: data[k] for k in data if k not in {"id", "request"}},
                id=UUID(data["id"]),
                request=TipGenerationRequest.model_validate_json(data["request"]),
            )
        return None

    def list_tips_for_question(self, question_id: UUID) -> List[GeneratedTip]:
        """List all tips generated for a given question."""
        question_tips_key = self._get_question_tips_key(question_id)
        tip_ids = self.conn.smembers(question_tips_key)
        return [self.get_tip(UUID(tid)) for tid in tip_ids if self.get_tip(UUID(tid))]

    def _get_tip_key(self, tip_id: UUID) -> str:
        """Generate the Redis key for a tip."""
        return f"tip:{tip_id}"

    def _get_question_tips_key(self, question_id: UUID) -> str:
        """Generate the Redis key for all tips related to a question."""
        return f"question:{question_id}:tips"


# Create global singletons
question_storage = QuestionStorage()
tip_storage = TipStorage()
