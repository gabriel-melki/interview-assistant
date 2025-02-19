import uuid
from datetime import datetime
import pytest
from dotenv import load_dotenv

from interview_assistant.core.schemas import (
    QuestionContent,
    TipContent,
    GeneratedQuestion,
    GeneratedTip,
    QuestionGenerationRequest,
    TipGenerationRequest,
)

load_dotenv(override=True)


@pytest.fixture(autouse=True)
def clear_storage():
    """
    Ensure that both QuestionStorage and TipStorage Redis instances are cleared
    before and after each test, to guarantee test isolation.
    """
    from interview_assistant.storage.storage import QuestionStorage, TipStorage

    # Clear before test
    for storage_cls in (QuestionStorage, TipStorage):
        storage_cls().conn.flushall()
    yield
    # Clear after test
    for storage_cls in (QuestionStorage, TipStorage):
        storage_cls().conn.flushall()


@pytest.fixture(scope="module")
def question_storage():
    """Provides the singleton instance of QuestionStorage."""
    from interview_assistant.storage.storage import QuestionStorage

    return QuestionStorage()


@pytest.fixture(scope="module")
def tip_storage():
    """Provides the singleton instance of TipStorage."""
    from interview_assistant.storage.storage import TipStorage

    return TipStorage()


# --- Tests for QuestionStorage ---
class TestQuestionStorage:
    def test_add_question(self, question_storage):
        user_id = uuid.uuid4()
        request = QuestionGenerationRequest(
            user_id=user_id,
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="SQL",
            n=1,
        )
        question_content = QuestionContent(
            question="What is data analytics?",
            expected_answer="Data analytics is the process of examining data sets...",
            evaluation_criteria="Look for mention of data cleaning and insights",
            expected_duration="10 minutes",
        )

        generated_question = question_storage.add_question(request, question_content)

        # Basic checks on the returned GeneratedQuestion
        assert isinstance(generated_question, GeneratedQuestion)
        assert generated_question.request == request
        assert generated_question.question == question_content.question
        assert generated_question.expected_answer == question_content.expected_answer
        assert generated_question.evaluation_criteria == question_content.evaluation_criteria
        assert generated_question.expected_duration == question_content.expected_duration
        # Check the metadata
        assert generated_question.app_version == "0.1.0"
        assert isinstance(generated_question.created_at, datetime)

    def test_add_question_duplicate_id_raises_error(self, question_storage, monkeypatch):
        """
        Force uuid.uuid1 to always return the same value so that the second
        add_question call triggers a ValueError.
        """
        fixed_id = uuid.uuid4()

        def mock_uuid1():
            return fixed_id

        monkeypatch.setattr(uuid, "uuid1", mock_uuid1)

        user_id = uuid.uuid4()
        request = QuestionGenerationRequest(
            user_id=user_id,
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="SQL",
            n=1,
        )
        question_content = QuestionContent(
            question="Duplicate question?",
            expected_answer="Yes",
            evaluation_criteria="Any mention of duplication",
            expected_duration="5 minutes",
        )

        # First call works
        question_storage.add_question(request, question_content)

        # Second call should raise an error because the question_id is a duplicate
        with pytest.raises(ValueError) as exc:
            question_storage.add_question(request, question_content)
        assert "already exists" in str(exc.value)

    def test_get_question(self, question_storage):
        user_id = uuid.uuid4()
        request = QuestionGenerationRequest(
            user_id=user_id,
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="maths",
            n=1,
        )
        question_content = QuestionContent(
            question="What is 2 + 2?",
            expected_answer="4",
            evaluation_criteria="Correct numeric answer",
            expected_duration="1 minute",
        )

        generated_question = question_storage.add_question(request, question_content)
        # Retrieve it
        retrieved = question_storage.get_question(generated_question.id)
        assert retrieved is not None
        assert retrieved.id == generated_question.id
        assert retrieved.request.user_id == user_id
        assert retrieved.question == "What is 2 + 2?"

    def test_get_question_not_found(self, question_storage):
        # Querying a non-existent question should return None
        random_id = uuid.uuid4()
        retrieved = question_storage.get_question(random_id)
        assert retrieved is None

    def test_list_questions_by_user(self, question_storage):
        user_id = uuid.uuid4()

        # Add two questions for the same user
        request_1 = QuestionGenerationRequest(
            user_id=user_id,
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="SQL",
            n=1,
        )
        question_content_1 = QuestionContent(
            question="Q1",
            expected_answer="A1",
            evaluation_criteria="E1",
            expected_duration="D1",
        )
        question_content_2 = QuestionContent(
            question="Q2",
            expected_answer="A2",
            evaluation_criteria="E2",
            expected_duration="D2",
        )
        q1 = question_storage.add_question(request_1, question_content_1)
        q2 = question_storage.add_question(request_1, question_content_2)

        # Add a question for another user (should not appear in the list)
        another_user_id = uuid.uuid4()
        request_2 = QuestionGenerationRequest(
            user_id=another_user_id,
            question_type="exercise",
            job_title="data analyst",
            skill_to_test="SQL",
            n=1,
        )
        question_storage.add_question(
            request_2,
            QuestionContent(
                question="Other Q",
                expected_answer="Other A",
                evaluation_criteria="Other E",
                expected_duration="Other D",
            ),
        )

        user_questions = question_storage.list_questions_by_user(user_id)
        assert len(user_questions) == 2
        question_ids = {q.id for q in user_questions}
        assert q1.id in question_ids
        assert q2.id in question_ids


# --- Tests for TipStorage ---
class TestTipStorage:
    def test_add_tip(self, tip_storage):
        question_id = uuid.uuid4()
        request = TipGenerationRequest(question_id=question_id)
        tip_content = TipContent(tip="Consider verifying data quality first.")

        generated_tip = tip_storage.add_tip(request, tip_content)

        # Basic checks on the returned GeneratedTip
        assert isinstance(generated_tip, GeneratedTip)
        assert generated_tip.request == request
        assert generated_tip.tip == tip_content.tip
        # Check metadata
        assert generated_tip.app_version == "0.1.0"
        assert isinstance(generated_tip.created_at, datetime)

    def test_add_tip_duplicate_id_raises_error(self, tip_storage, monkeypatch):
        """
        Force uuid.uuid1 to always return the same value so that the second
        add_tip call triggers a ValueError.
        """
        fixed_id = uuid.uuid4()

        def mock_uuid1():
            return fixed_id

        monkeypatch.setattr(uuid, "uuid1", mock_uuid1)

        question_id = uuid.uuid4()
        tip_request = TipGenerationRequest(question_id=question_id)
        tip_content = TipContent(tip="First tip")

        # First call is successful
        tip_storage.add_tip(tip_request, tip_content)

        # Second call should raise an error because the tip_id is a duplicate
        with pytest.raises(ValueError) as exc:
            tip_storage.add_tip(tip_request, tip_content)
        assert "already exists" in str(exc.value)

    def test_get_tip(self, tip_storage):
        question_id = uuid.uuid4()
        tip_request = TipGenerationRequest(question_id=question_id)
        tip_content = TipContent(tip="Use appropriate libraries for data wrangling.")
        generated_tip = tip_storage.add_tip(tip_request, tip_content)

        retrieved = tip_storage.get_tip(generated_tip.id)
        assert retrieved is not None
        assert retrieved.id == generated_tip.id
        assert retrieved.tip == tip_content.tip
        assert isinstance(retrieved.created_at, datetime)

    def test_get_tip_not_found(self, tip_storage):
        random_id = uuid.uuid4()
        retrieved = tip_storage.get_tip(random_id)
        assert retrieved is None

    def test_list_tips_for_question(self, tip_storage):
        question_id = uuid.uuid4()
        tip_request = TipGenerationRequest(question_id=question_id)
        tip_content_1 = TipContent(tip="First tip")
        tip_content_2 = TipContent(tip="Second tip")

        t1 = tip_storage.add_tip(tip_request, tip_content_1)
        t2 = tip_storage.add_tip(tip_request, tip_content_2)

        # Add a tip for a different question (should not be in the list)
        another_question_id = uuid.uuid4()
        tip_storage.add_tip(
            TipGenerationRequest(question_id=another_question_id),
            TipContent(tip="Other question's tip"),
        )

        question_tips = tip_storage.list_tips_for_question(question_id)
        assert len(question_tips) == 2
        tip_ids = {t.id for t in question_tips}
        assert t1.id in tip_ids
        assert t2.id in tip_ids


# --- Singleton Behavior Tests ---
class TestSingleton:
    def test_question_storage_singleton(self):
        from interview_assistant.storage.storage import QuestionStorage

        instance1 = QuestionStorage()
        instance2 = QuestionStorage()
        assert (
            instance1 is instance2
        ), "QuestionStorage instances are not the same (singleton violation)"

    def test_tip_storage_singleton(self):
        from interview_assistant.storage.storage import TipStorage

        instance1 = TipStorage()
        instance2 = TipStorage()
        assert instance1 is instance2, "TipStorage instances are not the same (singleton violation)"
