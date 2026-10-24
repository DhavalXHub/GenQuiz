"""
tests/test_local_batching.py - Automated tests for safe local model batch orchestration.

Verifies:
1. Batch planning for 8, 9, 15, and 20 questions.
2. Mixed question-type distribution preservation across batches.
3. No single batch exceeds LOCAL_MAX_QUESTIONS (8).
4. Multi-batch generation via generate_quiz_local_batched (mocked).
5. Duplicate question detection across batches.
6. Progress callback invocation with actual batch numbers.
7. Graceful partial result delivery on batch failure.
8. Complete failure reporting when all batches fail.
9. End-to-end QuizGenerator integration for local batched generation.
"""

from unittest.mock import MagicMock, patch
import pytest

from local_model_service import (
    LOCAL_MAX_QUESTIONS,
    split_request_into_batches,
    normalize_question_for_dedup,
    are_questions_duplicate,
    generate_quiz_local_batched,
)
from quiz_generator import QuizGenerator, parse_user_prompt, GeneratedQuiz
from dataset import DatasetManager
from llm_service import LLMService


# ===========================================================================
# 1. Batch Planning & Splitting Unit Tests
# ===========================================================================

def test_split_request_8_questions_single_batch():
    """8 questions is exactly at the limit; must produce a single batch of 8."""
    batches = split_request_into_batches(
        target_count=8,
        type_counts={"MCQ": 8},
        target_types=["MCQ"],
        max_batch_size=8,
    )
    assert len(batches) == 1
    assert batches[0]["count"] == 8
    assert batches[0]["type_counts"] == {"MCQ": 8}


def test_split_request_9_questions_two_batches():
    """9 questions exceeds limit; must split into batches of 8 and 1."""
    batches = split_request_into_batches(
        target_count=9,
        type_counts={"MCQ": 9},
        target_types=["MCQ"],
        max_batch_size=8,
    )
    assert len(batches) == 2
    assert batches[0]["count"] == 8
    assert batches[1]["count"] == 1
    assert sum(b["count"] for b in batches) == 9
    assert sum(b["type_counts"].get("MCQ", 0) for b in batches) == 9


def test_split_request_15_questions_two_batches():
    """15 questions must produce batches of 8 and 7."""
    batches = split_request_into_batches(
        target_count=15,
        type_counts={"MCQ": 15},
        target_types=["MCQ"],
        max_batch_size=8,
    )
    assert len(batches) == 2
    assert batches[0]["count"] == 8
    assert batches[1]["count"] == 7
    assert sum(b["count"] for b in batches) == 15
    assert sum(b["type_counts"].get("MCQ", 0) for b in batches) == 15


def test_split_request_20_questions_three_batches():
    """20 questions must produce batches of 8, 8, and 4."""
    batches = split_request_into_batches(
        target_count=20,
        type_counts={"MCQ": 20},
        target_types=["MCQ"],
        max_batch_size=8,
    )
    assert len(batches) == 3
    assert batches[0]["count"] == 8
    assert batches[1]["count"] == 8
    assert batches[2]["count"] == 4
    assert sum(b["count"] for b in batches) == 20
    assert sum(b["type_counts"].get("MCQ", 0) for b in batches) == 20


def test_split_request_mixed_types_preserved():
    """
    9 MCQs and 4 Short-Answer questions (total 13):
    Must split into batches of 8 and 5 while preserving exact totals:
    sum(MCQ) == 9 and sum(Short Answer) == 4.
    """
    batches = split_request_into_batches(
        target_count=13,
        type_counts={"MCQ": 9, "Short Answer": 4},
        target_types=["MCQ", "Short Answer"],
        max_batch_size=8,
    )
    assert len(batches) == 2
    assert batches[0]["count"] == 8
    assert batches[1]["count"] == 5
    assert sum(b["count"] for b in batches) == 13

    total_mcqs = sum(b["type_counts"].get("MCQ", 0) for b in batches)
    total_sa = sum(b["type_counts"].get("Short Answer", 0) for b in batches)
    assert total_mcqs == 9
    assert total_sa == 4


def test_split_request_never_exceeds_max_batch_size():
    """Verify invariant: No planned batch ever exceeds max_batch_size."""
    for count in [1, 5, 8, 9, 15, 16, 20, 25, 30]:
        batches = split_request_into_batches(
            target_count=count,
            type_counts={},
            target_types=["MCQ"],
            max_batch_size=8,
        )
        assert sum(b["count"] for b in batches) == count
        for b in batches:
            assert b["count"] <= 8


# ===========================================================================
# 2. Deduplication Unit Tests
# ===========================================================================

def test_dedup_exact_and_phrasing_variations():
    q1 = "What is the primary role of self-attention in Transformers?"
    q2 = "What is the primary role of self-attention in Transformers?"
    assert are_questions_duplicate(q1, q2) is True

    # Leading phrasing variation ("Which of the following describes...")
    q3 = "Which of the following describes the primary role of self-attention in Transformers?"
    assert are_questions_duplicate(q1, q3) is True

    # Entirely distinct question
    q4 = "How does positional encoding provide sequence order information?"
    assert are_questions_duplicate(q1, q4) is False


# ===========================================================================
# 3. Multi-Batch Local Quiz Generation (Mocked Single-Batch Engine)
# ===========================================================================

def _make_mock_question(idx: int, qtype: str = "MCQ") -> dict:
    if qtype == "MCQ":
        return {
            "question_id": f"Q{idx}",
            "question": f"Question number {idx} regarding transformer attention mechanisms?",
            "question_type": "MCQ",
            "options": {
                "A": f"Distinct specific choice Alpha {idx}",
                "B": f"Distinct specific choice Beta {idx}",
                "C": f"Distinct specific choice Gamma {idx}",
                "D": f"Distinct specific choice Delta {idx}",
            },
            "correct_answer": "A",
            "explanation": f"Detailed educational explanation for question {idx}.",
            "topic": "Transformers",
            "difficulty": "Medium",
        }
    return {
        "question_id": f"Q{idx}",
        "question": f"Question number {idx} regarding transformer attention mechanisms?",
        "question_type": "Short Answer",
        "options": None,
        "correct_answer": f"Specific grounded answer phrase {idx}",
        "explanation": f"Detailed educational explanation for question {idx}.",
        "topic": "Transformers",
        "difficulty": "Medium",
    }


def test_generate_quiz_local_batched_15_questions():
    """15 questions: executes 2 batches (8 + 7) and delivers 15 questions."""
    call_counts = []

    def fake_generate_quiz_local(**kwargs):
        cnt = kwargs["target_count"]
        start = kwargs.get("start_idx", 1)
        call_counts.append(cnt)
        # Ensure single-batch limit was respected
        assert cnt <= LOCAL_MAX_QUESTIONS
        qs = [_make_mock_question(start + i) for i in range(cnt)]
        return {"title": "Test", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        quiz_dict, err = generate_quiz_local_batched(
            pdf_context="Sample context about transformers attention.",
            target_count=15,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={"MCQ": 15},
            adapter_path="/fake/adapter",
        )

    assert err is None
    assert quiz_dict is not None
    assert len(quiz_dict["questions"]) == 15
    assert call_counts == [8, 7]
    # Verify sequential IDs Q1 to Q15
    expected_ids = [f"Q{i}" for i in range(1, 16)]
    assert [q["question_id"] for q in quiz_dict["questions"]] == expected_ids


def test_generate_quiz_local_batched_20_questions():
    """20 questions: executes 3 batches (8 + 8 + 4) and delivers 20 questions."""
    call_counts = []

    def fake_generate_quiz_local(**kwargs):
        cnt = kwargs["target_count"]
        start = kwargs.get("start_idx", 1)
        call_counts.append(cnt)
        assert cnt <= LOCAL_MAX_QUESTIONS
        qs = [_make_mock_question(start + i) for i in range(cnt)]
        return {"title": "Test", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        quiz_dict, err = generate_quiz_local_batched(
            pdf_context="Sample context about transformers.",
            target_count=20,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={"MCQ": 20},
            adapter_path="/fake/adapter",
        )

    assert err is None
    assert quiz_dict is not None
    assert len(quiz_dict["questions"]) == 20
    assert call_counts == [8, 8, 4]


def test_generate_quiz_local_batched_mixed_types():
    """9 MCQs + 4 Short Answer (13 total): delivers 9 MCQs and 4 Short Answer."""
    def fake_generate_quiz_local(**kwargs):
        cnt = kwargs["target_count"]
        tcounts = kwargs.get("type_counts", {})
        start = kwargs.get("start_idx", 1)
        qs = []
        i = 0
        for qt, qcnt in tcounts.items():
            for _ in range(qcnt):
                qs.append(_make_mock_question(start + i, qtype=qt))
                i += 1
        return {"title": "Test", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        quiz_dict, err = generate_quiz_local_batched(
            pdf_context="Sample context about transformers.",
            target_count=13,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ", "Short Answer"],
            type_counts={"MCQ": 9, "Short Answer": 4},
            adapter_path="/fake/adapter",
        )

    assert err is None
    assert quiz_dict is not None
    assert len(quiz_dict["questions"]) == 13
    mcqs = [q for q in quiz_dict["questions"] if q["question_type"] == "MCQ"]
    sas = [q for q in quiz_dict["questions"] if q["question_type"] == "Short Answer"]
    assert len(mcqs) == 9
    assert len(sas) == 4


def test_generate_quiz_local_batched_progress_callback():
    """Progress callback is called for each batch with accurate batch numbering."""
    progress_calls = []

    def on_progress(batch_num, total_batches, msg):
        progress_calls.append((batch_num, total_batches, msg))

    def fake_generate_quiz_local(**kwargs):
        cnt = kwargs["target_count"]
        start = kwargs.get("start_idx", 1)
        qs = [_make_mock_question(start + i) for i in range(cnt)]
        return {"title": "Test", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        generate_quiz_local_batched(
            pdf_context="Context",
            target_count=15,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={"MCQ": 15},
            adapter_path="/fake",
            progress_callback=on_progress,
        )

    assert len(progress_calls) == 2
    assert progress_calls[0][0] == 1
    assert progress_calls[0][1] == 2
    assert progress_calls[1][0] == 2
    assert progress_calls[1][1] == 2


def test_generate_quiz_local_batched_partial_failure():
    """Batch 1 succeeds, Batch 2 fails -> delivers partial results with informative error."""
    call_idx = 0

    def fake_generate_quiz_local(**kwargs):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            cnt = kwargs["target_count"]
            qs = [_make_mock_question(i + 1) for i in range(cnt)]
            return {"title": "Test", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None
        else:
            return None, "LOCAL_CUDA_OOM: GPU memory exhausted."

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        quiz_dict, err = generate_quiz_local_batched(
            pdf_context="Context",
            target_count=15,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={"MCQ": 15},
            adapter_path="/fake",
        )

    # Partial results returned
    assert quiz_dict is not None
    assert len(quiz_dict["questions"]) == 8
    assert err is not None
    assert "Partial generation" in err
    assert "Batch 2" in err


def test_generate_quiz_local_batched_all_batches_fail():
    """All batches fail -> returns None and clear error message."""
    def fake_generate_quiz_local(**kwargs):
        return None, "LOCAL_INFERENCE_ERROR: Model failure"

    with patch("local_model_service.generate_quiz_local", side_effect=fake_generate_quiz_local):
        quiz_dict, err = generate_quiz_local_batched(
            pdf_context="Context",
            target_count=15,
            target_topic="Transformers",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={"MCQ": 15},
            adapter_path="/fake",
        )

    assert quiz_dict is None
    assert err is not None
    assert "failed for all batches" in err


# ===========================================================================
# 4. End-to-End QuizGenerator Integration
# ===========================================================================

def test_quiz_generator_local_batched_15_mcqs():
    """QuizGenerator in local mode requests 15 MCQs, batches them safely, and delivers GeneratedQuiz."""
    dm = DatasetManager("genquiz_dataset.csv")
    llm = LLMService(api_key=None, provider="local")

    def fake_generate_quiz_local_batched(**kwargs):
        cnt = kwargs["target_count"]
        qs = [_make_mock_question(i + 1) for i in range(cnt)]
        return {"title": "Transformers Assessment", "topic": "Transformers", "difficulty": "Medium", "questions": qs}, None

    gen = QuizGenerator(dm, llm)
    with patch("local_model_service.get_adapter_status", return_value={"is_available": True, "adapter_path": "/fake"}), \
         patch("local_model_service.generate_quiz_local_batched", side_effect=fake_generate_quiz_local_batched):
        quiz = gen.generate_quiz(
            pdf_text_or_dict="The Transformer relies on self-attention mechanisms to model contextual representations.",
            user_prompt="Generate 15 MCQs about Transformers. Use medium difficulty.",
            force_demo=False,
        )

    assert isinstance(quiz, GeneratedQuiz)
    assert len(quiz.questions) == 15
    assert quiz.topic == "Transformers"
    assert quiz.is_demo_mode is False
    assert all(q.question_type == "MCQ" for q in quiz.questions)
    assert quiz.questions[0].question_id == "Q1"
    assert quiz.questions[14].question_id == "Q15"
