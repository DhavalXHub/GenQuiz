"""
tests/test_grounding_and_alignment.py - Topic Grounding and Question-Answer Alignment Tests.

Verifies:
1. Transformers PDF + Transformers request generates valid questions (mocked).
2. Cybercrime PDF + Transformers request is blocked (TopicMismatchError).
3. Related terminology is not rejected when exact keyword is absent (semantic topic expansion).
4. Cybercrime question paired with positional-encoding answer is rejected.
5. Valid question-answer pair passes validation.
6. Valid MCQ requires 4 distinct options and matching correct_answer letter.
7. Malformed or placeholder output cannot be presented as a successful quiz.
8. Retry exhaustion produces a clear error.
9. Displayed question and expected answer remain paired throughout generation.
10. Evaluation and export use the validated question-answer pairs.
11. Cloud, local, and demo modes remain compatible with the repair.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from dataset import DatasetManager
from llm_service import LLMService
from pdf_processor import prioritize_topic_passages, check_topic_in_text, get_topic_related_terms
from quiz_generator import (
    QuizGenerator,
    GeneratedQuiz,
    GeneratedQuestion,
    TopicMismatchError,
    validate_question_answer_alignment,
)
from quiz_evaluator import QuizEvaluator
from export_utils import generate_assessment_pdf


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset_mgr():
    return DatasetManager("genquiz_dataset.csv")


def make_mock_pdf_result(pages_content: list, is_valid: bool = True):
    page_texts = [
        {"page": i + 1, "text": text, "char_count": len(text)}
        for i, text in enumerate(pages_content)
    ]
    full_text = "\n\n".join(pages_content)
    return {
        "is_valid": is_valid,
        "text": full_text,
        "page_count": len(pages_content),
        "page_texts": page_texts,
        "preview": full_text[:400],
        "truncated": False,
        "original_char_count": len(full_text),
        "final_char_count": len(full_text),
        "error_message": None,
    }


TRANSFORMERS_PDF_TEXT = [
    "The Transformer architecture replaces recurrent neural networks with self-attention mechanisms.",
    "Multi-head attention applies several attention functions in parallel. Positional encoding supplies token order.",
]

CYBERCRIME_PDF_TEXT = [
    "Cybercrime refers to illegal acts committed through computers and networks, including computer fraud.",
    "Cyberterrorism involves politically motivated attacks on information infrastructure and unauthorized access.",
]


# ===========================================================================
# 1. Transformers PDF + Transformers Request -> Success
# ===========================================================================

def test_transformers_pdf_with_transformers_request_generates_quiz(dataset_mgr):
    pdf_data = make_mock_pdf_result(TRANSFORMERS_PDF_TEXT)
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.provider = "groq"
    mock_llm.model = "llama-3.3-70b-versatile"
    mock_llm.has_active_credentials.return_value = True

    mock_llm_json = {
        "title": "Transformers Quiz",
        "topic": "Transformers",
        "difficulty": "Medium",
        "questions": [
            {
                "question_id": "Q1",
                "question": "What is the primary role of self-attention in Transformers?",
                "question_type": "MCQ",
                "options": {
                    "A": "Computes weighted representations across all sequence tokens",
                    "B": "Discards distant token representations",
                    "C": "Applies recurrent cell hidden state updates",
                    "D": "Performs spatial pooling over grid features",
                },
                "correct_answer": "A",
                "explanation": "Self-attention weighs token interactions directly without recurrence.",
                "topic": "Transformers",
                "difficulty": "Medium",
            },
            {
                "question_id": "Q2",
                "question": "How does positional encoding address the order-invariance of self-attention?",
                "question_type": "Short Answer",
                "options": None,
                "correct_answer": "By adding token order signals to input embeddings",
                "explanation": "Positional encodings provide sequence position information to attention layers.",
                "topic": "Transformers",
                "difficulty": "Medium",
            },
        ],
    }
    mock_llm.generate_json.return_value = (mock_llm_json, None)

    gen = QuizGenerator(dataset_mgr, mock_llm)
    quiz = gen.generate_quiz(
        pdf_text_or_dict=pdf_data,
        user_prompt="Generate 2 questions about Transformers. Include 1 MCQ and 1 short-answer.",
        force_demo=False,
    )

    assert isinstance(quiz, GeneratedQuiz)
    assert len(quiz.questions) == 2
    assert quiz.topic == "Transformers"
    assert quiz.is_demo_mode is False
    assert quiz.questions[0].question_type == "MCQ"
    assert quiz.questions[0].correct_answer == "A"
    assert quiz.questions[1].question_type == "Short Answer"


# ===========================================================================
# 2. Cybercrime PDF + Transformers Request -> Blocked (TopicMismatchError)
# ===========================================================================

def test_cybercrime_pdf_with_transformers_request_is_blocked(dataset_mgr):
    pdf_data = make_mock_pdf_result(CYBERCRIME_PDF_TEXT)
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.provider = "groq"
    mock_llm.has_active_credentials.return_value = True

    gen = QuizGenerator(dataset_mgr, mock_llm)

    with pytest.raises(TopicMismatchError) as exc_info:
        gen.generate_quiz(
            pdf_text_or_dict=pdf_data,
            user_prompt="Generate 5 questions about Transformers. Include 3 MCQs and 2 short-answer questions.",
            force_demo=False,
        )

    err = exc_info.value
    assert "Transformers" in err.topic
    assert "0 keyword matches" in err.warning or "does not appear to contain" in err.warning
    # Ensure LLM generate_json was NEVER called with mismatched context
    mock_llm.generate_json.assert_not_called()


# ===========================================================================
# 3. Related Terminology is Not Rejected (Semantic Topic Expansion)
# ===========================================================================

def test_related_terminology_not_rejected_when_exact_keyword_absent():
    # PDF discusses "self-attention" and "multi-head attention" but does not contain "Transformers"
    pdf_data = make_mock_pdf_result([
        "Self-attention computes a weighted representation for every token in parallel.",
        "Multi-head attention applies several attention heads simultaneously to improve representation.",
    ])

    prioritized = prioritize_topic_passages(pdf_data, topic="Transformers")
    # Should match via related terms "self-attention" and "multi-head"
    assert prioritized["has_topic_match"] is True
    assert prioritized["match_count"] > 0
    assert prioritized["warning"] is None


def test_get_topic_related_terms_includes_domain_synonyms():
    terms = get_topic_related_terms("Transformers")
    assert "transformer" in terms or "transformers" in terms
    assert "attention" in terms
    assert "self-attention" in terms

    cyber_terms = get_topic_related_terms("Cybercrime")
    assert "cyber" in cyber_terms or "cybercrime" in cyber_terms
    assert "malware" in cyber_terms
    assert "phishing" in cyber_terms


# ===========================================================================
# 4. Cybercrime Question + Positional Encoding Answer is Rejected
# ===========================================================================

def test_cybercrime_question_paired_with_positional_encoding_answer_rejected():
    cyber_question = "What is the primary difference between cybercrime and cyberterrorism?"
    mismatched_answer = "To give the model information about the order of tokens in positional encoding."
    cyber_context = "Cybercrime involves computer fraud. Cyberterrorism targets critical infrastructure."

    with pytest.raises(ValueError) as exc_info:
        validate_question_answer_alignment(
            question=cyber_question,
            correct_answer=mismatched_answer,
            qtype="Short Answer",
            context=cyber_context,
        )

    err_msg = str(exc_info.value)
    assert "mismatch" in err_msg.lower() or "ungrounded" in err_msg.lower()


# ===========================================================================
# 5. Valid Question-Answer Pair Passes Validation
# ===========================================================================

def test_valid_question_answer_pair_passes():
    q = "What is the primary difference between cybercrime and cyberterrorism?"
    a = "Cybercrime is motivated by financial gain while cyberterrorism is politically motivated."
    ctx = "Cybercrime is motivated by financial gain. Cyberterrorism is politically motivated."

    # Should not raise any ValueError
    validate_question_answer_alignment(
        question=q,
        correct_answer=a,
        qtype="Short Answer",
        context=ctx,
    )


# ===========================================================================
# 6. Valid MCQ Requires 4 Distinct Options and Matching Correct Option
# ===========================================================================

def test_valid_mcq_passes_validation():
    opts = {
        "A": "Computes weighted representations across all sequence tokens",
        "B": "Discards distant token representations",
        "C": "Applies recurrent cell hidden state updates",
        "D": "Performs spatial pooling over grid features",
    }
    validate_question_answer_alignment(
        question="What is the primary role of self-attention?",
        correct_answer="A",
        qtype="MCQ",
        options=opts,
    )


def test_mcq_with_invalid_answer_letter_rejected():
    opts = {"A": "Option One", "B": "Option Two", "C": "Option Three", "D": "Option Four"}
    with pytest.raises(ValueError, match="correct_answer"):
        validate_question_answer_alignment(
            question="What is the primary role of self-attention?",
            correct_answer="E",
            qtype="MCQ",
            options=opts,
        )


def test_mcq_with_duplicate_options_rejected():
    opts = {"A": "Identical text", "B": "Identical text", "C": "Distinct three", "D": "Distinct four"}
    with pytest.raises(ValueError, match="distinct|duplicate"):
        validate_question_answer_alignment(
            question="What is the primary role of self-attention?",
            correct_answer="A",
            qtype="MCQ",
            options=opts,
        )


def test_mcq_with_generic_options_rejected():
    opts = {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"}
    with pytest.raises(ValueError, match="generic placeholder"):
        validate_question_answer_alignment(
            question="What is the primary role of self-attention?",
            correct_answer="A",
            qtype="MCQ",
            options=opts,
        )


# ===========================================================================
# 7. Malformed or Placeholder Output Cannot Pass as Successful Quiz
# ===========================================================================

def test_placeholder_question_cannot_pass():
    with pytest.raises(ValueError, match="placeholder|short"):
        GeneratedQuestion(
            question_id="Q1",
            question="...",
            question_type="MCQ",
            options={"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
            correct_answer="A",
            explanation="Explanation",
            topic="Topic",
            difficulty="Medium",
        )


# ===========================================================================
# 8. All Invalid Questions Cause Clear Failure (No Silent Substitution)
# ===========================================================================

def test_all_invalid_questions_raise_clear_error(dataset_mgr):
    pdf_data = make_mock_pdf_result(TRANSFORMERS_PDF_TEXT)
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.provider = "groq"
    mock_llm.has_active_credentials.return_value = True

    # LLM returns question with generic options
    mock_llm_json = {
        "title": "Bad Quiz",
        "topic": "Transformers",
        "difficulty": "Medium",
        "questions": [
            {
                "question_id": "Q1",
                "question": "What is self-attention?",
                "question_type": "MCQ",
                "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
                "correct_answer": "A",
                "explanation": "...",
                "topic": "Transformers",
                "difficulty": "Medium",
            }
        ],
    }
    mock_llm.generate_json.return_value = (mock_llm_json, None)

    gen = QuizGenerator(dataset_mgr, mock_llm)
    with pytest.raises(ValueError, match="0 valid questions"):
        gen.generate_quiz(
            pdf_text_or_dict=pdf_data,
            user_prompt="Generate 1 question about Transformers",
            force_demo=False,
        )


# ===========================================================================
# 9. Question and Expected Answer Remain Correctly Paired
# ===========================================================================

def test_question_answer_pairs_remain_paired_throughout():
    q1 = GeneratedQuestion(
        question_id="Q1",
        question="What is the difference between cybercrime and cyberterrorism?",
        question_type="Short Answer",
        correct_answer="Cybercrime seeks financial gain; cyberterrorism seeks political influence.",
        explanation="The primary distinction is motivation.",
        topic="Cybersecurity",
        difficulty="Medium",
    )
    quiz = GeneratedQuiz(
        title="Cybersecurity Assessment",
        topic="Cybersecurity",
        difficulty="Medium",
        questions=[q1],
        source_grounding="Authoritative uploaded PDF",
    )

    # Question and answer are bound within the same object
    assert quiz.questions[0].question == "What is the difference between cybercrime and cyberterrorism?"
    assert quiz.questions[0].correct_answer == "Cybercrime seeks financial gain; cyberterrorism seeks political influence."

    # Evaluate with user answer
    user_answers = {"Q1": "Cybercrime is for financial profit, whereas cyberterrorism has political goals."}
    eval_res = QuizEvaluator.evaluate_quiz(quiz, user_answers)

    det = eval_res["detailed_results"][0]
    assert det["question"] == quiz.questions[0].question
    assert det["correct_answer"] == quiz.questions[0].correct_answer
    assert det["score"] > 0.0


# ===========================================================================
# 10. Evaluation and Export Use Validated Question-Answer Pairs
# ===========================================================================

def test_evaluation_and_export_use_validated_pairs():
    q1 = GeneratedQuestion(
        question_id="Q1",
        question="Which component of the Transformer encodes sequence order?",
        question_type="MCQ",
        options={
            "A": "Positional encoding",
            "B": "Multi-head attention",
            "C": "Feed-forward layer",
            "D": "Layer normalization",
        },
        correct_answer="A",
        explanation="Positional encoding adds position vectors to token embeddings.",
        topic="Transformers",
        difficulty="Medium",
    )
    quiz = GeneratedQuiz(
        title="Transformers Assessment",
        topic="Transformers",
        difficulty="Medium",
        questions=[q1],
        source_grounding="PDF Document",
    )

    eval_res = QuizEvaluator.evaluate_quiz(quiz, {"Q1": "A"})
    assert eval_res["earned_points"] == 1.0

    # Generate assessment PDF
    pdf_bytes = generate_assessment_pdf(quiz, eval_res)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


# ===========================================================================
# 11. Compatibility Across Cloud, Local, and Demo Modes
# ===========================================================================

def test_demo_mode_compatibility(dataset_mgr):
    llm = LLMService(api_key=None, provider="demo")
    gen = QuizGenerator(dataset_mgr, llm)

    quiz = gen.generate_quiz(
        pdf_text_or_dict="Any text",
        user_prompt="Generate 3 questions about Transformers",
        force_demo=True,
    )
    assert quiz.is_demo_mode is True
    assert len(quiz.questions) == 3
    for q in quiz.questions:
        assert q.is_saved_dataset is True
        assert q.question_id.startswith("SAVED-")
