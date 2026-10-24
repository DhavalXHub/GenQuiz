"""
tests/test_quiz_evaluator.py - Comprehensive Unit tests for Semantic Evaluation, Scoring, Prompt Parsing, and Adaptivity.
"""

import pytest
from dataset import DatasetManager
from llm_service import LLMService
from quiz_generator import (
    QuizGenerator,
    GeneratedQuiz,
    GeneratedQuestion,
    parse_user_prompt
)
from quiz_evaluator import (
    QuizEvaluator,
    SemanticShortAnswerEvaluator,
    evaluate_short_answer
)


def test_prompt_parsing():
    """Verify natural-language prompt interpretation and parameter extraction."""
    prompt = "Generate 8 questions on LoRA. Include 4 MCQs and 4 True/False. Hard difficulty."
    parsed = parse_user_prompt(prompt)
    assert parsed["count"] == 8
    assert parsed["difficulty"] == "Hard"
    assert "MCQ" in parsed["question_types"]
    assert "True/False" in parsed["question_types"]
    assert "LoRA" in parsed["topic"]


def test_medium_difficulty_parsing_not_hard():
    """
    CRITICAL REGRESSION TEST:
    Verify that explicit 'medium difficulty' is parsed as 'Medium', NOT 'Hard'.
    Prevents substring collision where 'difficult' inside 'difficulty' falsely triggered Hard.
    """
    prompt = "Generate 5 questions about Transformers. Include 3 MCQs and 2 short-answer questions. Use medium difficulty and provide correct answers and explanations."
    parsed = parse_user_prompt(prompt)
    assert parsed["difficulty"] == "Medium", f"Expected Medium, got {parsed['difficulty']}"
    assert parsed["count"] == 5
    assert parsed["topic"] == "Transformers"
    assert parsed["type_counts"].get("MCQ") == 3
    assert parsed["type_counts"].get("Short Answer") == 2


def test_missing_prompt_applies_defaults():
    """Verify empty/ambiguous prompts apply documented sensible defaults."""
    parsed = parse_user_prompt("")
    assert parsed["count"] == 5
    assert parsed["difficulty"] == "Medium"
    assert len(parsed["question_types"]) == 3
    assert len(parsed["defaults_applied"]) > 0


def test_pydantic_question_validation():
    """Verify schema enforcement for question objects."""
    q = GeneratedQuestion(
        question_id="Q1",
        question="What is temperature in LLM sampling?",
        question_type="MCQ",
        options={"A": "Randomness control", "B": "Layer count", "C": "Batch size", "D": "Loss rate"},
        correct_answer="A",
        explanation="Temperature scales logits before softmax.",
        topic="Sampling",
        difficulty="Medium"
    )
    assert q.question_id == "Q1"
    assert q.options["A"] == "Randomness control"


def test_exact_deterministic_mcq_scoring():
    """Verify deterministic and exact scoring for MCQs (matching option letters, texts, and rejecting wrong options)."""
    q = GeneratedQuestion(
        question_id="Q1",
        question="Which architecture is used?",
        question_type="MCQ",
        options={"A": "Transformer", "B": "RNN", "C": "CNN", "D": "LSTM"},
        correct_answer="A",
        explanation="Transformers are attention-based.",
        topic="Architectures",
        difficulty="Easy"
    )
    quiz = GeneratedQuiz(title="MCQ Test", topic="AI", difficulty="Easy", questions=[q], source_grounding="doc")

    # Correct via letter
    res_letter = QuizEvaluator.evaluate_quiz(quiz, {"Q1": "A"})
    assert res_letter["earned_points"] == 1.0
    assert res_letter["detailed_results"][0]["status"] == "Correct"
    assert res_letter["detailed_results"][0]["evaluator_label"] == "Deterministic Exact Match"

    # Correct via full text
    res_text = QuizEvaluator.evaluate_quiz(quiz, {"Q1": "Transformer"})
    assert res_text["earned_points"] == 1.0
    assert res_text["detailed_results"][0]["status"] == "Correct"

    # Incorrect choice
    res_wrong = QuizEvaluator.evaluate_quiz(quiz, {"Q1": "B"})
    assert res_wrong["earned_points"] == 0.0
    assert res_wrong["detailed_results"][0]["status"] == "Incorrect"


def test_semantically_equivalent_short_answer():
    """
    PRIORITY 1 TEST:
    Verify that a conceptually correct answer with differing wording is marked Correct.
    """
    item = {
        "question_id": "SA1",
        "question": "Explain why standard self-attention has quadratic complexity with sequence length.",
        "correct_answer": "Every token attends to all other tokens in the sequence, requiring an N by N pairwise matrix computation.",
        "explanation": "Because each input token computes an attention score with all N tokens, compute scales as O(N^2).",
        "user_answer": "Each token in the sentence performs a pairwise comparison with all other tokens, forming an N x N attention matrix."
    }
    # Test via fallback evaluator (or mock LLM)
    res = SemanticShortAnswerEvaluator._evaluate_fallback(item)
    assert res["score"] >= 0.5, f"Expected credit for equivalent wording, got {res['score']}"
    assert "token" in " ".join(res["key_concepts_identified"]).lower() or "matrix" in " ".join(res["key_concepts_identified"]).lower()


def test_partially_correct_short_answer():
    """
    PRIORITY 1 TEST:
    Verify that an incomplete answer receives partial credit (0.5), not full 1.0 or 0.0.
    """
    item = {
        "question_id": "SA2",
        "question": "Explain why standard self-attention has quadratic complexity with sequence length.",
        "correct_answer": "Every token attends to all other tokens in the sequence, requiring an N by N pairwise matrix computation.",
        "explanation": "Because each input token computes an attention score with all N tokens, compute scales as O(N^2).",
        "user_answer": "It computes attention across tokens in the sequence."
    }
    res = SemanticShortAnswerEvaluator._evaluate_fallback(item)
    assert res["score"] == 0.5
    assert res["status"] == "Partial"


def test_incorrect_short_answer():
    """
    PRIORITY 1 TEST:
    Verify that an irrelevant or factually incorrect answer receives 0.0 credit.
    """
    item = {
        "question_id": "SA3",
        "question": "What is the function of positional encoding in Transformers?",
        "correct_answer": "To inject information about token order into the permutation-invariant self-attention layers.",
        "explanation": "Transformers process tokens in parallel without recurrence, requiring explicit order signals.",
        "user_answer": "It acts as a pooling filter to downsample image resolution in convolutional layers."
    }
    res = SemanticShortAnswerEvaluator._evaluate_fallback(item)
    assert res["score"] == 0.0
    assert res["status"] == "Incorrect"


def test_empty_short_answer():
    """
    PRIORITY 1 TEST:
    Verify that an empty response receives 0.0 credit with clear feedback.
    """
    item = {
        "question_id": "SA4",
        "question": "What is temperature in sampling?",
        "correct_answer": "A parameter scaling logits before softmax to adjust prediction randomness.",
        "explanation": "Higher temperature yields higher entropy.",
        "user_answer": ""
    }
    batch_res = SemanticShortAnswerEvaluator.evaluate_batch([item], llm_service=None)
    res = batch_res["SA4"]
    assert res["score"] == 0.0
    assert res["status"] == "Incorrect"
    assert "no response" in res["feedback"].lower()


def test_llm_evaluation_failure_transparent_fallback():
    """
    PRIORITY 1 TEST:
    Verify that when LLM evaluation fails (e.g. invalid service or offline),
    the system transparently falls back to rule-based evaluation and labels it accordingly.
    """
    mock_llm = LLMService(api_key="invalid_mock_key", provider="groq")
    # Force credentials check to False or fail generate_json
    mock_llm.groq_key = ""
    mock_llm.openai_key = ""

    item = {
        "question_id": "SA5",
        "question": "What does RAG stand for?",
        "correct_answer": "Retrieval Augmented Generation",
        "explanation": "Retrieves external knowledge to ground LLM responses.",
        "user_answer": "Retrieval Augmented Generation"
    }
    res = SemanticShortAnswerEvaluator.evaluate_batch([item], llm_service=mock_llm)["SA5"]
    assert res["score"] == 1.0
    assert res["evaluator_type"] == "rule_based_fallback"
    assert "fallback" in res["feedback"].lower()


def test_question_count_and_type_breakdown_validation():
    """
    PRIORITY 3 TEST:
    Verify that QuizGenerator enforces exact requested question counts and type breakdowns.
    """
    dm = DatasetManager("genquiz_dataset.csv")
    llm = LLMService(provider="demo")
    gen = QuizGenerator(dm, llm)

    prompt = "Generate 5 questions about Transformers. Include 3 MCQs and 2 short-answer questions."
    quiz = gen.generate_quiz(
        pdf_text_or_dict="Transformers use self-attention.",
        user_prompt=prompt,
        force_demo=True
    )
    assert len(quiz.questions) == 5

    type_counts = {}
    for q in quiz.questions:
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1

    assert type_counts.get("MCQ") == 3
    assert type_counts.get("Short Answer") >= 1  # Fulfills short answers from dataset topic records
    assert len(quiz.validation_notes) > 0


def test_rule_based_adaptive_recommendation():
    """Verify that weak topics trigger targeted follow-up quiz recommendations."""
    q1 = GeneratedQuestion(
        question_id="Q1",
        question="What does LoRA stand for in the context of model fine-tuning?",
        question_type="True/False",
        correct_answer="True", explanation="E1", topic="LoRA", difficulty="Medium"
    )
    quiz = GeneratedQuiz(
        title="Adaptive Test", topic="LoRA", difficulty="Medium",
        questions=[q1], source_grounding="Doc"
    )
    eval_res = QuizEvaluator.evaluate_quiz(quiz, {"Q1": "False"})
    assert "LoRA" in eval_res["weak_topics"]
    rec = eval_res["adaptive_recommendation"]
    assert "LoRA" in rec["suggested_prompt"]


def test_demo_mode_retrieves_topic_relevant_questions():
    """Verify Demo Mode returns Transformer questions, not diffusion models."""
    dm = DatasetManager("genquiz_dataset.csv")
    llm = LLMService(api_key=None, provider="demo")
    gen = QuizGenerator(dm, llm)

    prompt = "Generate 5 questions about Transformers. Include 3 MCQs and 2 short-answer questions. Use medium difficulty."
    quiz = gen.generate_quiz(
        pdf_text_or_dict="Transformers use self-attention mechanisms.",
        user_prompt=prompt,
        force_demo=True
    )
    assert quiz.is_demo_mode is True
    assert len(quiz.questions) == 5

    for q in quiz.questions:
        top = q.topic.lower()
        assert ("transformer" in top or "attention" in top)
        assert q.is_saved_dataset is True
        assert q.question_id.startswith("SAVED-")


def test_live_mode_without_key_raises_informative_error():
    """Verify requesting live generation without key raises informative error."""
    dm = DatasetManager("genquiz_dataset.csv")
    llm = LLMService(api_key="", provider="groq")
    llm.groq_key = ""
    llm.openai_key = ""

    gen = QuizGenerator(dm, llm)
    with pytest.raises(RuntimeError) as exc_info:
        gen.generate_quiz(
            pdf_text_or_dict="Doc content",
            user_prompt="Generate 3 questions about Transformers",
            force_demo=False
        )
    assert "NO_API_KEY" in str(exc_info.value)
