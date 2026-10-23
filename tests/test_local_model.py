"""
test_local_model.py - Unit Tests for Local PEFT/LoRA Model Service and Provider Integration.
"""

import os
import json
import tempfile
import pytest
from local_model_service import (
    LOCAL_MAX_INPUT_TOKENS,
    LOCAL_MAX_NEW_TOKENS,
    LOCAL_MAX_QUESTIONS,
    _is_cuda_oom,
    get_adapter_status,
    generate_quiz_local,
    load_cached_peft_model,
    resolve_adapter_path,
    clean_local_json_text
)
import local_model_service
from llm_service import LLMService
from quiz_generator import QuizGenerator
from dataset import DatasetManager


def test_clean_local_json_text():
    # Markdown code fences with json tag
    raw1 = "```json\n{\"question\": \"Test?\", \"correct_answer\": \"A\"}\n```"
    assert clean_local_json_text(raw1) == '{"question": "Test?", "correct_answer": "A"}'

    # Fences without json tag
    raw2 = "```\n{\"question\": \"Test?\"}\n```"
    assert clean_local_json_text(raw2) == '{"question": "Test?"}'

    # Conversational text surrounding JSON
    raw3 = "Here is the requested question:\n{\"question\": \"Test?\"}\nI hope this helps!"
    assert clean_local_json_text(raw3) == '{"question": "Test?"}'


def test_clean_local_json_text_rejects_ambiguous_multiple_objects():
    raw = '{"question": "First"}\n{"question": "Second"}'
    assert clean_local_json_text(raw) == raw


def test_local_question_schema_accepts_mcq_and_requires_answer_key():
    valid = {
        "question_id": "Q1",
        "question": "Which mechanism combines token information?",
        "question_type": "MCQ",
        "options": {"A": "Attention", "B": "Pooling", "C": "Sorting", "D": "Hashing"},
        "correct_answer": "A",
        "explanation": "Attention combines information across positions.",
        "topic": "Transformers",
        "difficulty": "Easy",
    }
    parsed = local_model_service._validate_local_question(valid, "MCQ", "Transformers", "Easy")
    assert parsed["correct_answer"] == "A"

    invalid = dict(valid, correct_answer="Attention")
    with pytest.raises(ValueError, match="correct_answer"):
        local_model_service._validate_local_question(invalid, "MCQ", "Transformers", "Easy")


def test_adapter_status_missing():
    status = get_adapter_status(custom_path="non_existent_adapter_dir_12345")
    assert status["is_available"] is False
    assert status["has_weights"] is False
    assert "non_existent_adapter_dir_12345" in status["adapter_path"]
    assert "not found" in status["message"].lower()


def test_adapter_status_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_cfg = os.path.join(tmpdir, "adapter_config.json")
        with open(adapter_cfg, "w", encoding="utf-8") as f:
            json.dump({
                "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "r": 16,
                "lora_alpha": 32
            }, f)

        # Create dummy weights file
        with open(os.path.join(tmpdir, "adapter_model.safetensors"), "w") as f:
            f.write("mock_weights")

        status = get_adapter_status(custom_path=tmpdir)
        assert status["is_available"] is True
        assert status["has_weights"] is True
        assert status["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_llm_service_local_provider_initialization():
    service = LLMService(provider="Fine-Tuned Local Model (QLoRA)")
    assert service.provider == "local"

    info = service.get_active_provider_info()
    assert "Fine-Tuned Local Model" in info["provider"]
    assert "LoRA Adapter" in info["model"]


def test_local_model_missing_adapter_raises_clean_error():
    service = LLMService(provider="local")
    # If no local adapter is on disk, generate_json must return a clean actionable error
    # and MUST NOT silently call Groq or OpenAI
    if not service.has_active_credentials():
        parsed, err = service.generate_json("System prompt", "User prompt")
        assert parsed is None
        assert err is not None
        assert "LORA_ADAPTER_NOT_FOUND" in err
        assert "GenQuiz_FineTuning.ipynb" in err


def test_quiz_generator_local_mode_missing_adapter():
    dm = DatasetManager("genquiz_dataset.csv")
    service = LLMService(provider="local")

    if not service.has_active_credentials():
        generator = QuizGenerator(dm, service)
        with pytest.raises(RuntimeError) as exc_info:
            generator.generate_quiz(
                pdf_text_or_dict="Sample PDF text on Transformers and Attention.",
                user_prompt="Generate 3 questions about Transformers.",
                force_demo=False
            )
        assert "LORA_ADAPTER_NOT_FOUND" in str(exc_info.value)
        assert "GenQuiz_FineTuning.ipynb" in str(exc_info.value)


def test_local_limits_are_conservative():
    assert LOCAL_MAX_INPUT_TOKENS <= 2048
    assert LOCAL_MAX_NEW_TOKENS <= 384
    assert LOCAL_MAX_QUESTIONS <= 8


def test_mcq_prompt_requires_options_object():
    system, user = local_model_service._question_prompt(
        "Attention combines token information.",
        "Transformers",
        "MCQ",
        "Easy",
        "Understand",
    )
    assert "Never output null for options" in user
    assert '"A":"First"' in user


def test_short_answer_prompt_uses_short_answer_schema():
    _, user = local_model_service._question_prompt(
        "Attention combines token information.",
        "Transformers",
        "Short Answer",
        "Medium",
        "Understand",
    )
    assert '"question_type":"Short Answer"' in user
    assert '"options":null' in user
    assert "Never output null for options" not in user


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded output"


class _FakeIds:
    shape = (1, 3)

    def __getitem__(self, item):
        return self


def test_decode_generated_text_slices_prompt_tokens():
    outputs = [_FakeIds()]
    text, info = local_model_service._decode_generated_text(
        outputs, input_token_count=2, tokenizer=_FakeTokenizer(), max_new_tokens=4
    )
    assert text == "decoded output"
    assert info["input_tokens"] == 2
    assert info["generated_tokens"] == 3
    assert info["hit_token_limit"] is False


class _EmptyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return ""


def test_decode_generated_text_reports_empty_and_limit():
    outputs = [_FakeIds()]
    _, info = local_model_service._decode_generated_text(
        outputs, input_token_count=0, tokenizer=_EmptyTokenizer(), max_new_tokens=3
    )
    assert info["empty"] is True
    assert info["generated_tokens"] == 3
    assert info["hit_token_limit"] is True


def test_cuda_oom_detection():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not _is_cuda_oom(RuntimeError("invalid JSON"))


def test_local_rejects_base_model_mismatch(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "expected/model"}),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"mock")
    with pytest.raises(ValueError, match="LORA_BASE_MODEL_MISMATCH"):
        load_cached_peft_model("different/model", str(tmp_path))


def test_local_rejects_oversized_request():
    result, error = generate_quiz_local(
        pdf_context="A short context.",
        target_count=LOCAL_MAX_QUESTIONS + 1,
        target_topic="Transformers",
        target_diff="Easy",
        target_types=["MCQ"],
        type_counts={},
        adapter_path="missing",
    )
    assert result is None
    assert error is not None
    assert "LOCAL_REQUEST_TOO_LARGE" in error


def test_local_retries_malformed_output_and_returns_exact_count(monkeypatch):
    valid = {
        "question_id": "model-id",
        "question": "What does attention combine?",
        "question_type": "MCQ",
        "options": {"A": "Token information", "B": "Images", "C": "Audio", "D": "Labels"},
        "correct_answer": "A",
        "explanation": "The passage states that attention combines token information.",
        "topic": "Transformers",
        "difficulty": "Easy",
    }
    responses = [None, valid, valid]
    calls = []

    monkeypatch.setattr(local_model_service, "load_cached_peft_model", lambda adapter_path=None: (object(), object()))

    def fake_generate(**kwargs):
        calls.append(kwargs["correction"])
        response = responses.pop(0)
        if response is None:
            return None, "Local model produced invalid JSON"
        return dict(response, question=f"{response['question']} {len(calls)}"), None

    monkeypatch.setattr(local_model_service, "generate_single_question_local", fake_generate)
    result, error = local_model_service.generate_quiz_local(
        "Transformers use attention.\n\nTransformers use positional encoding.",
        2, "Transformers", "Easy", ["MCQ"], {}, "adapter"
    )
    assert error is None
    assert result is not None
    assert len(result["questions"]) == 2
    assert calls[0] is None
    assert calls[1] is not None
    assert len({q["question"] for q in result["questions"]}) == 2


def test_local_reports_honest_shortfall_after_retries(monkeypatch):
    monkeypatch.setattr(local_model_service, "load_cached_peft_model", lambda adapter_path=None: (object(), object()))
    monkeypatch.setattr(
        local_model_service,
        "generate_single_question_local",
        lambda **kwargs: (None, "schema-invalid"),
    )
    result, error = local_model_service.generate_quiz_local(
        "Transformers use attention.", 1, "Transformers", "Easy", ["MCQ"], {}, "adapter"
    )
    assert result is None
    assert error is not None
    assert "returned 0 of 1" in error
    assert "schema-invalid" in error


def test_local_rejects_duplicate_questions(monkeypatch):
    valid = {
        "question_id": "model-id",
        "question": "What does attention combine?",
        "question_type": "MCQ",
        "options": {"A": "Token information", "B": "Images", "C": "Audio", "D": "Labels"},
        "correct_answer": "A",
        "explanation": "Attention combines information across positions.",
        "topic": "Transformers",
        "difficulty": "Easy",
    }
    monkeypatch.setattr(local_model_service, "load_cached_peft_model", lambda adapter_path=None: (object(), object()))
    monkeypatch.setattr(
        local_model_service,
        "generate_single_question_local",
        lambda **kwargs: (valid, None),
    )
    result, error = local_model_service.generate_quiz_local(
        "Transformers use attention.\n\nTransformers use attention.",
        2, "Transformers", "Easy", ["MCQ"], {}, "adapter"
    )
    assert result is None
    assert error is not None
    assert "returned 1 of 2" in error
    assert "duplicate question text" in error
