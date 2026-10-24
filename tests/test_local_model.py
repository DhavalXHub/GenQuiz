"""
test_local_model.py - Comprehensive tests for local PEFT/LoRA service and validation.

Covers:
- JSON cleaning / extraction
- Adapter status detection (present / missing)
- _validate_local_question: all question types, placeholder rejection,
  generic MCQ option rejection, duplicate option rejection
- _is_placeholder and _is_generic_option helpers
- _question_prompt: confirms NO "..." appears in the schema example
- generate_single_question_local: mocked model behaviour
- generate_quiz_local: mocked model, partial-result policy
- LLMService initialization for local provider
- quiz_generator.py: Pydantic rejects placeholder question text
- Missing-adapter error propagation
"""

import json
import os
import sys
import tempfile
import types
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from local_model_service import (
    _is_generic_option,
    _is_placeholder,
    _question_prompt,
    _validate_local_question,
    clean_local_json_text,
    get_adapter_status,
    resolve_adapter_path,
)
from llm_service import LLMService
from quiz_generator import GeneratedQuestion, QuizGenerator
from dataset import DatasetManager


# ===========================================================================
# JSON cleaning
# ===========================================================================

class TestCleanLocalJsonText:
    def test_strips_json_fence(self):
        raw = "```json\n{\"question\": \"Test?\"}\n```"
        assert clean_local_json_text(raw) == '{"question": "Test?"}'

    def test_strips_plain_fence(self):
        raw = "```\n{\"question\": \"Test?\"}\n```"
        assert clean_local_json_text(raw) == '{"question": "Test?"}'

    def test_extracts_json_from_surrounding_text(self):
        raw = "Here is your question:\n{\"question\": \"What is X?\"}\nHope this helps!"
        assert clean_local_json_text(raw) == '{"question": "What is X?"}'

    def test_passthrough_clean_json(self):
        raw = '{"a": 1}'
        assert clean_local_json_text(raw) == '{"a": 1}'

    def test_empty_string(self):
        assert clean_local_json_text("") == ""

    def test_no_braces_returns_stripped(self):
        raw = "No JSON here"
        assert clean_local_json_text(raw) == "No JSON here"


# ===========================================================================
# Placeholder / generic option detection
# ===========================================================================

class TestPlaceholderDetection:
    def test_ellipsis_is_placeholder(self):
        assert _is_placeholder("...")
        assert _is_placeholder("…")

    def test_empty_string_is_placeholder(self):
        assert _is_placeholder("")

    def test_none_is_placeholder(self):
        assert _is_placeholder(None)

    def test_real_question_not_placeholder(self):
        assert not _is_placeholder("What is the role of attention in Transformers?")

    def test_first_is_generic_option(self):
        assert _is_generic_option("First")
        assert _is_generic_option("first")
        assert _is_generic_option("FIRST")

    def test_second_third_fourth_are_generic(self):
        for word in ("Second", "Third", "Fourth", "SECOND", "third"):
            assert _is_generic_option(word), f"Expected '{word}' to be generic"

    def test_option_a_label_is_generic(self):
        assert _is_generic_option("Option A")
        assert _is_generic_option("option a")

    def test_ellipsis_is_generic_option(self):
        assert _is_generic_option("...")

    def test_specific_option_is_not_generic(self):
        assert not _is_generic_option("The Transformer encoder uses multi-head attention")

    def test_short_option_below_min_len(self):
        assert _is_generic_option("AB")   # 2 chars < min 3


# ===========================================================================
# _question_prompt — no placeholders in schema text
# ===========================================================================

class TestQuestionPrompt:
    def _check_no_placeholders_in_schema(self, user_prompt: str):
        # The schema portion must not contain the old stub values
        bad_patterns = ['"question":"..."', '"question": "..."',
                        '"A":"First"', '"A": "First"',
                        '"B":"Second"', '"B": "Second"',
                        '"C":"Third"', '"C": "Third"',
                        '"D":"Fourth"', '"D": "Fourth"',
                        '"correct_answer":"..."', '"correct_answer": "..."',
                        '"explanation":"..."', '"explanation": "..."']
        for pat in bad_patterns:
            assert pat not in user_prompt, (
                f"Found forbidden placeholder pattern in prompt: {pat!r}"
            )

    def test_mcq_prompt_no_stub_placeholders(self):
        _, user_prompt = _question_prompt(
            "Context about attention", "Attention", "MCQ", "Medium", "Understand"
        )
        self._check_no_placeholders_in_schema(user_prompt)

    def test_short_answer_prompt_no_stub_placeholders(self):
        _, user_prompt = _question_prompt(
            "Context text", "Topic", "Short Answer", "Easy", "Remember"
        )
        self._check_no_placeholders_in_schema(user_prompt)

    def test_true_false_prompt_no_stub_placeholders(self):
        _, user_prompt = _question_prompt(
            "Context text", "Topic", "True/False", "Hard", "Analyze"
        )
        self._check_no_placeholders_in_schema(user_prompt)

    def test_mcq_prompt_contains_context(self):
        ctx = "Neural networks learn representations from data."
        _, user_prompt = _question_prompt(ctx, "ML", "MCQ", "Medium", "Apply")
        assert ctx in user_prompt

    def test_correction_appended_when_given(self):
        _, user_prompt = _question_prompt(
            "Some context", "Topic", "MCQ", "Medium", "Understand",
            correction="question was too vague"
        )
        assert "question was too vague" in user_prompt


# ===========================================================================
# _validate_local_question
# ===========================================================================

def _good_mcq(**overrides) -> Dict[str, Any]:
    base = {
        "question_id": "Q1",
        "question": "What mechanism does Transformer use to attend to all tokens?",
        "question_type": "MCQ",
        "options": {
            "A": "Self-attention",
            "B": "Recurrent layers",
            "C": "Convolution filters",
            "D": "Max-pooling",
        },
        "correct_answer": "A",
        "explanation": "Transformers use self-attention to relate tokens.",
        "topic": "Transformers",
        "difficulty": "Medium",
    }
    base.update(overrides)
    return base


class TestValidateLocalQuestion:

    # ── Valid cases ──────────────────────────────────────────────────────────

    def test_valid_mcq_passes(self):
        result = _validate_local_question(_good_mcq(), "MCQ", "Transformers", "Medium")
        assert result["correct_answer"] == "A"
        assert result["question_type"] == "MCQ"

    def test_valid_mcq_lowercase_answer_normalized(self):
        q = _good_mcq(correct_answer="b")
        result = _validate_local_question(q, "MCQ", "Transformers", "Medium")
        assert result["correct_answer"] == "B"

    def test_valid_true_false(self):
        q = {
            "question_id": "Q1",
            "question": "True or False: Self-attention uses queries, keys, and values.",
            "question_type": "True/False",
            "options": None,
            "correct_answer": "True",
            "explanation": "Self-attention does use Q, K, V matrices.",
            "topic": "Attention",
            "difficulty": "Easy",
        }
        result = _validate_local_question(q, "True/False", "Attention", "Easy")
        assert result["correct_answer"] == "True"

    def test_valid_short_answer(self):
        q = {
            "question_id": "Q1",
            "question": "What is the purpose of positional encoding in Transformers?",
            "question_type": "Short Answer",
            "options": None,
            "correct_answer": "To inject position information into token embeddings.",
            "explanation": "Without positional encoding the model cannot distinguish order.",
            "topic": "Transformers",
            "difficulty": "Medium",
        }
        result = _validate_local_question(q, "Short Answer", "Transformers", "Medium")
        assert "position" in result["correct_answer"]

    # ── Placeholder question text ────────────────────────────────────────────

    def test_ellipsis_question_rejected(self):
        q = _good_mcq(question="...")
        with pytest.raises(ValueError, match="placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_unicode_ellipsis_question_rejected(self):
        q = _good_mcq(question="…")
        with pytest.raises(ValueError, match="placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_empty_question_rejected(self):
        q = _good_mcq(question="")
        with pytest.raises(ValueError, match="Missing required field"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_short_question_rejected(self):
        q = _good_mcq(question="Q?")
        with pytest.raises(ValueError, match="too short"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── Generic / stub MCQ options ───────────────────────────────────────────

    def test_first_second_third_fourth_options_rejected(self):
        q = _good_mcq(options={"A": "First", "B": "Second", "C": "Third", "D": "Fourth"})
        with pytest.raises(ValueError, match="generic placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_option_a_label_options_rejected(self):
        q = _good_mcq(options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"})
        with pytest.raises(ValueError, match="generic placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_empty_option_rejected(self):
        q = _good_mcq(options={"A": "", "B": "Something", "C": "Another", "D": "Last"})
        with pytest.raises(ValueError, match="generic placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_ellipsis_option_rejected(self):
        q = _good_mcq(options={"A": "...", "B": "Real choice", "C": "Another", "D": "Last"})
        with pytest.raises(ValueError, match="generic placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── Duplicate MCQ options ────────────────────────────────────────────────

    def test_duplicate_options_rejected(self):
        q = _good_mcq(options={"A": "Self-attention", "B": "Self-attention", "C": "Something", "D": "Another"})
        with pytest.raises(ValueError, match="distinct"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── Invalid answer letters ───────────────────────────────────────────────

    def test_mcq_answer_not_a_letter_rejected(self):
        q = _good_mcq(correct_answer="Self-attention")
        with pytest.raises(ValueError, match="one of A, B, C, or D"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_mcq_answer_e_rejected(self):
        q = _good_mcq(correct_answer="E")
        with pytest.raises(ValueError, match="one of A, B, C, or D"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── Missing MCQ options ──────────────────────────────────────────────────

    def test_mcq_options_none_rejected(self):
        q = _good_mcq(options=None)
        with pytest.raises(ValueError, match="object with exactly"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    def test_mcq_options_missing_d_rejected(self):
        q = _good_mcq(options={"A": "Self-attention", "B": "Recurrent", "C": "Conv"})
        with pytest.raises(ValueError, match="object with exactly"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── question_type mismatch ───────────────────────────────────────────────

    def test_type_mismatch_rejected(self):
        q = _good_mcq(question_type="Short Answer")
        with pytest.raises(ValueError, match="MCQ"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── True/False invalid answer ────────────────────────────────────────────

    def test_true_false_invalid_answer(self):
        q = {
            "question": "Is self-attention used in Transformers?",
            "question_type": "True/False",
            "options": None,
            "correct_answer": "Yes",
            "explanation": "Self-attention is the core of Transformers.",
            "topic": "Attention",
            "difficulty": "Easy",
        }
        with pytest.raises(ValueError, match="True.*False"):
            _validate_local_question(q, "True/False", "Attention", "Easy")

    # ── Placeholder explanation ───────────────────────────────────────────────

    def test_ellipsis_explanation_rejected(self):
        q = _good_mcq(explanation="...")
        with pytest.raises(ValueError, match="placeholder"):
            _validate_local_question(q, "MCQ", "Transformers", "Medium")

    # ── Short Answer placeholder answer ─────────────────────────────────────

    def test_short_answer_placeholder_answer_rejected(self):
        q = {
            "question": "What is positional encoding used for in Transformers?",
            "question_type": "Short Answer",
            "options": None,
            "correct_answer": "...",
            "explanation": "Positional encoding adds position information.",
            "topic": "Transformers",
            "difficulty": "Easy",
        }
        with pytest.raises(ValueError, match="placeholder"):
            _validate_local_question(q, "Short Answer", "Transformers", "Easy")


# ===========================================================================
# Adapter detection
# ===========================================================================

class TestAdapterStatus:
    def test_missing_adapter(self):
        status = get_adapter_status(custom_path="non_existent_dir_xyz_12345")
        assert status["is_available"] is False
        assert "non_existent_dir_xyz_12345" in status["adapter_path"]

    def test_detected_adapter_with_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write minimal adapter_config.json
            cfg = {
                "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "peft_type": "LORA",
                "r": 16,
            }
            with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
                json.dump(cfg, f)
            # Write dummy weights
            with open(os.path.join(tmpdir, "adapter_model.safetensors"), "wb") as f:
                f.write(b"\x00" * 16)

            status = get_adapter_status(custom_path=tmpdir)
            assert status["is_available"] is True
            assert status["has_weights"] is True
            assert status["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_detected_without_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}
            with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
                json.dump(cfg, f)
            # No weights file
            status = get_adapter_status(custom_path=tmpdir)
            assert status["is_available"] is True
            assert status["has_weights"] is False


# ===========================================================================
# LLMService — local provider
# ===========================================================================

class TestLLMServiceLocalProvider:
    def test_local_provider_recognized(self):
        svc = LLMService(provider="Fine-Tuned Local Model (QLoRA)")
        assert svc.provider == "local"

    def test_local_provider_peft_keyword(self):
        svc = LLMService(provider="local")
        assert svc.provider == "local"

    def test_local_info_structure(self):
        svc = LLMService(provider="local")
        info = svc.get_active_provider_info()
        assert "Fine-Tuned Local Model" in info["provider"]
        assert "LoRA" in info["model"]

    def test_local_missing_adapter_returns_false_configured(self):
        with patch("local_model_service.resolve_adapter_path", return_value=None):
            svc = LLMService(provider="local")
            assert svc.has_active_credentials() is False


# ===========================================================================
# generate_json — local provider mocked
# ===========================================================================

class TestLLMServiceGenerateJsonLocal:
    """Verify that LLMService.generate_json for local provider routes correctly."""

    def _make_service(self):
        svc = LLMService(provider="local")
        return svc

    def test_missing_adapter_returns_error_not_exception(self):
        with patch("local_model_service.resolve_adapter_path", return_value=None):
            svc = self._make_service()
            result, err = svc.generate_json("sys", "user")
            assert result is None
            assert err is not None
            assert "LORA_ADAPTER_NOT_FOUND" in err

    def test_no_cloud_call_when_local(self):
        """Ensure no Groq/OpenAI import occurs when provider is local."""
        with (
            patch("local_model_service.resolve_adapter_path", return_value=None),
            patch("groq.Groq") as mock_groq,
            patch("openai.OpenAI") as mock_oai,
        ):
            svc = self._make_service()
            svc.generate_json("sys", "user")
            mock_groq.assert_not_called()
            mock_oai.assert_not_called()


# ===========================================================================
# Pydantic GeneratedQuestion — placeholder rejection
# ===========================================================================

class TestPydanticGeneratedQuestion:
    def test_valid_question_accepted(self):
        q = GeneratedQuestion(
            question_id="Q1",
            question="What is the primary function of the attention mechanism?",
            question_type="MCQ",
            options={"A": "Focus on relevant tokens", "B": "Pool features", "C": "Embed tokens", "D": "Normalize values"},
            correct_answer="A",
            explanation="Attention selects relevant tokens dynamically.",
            topic="Attention",
            difficulty="Medium",
        )
        assert q.question == "What is the primary function of the attention mechanism?"

    def test_ellipsis_question_rejected_by_pydantic(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="placeholder"):
            GeneratedQuestion(
                question_id="Q1",
                question="...",
                question_type="MCQ",
                options={"A": "A", "B": "B", "C": "C", "D": "D"},
                correct_answer="A",
                explanation="Explanation.",
                topic="Topic",
                difficulty="Medium",
            )

    def test_too_short_question_rejected_by_pydantic(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="too short"):
            GeneratedQuestion(
                question_id="Q1",
                question="Why?",   # 4 chars < 10
                question_type="Short Answer",
                options=None,
                correct_answer="Because.",
                explanation="Explanation.",
                topic="Topic",
                difficulty="Easy",
            )

    def test_empty_question_rejected_by_pydantic(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GeneratedQuestion(
                question_id="Q1",
                question="",
                question_type="MCQ",
                options={"A": "A", "B": "B", "C": "C", "D": "D"},
                correct_answer="A",
                explanation="Explanation.",
                topic="Topic",
                difficulty="Medium",
            )


# ===========================================================================
# generate_single_question_local — mocked model
# ===========================================================================

class TestGenerateSingleQuestionLocalMocked:
    """Tests using a mocked model to avoid GPU dependency."""

    def _make_model_tokenizer(self, generated_text: str):
        import torch
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        tokenizer.eos_token_id = 2
        # tokenizer.__call__ returns input ids with shape (1, 10)
        fake_inputs = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.return_value = fake_inputs
        tokenizer.decode.return_value = generated_text

        model = MagicMock()
        # Use a real torch.device so that tensor.to(device) succeeds
        cpu_device = torch.device("cpu")
        fake_param = MagicMock()
        fake_param.device = cpu_device
        model.parameters.return_value = iter([fake_param])
        # model.generate returns shape (1, 50) tensor
        model.generate.return_value = torch.zeros(1, 50, dtype=torch.long)

        return model, tokenizer

    def _import_and_patch(self, generated_text: str):
        from local_model_service import generate_single_question_local
        model, tokenizer = self._make_model_tokenizer(generated_text)
        return generate_single_question_local, model, tokenizer

    def test_valid_mcq_from_model(self):
        from local_model_service import generate_single_question_local
        valid_json = json.dumps({
            "question_id": "Q1",
            "question": "Which mechanism allows Transformers to attend globally?",
            "question_type": "MCQ",
            "options": {"A": "Self-attention", "B": "LSTM hidden states", "C": "Convolution", "D": "Pooling"},
            "correct_answer": "A",
            "explanation": "Self-attention computes global dependencies.",
            "topic": "Transformers",
            "difficulty": "Medium",
        })
        model, tokenizer = self._make_model_tokenizer(valid_json)
        result, err = generate_single_question_local(
            context="Transformers use self-attention for global dependency modelling.",
            topic="Transformers",
            qtype="MCQ",
            difficulty="Medium",
            bloom_level="Understand",
            model=model,
            tokenizer=tokenizer,
        )
        assert err is None
        assert result is not None
        assert result["question_type"] == "MCQ"

    def test_placeholder_question_propagates_json(self):
        """The single-question function itself does not validate content — that is _validate's job."""
        from local_model_service import generate_single_question_local
        placeholder_json = json.dumps({
            "question_id": "Q1",
            "question": "...",
            "question_type": "MCQ",
            "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
            "correct_answer": "A",
            "explanation": "...",
            "topic": "Topic",
            "difficulty": "Medium",
        })
        model, tokenizer = self._make_model_tokenizer(placeholder_json)
        result, err = generate_single_question_local(
            context="Some context.",
            topic="Topic",
            qtype="MCQ",
            difficulty="Medium",
            bloom_level="Understand",
            model=model,
            tokenizer=tokenizer,
        )
        # generate_single returns raw parsed JSON; _validate catches the placeholder
        assert result is not None
        assert result["question"] == "..."

    def test_invalid_json_returns_error(self):
        from local_model_service import generate_single_question_local
        model, tokenizer = self._make_model_tokenizer("not valid json at all")
        result, err = generate_single_question_local(
            context="Context",
            topic="Topic",
            qtype="MCQ",
            difficulty="Medium",
            bloom_level="Understand",
            model=model,
            tokenizer=tokenizer,
        )
        assert result is None
        assert err is not None
        assert "LOCAL_INVALID_JSON" in err or "LOCAL_NO_JSON" in err

    def test_empty_output_returns_error(self):
        from local_model_service import generate_single_question_local
        model, tokenizer = self._make_model_tokenizer("")
        result, err = generate_single_question_local(
            context="Context",
            topic="Topic",
            qtype="MCQ",
            difficulty="Medium",
            bloom_level="Understand",
            model=model,
            tokenizer=tokenizer,
        )
        assert result is None
        assert "EMPTY" in (err or "")


# ===========================================================================
# generate_quiz_local — integration with mocked inference
# ===========================================================================

def _make_valid_mcq_json(idx: int = 1) -> str:
    return json.dumps({
        "question_id": f"Q{idx}",
        "question": f"What does self-attention allow the model to do in example {idx}?",
        "question_type": "MCQ",
        "options": {
            "A": "Attend to all positions simultaneously",
            "B": "Process input sequentially",
            "C": "Apply fixed filters",
            "D": "Use max-pooling",
        },
        "correct_answer": "A",
        "explanation": "Self-attention enables parallel global attention.",
        "topic": "Attention",
        "difficulty": "Medium",
    })


class TestGenerateQuizLocalMocked:
    def _patch_generate(self, return_json: str):
        """Patch generate_single_question_local to return a fixed JSON string."""
        import torch
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<prompt>"
        tokenizer.eos_token_id = 2
        tokenizer.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        tokenizer.decode.return_value = return_json
        model = MagicMock()
        model.parameters.return_value = iter([MagicMock(device=MagicMock(type="cpu"))])
        model.generate.return_value = torch.zeros(1, 50, dtype=torch.long)
        return model, tokenizer

    def test_valid_single_mcq_quiz(self):
        from local_model_service import generate_quiz_local
        import torch

        mock_output = json.loads(_make_valid_mcq_json(1))

        with (
            patch("local_model_service.load_cached_peft_model") as mock_loader,
            patch("local_model_service.generate_single_question_local", return_value=(mock_output, None)),
        ):
            model_mock = MagicMock()
            tokenizer_mock = MagicMock()
            mock_loader.return_value = (model_mock, tokenizer_mock)

            quiz, err = generate_quiz_local(
                pdf_context="Attention mechanisms allow the model to focus on relevant positions in the input.",
                target_count=1,
                target_topic="Attention",
                target_diff="Medium",
                target_types=["MCQ"],
                type_counts={"MCQ": 1},
                adapter_path="/fake/path",
            )

        assert err is None, f"Unexpected error: {err}"
        assert quiz is not None
        assert len(quiz["questions"]) == 1
        assert quiz["questions"][0]["question_type"] == "MCQ"

    def test_placeholder_output_causes_retry_and_failure(self):
        """When the model always outputs a placeholder, quiz generation must fail."""
        from local_model_service import generate_quiz_local

        placeholder_output = {
            "question_id": "Q1",
            "question": "...",
            "question_type": "MCQ",
            "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
            "correct_answer": "A",
            "explanation": "...",
            "topic": "Topic",
            "difficulty": "Medium",
        }

        with (
            patch("local_model_service.load_cached_peft_model") as mock_loader,
            patch("local_model_service.generate_single_question_local", return_value=(placeholder_output, None)),
        ):
            mock_loader.return_value = (MagicMock(), MagicMock())
            quiz, err = generate_quiz_local(
                pdf_context="Some meaningful PDF content about transformers.",
                target_count=1,
                target_topic="Topic",
                target_diff="Medium",
                target_types=["MCQ"],
                type_counts={"MCQ": 1},
                adapter_path="/fake/path",
            )

        assert quiz is None
        assert err is not None
        assert "0" in err or "0 of" in err  # "returned 0 of 1"

    def test_invalid_json_causes_failure(self):
        """When the model always outputs invalid JSON, quiz generation must fail."""
        from local_model_service import generate_quiz_local

        with (
            patch("local_model_service.load_cached_peft_model") as mock_loader,
            patch("local_model_service.generate_single_question_local", return_value=(None, "LOCAL_INVALID_JSON: ...")),
        ):
            mock_loader.return_value = (MagicMock(), MagicMock())
            quiz, err = generate_quiz_local(
                pdf_context="Transformer architecture overview.",
                target_count=1,
                target_topic="Transformers",
                target_diff="Medium",
                target_types=["MCQ"],
                type_counts={"MCQ": 1},
                adapter_path="/fake/path",
            )

        assert quiz is None
        assert err is not None

    def test_target_count_exceeds_local_limit(self):
        from local_model_service import generate_quiz_local, LOCAL_MAX_QUESTIONS
        quiz, err = generate_quiz_local(
            pdf_context="Context",
            target_count=LOCAL_MAX_QUESTIONS + 1,
            target_topic="Topic",
            target_diff="Medium",
            target_types=["MCQ"],
            type_counts={},
            adapter_path="/fake/path",
        )
        assert quiz is None
        assert "LOCAL_REQUEST_TOO_LARGE" in err

    def test_empty_context_causes_failure(self):
        from local_model_service import generate_quiz_local
        with patch("local_model_service.load_cached_peft_model") as mock_loader:
            mock_loader.return_value = (MagicMock(), MagicMock())
            quiz, err = generate_quiz_local(
                pdf_context="",
                target_count=1,
                target_topic="Topic",
                target_diff="Medium",
                target_types=["MCQ"],
                type_counts={},
                adapter_path="/fake/path",
            )
        assert quiz is None
        assert "LOCAL_EMPTY_CONTEXT" in err


# ===========================================================================
# QuizGenerator — missing adapter path raises RuntimeError
# ===========================================================================

class TestQuizGeneratorLocalMissing:
    def test_missing_adapter_raises_runtime_error(self):
        dm = DatasetManager("genquiz_dataset.csv")
        svc = LLMService(provider="local")
        with patch("local_model_service.resolve_adapter_path", return_value=None):
            generator = QuizGenerator(dm, svc)
            with pytest.raises(RuntimeError) as exc_info:
                generator.generate_quiz(
                    pdf_text_or_dict="Some PDF content about Transformers.",
                    user_prompt="Generate 1 MCQ about Transformers.",
                    force_demo=False,
                )
            assert "LORA_ADAPTER_NOT_FOUND" in str(exc_info.value)
            assert "GenQuiz_FineTuning.ipynb" in str(exc_info.value)
