"""
local_model_service.py - Supervised Fine-Tuned (LoRA/QLoRA) Local Model Engine for GenQuiz.

Provides:
- Detection of local fine-tuned LoRA adapter weights (fine_tuned_adapter/ or models/genquiz_lora_adapter/)
- Cached loading of base model (Qwen/Qwen2.5-1.5B-Instruct) + PEFT LoRA adapter
- Memory-efficient CPU and GPU (CUDA) inference support
- Strict separation: never routes to external cloud APIs when local mode is selected
- Robust JSON extraction and grounding on uploaded PDF passages
- Placeholder detection: "...", "First"/"Second"/"Third"/"Fourth", and other schema stubs are rejected
- Clear, actionable status and missing-adapter diagnostics
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, Tuple

# Candidate paths for the exported LoRA adapter
CANDIDATE_ADAPTER_PATHS = [
    "fine_tuned_adapter",
    os.path.join("models", "genquiz_lora_adapter"),
    "genquiz_lora_adapter",
]

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MAX_INPUT_TOKENS = 2048
LOCAL_MAX_NEW_TOKENS = 512
LOCAL_MAX_QUESTIONS = 8
LOCAL_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Placeholder / stub detection constants
# ---------------------------------------------------------------------------
# These values appear in the old prompt schema example and in the training data.
# The fine-tuned model reproduces them verbatim — we must reject them.
_PLACEHOLDER_QUESTION_VALUES: frozenset = frozenset(
    ["...", "…", "<question>", "question text here", "enter question here",
     "your question here", "sample question", "placeholder", "n/a", "none", ""]
)
# Generic MCQ option labels that were used as schema stubs in the old prompt
_GENERIC_MCQ_OPTIONS: frozenset = frozenset(
    ["first", "second", "third", "fourth",
     "option a", "option b", "option c", "option d",
     "choice a", "choice b", "choice c", "choice d",
     "<option>", "...", "…", ""]
)
# Minimum meaningful lengths to reject stub-length content
_MIN_QUESTION_LEN = 10   # "What is X?" is 10 chars
_MIN_OPTION_LEN   = 3    # single letter is already too short
_MIN_ANSWER_LEN   = 1

# Global fallback cache for CLI and unit tests
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Device / CUDA helpers
# ---------------------------------------------------------------------------

def _is_cuda_oom(error: BaseException) -> bool:
    """Return whether an exception indicates CUDA memory exhaustion."""
    return "cuda" in str(error).lower() and "out of memory" in str(error).lower()


def _clear_cuda_cache() -> None:
    """Release cached CUDA allocations when torch and CUDA are available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _model_input_device(model):
    """Find the device on which model inputs should be placed."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


# ---------------------------------------------------------------------------
# Prompt construction — NO LITERAL PLACEHOLDERS
# ---------------------------------------------------------------------------

def _question_prompt(
    context: str,
    topic: str,
    qtype: str,
    difficulty: str,
    bloom_level: str,
    correction: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Build the system and user prompts for one grounded question.

    IMPORTANT: The JSON field schema description must NOT contain literal "..."
    or generic dummy values such as "First"/"Second"/"Third"/"Fourth" because
    the fine-tuned Qwen model was trained on those stub values and will
    reproduce them verbatim when it sees them in the prompt.
    """
    system_prompt = (
        "You are an expert educational assessment author. "
        "Using ONLY information from the SOURCE CONTEXT provided, write one pedagogically "
        "sound quiz question in the requested format. "
        "Return ONLY a raw JSON object — no markdown fences, no explanatory text, no preamble. "
        "Every field must contain real, specific content from the source context."
    )

    if qtype == "MCQ":
        options_instruction = (
            "For an MCQ, the \"options\" field must be a JSON object with exactly four keys "
            "\"A\", \"B\", \"C\", and \"D\". Each value must be a distinct, specific answer "
            "choice derived from the source context — not generic labels like 'First', 'Second', "
            "'Option A', or placeholder text. "
            "The \"correct_answer\" must be exactly one of the letters A, B, C, or D."
        )
        options_schema_note = (
            '"options": {"A": "<specific choice text>", "B": "<specific choice text>", '
            '"C": "<specific choice text>", "D": "<specific choice text>"},\n'
            '  "correct_answer": "A"'
        )
    elif qtype == "True/False":
        options_instruction = (
            "For a True/False question, the \"options\" field must be null. "
            "The \"correct_answer\" must be exactly the word \"True\" or the word \"False\"."
        )
        options_schema_note = '"options": null,\n  "correct_answer": "True"'
    else:
        # Short Answer or Fill in the Blank
        options_instruction = (
            "For a Short Answer question, the \"options\" field must be null. "
            "The \"correct_answer\" must be a concise, specific, factually grounded answer "
            "drawn directly from the source context."
        )
        options_schema_note = '"options": null,\n  "correct_answer": "<specific answer phrase>"'

    schema_example = (
        "Required JSON schema (all fields mandatory, no placeholders):\n"
        "{\n"
        '  "question_id": "Q1",\n'
        '  "question": "<a real, specific question about the source context>",\n'
        f'  "question_type": "{qtype}",\n'
        f'  {options_schema_note},\n'
        '  "explanation": "<why the answer is correct, citing evidence from the context>",\n'
        f'  "topic": "{topic}",\n'
        f'  "difficulty": "{difficulty}"\n'
        "}"
    )

    user_prompt = (
        f"SOURCE CONTEXT:\n{context.strip()}\n\n"
        f"TASK:\n"
        f"Write one {difficulty}-level {qtype} question strictly grounded in the source context above.\n"
        f"Bloom's Taxonomy level: {bloom_level}\n\n"
        f"{options_instruction}\n\n"
        f"{schema_example}"
    )

    if correction:
        user_prompt += (
            f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {correction}\n"
            "Please correct the issue and return only valid JSON."
        )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Placeholder / stub detection
# ---------------------------------------------------------------------------

def _is_placeholder(text: str) -> bool:
    """Return True if the text is a generic stub / placeholder value."""
    if text is None:
        return True
    t = str(text).strip().lower()
    return t in _PLACEHOLDER_QUESTION_VALUES


def _is_generic_option(text: str) -> bool:
    """Return True if an MCQ option text is a generic placeholder."""
    if text is None:
        return True
    t = str(text).strip().lower()
    return t in _GENERIC_MCQ_OPTIONS or len(t) < _MIN_OPTION_LEN


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Known cross-domain collision dictionaries to catch hallucinated answer keys
_CROSS_DOMAIN_PAIRS = [
    (
        {"cybercrime", "cyberterrorism", "cybersecurity", "malware", "phishing", "ransomware", "unauthorized access", "computer crime", "hacking", "firewall"},
        {"positional encoding", "transformer", "self-attention", "multi-head attention", "encoder-decoder", "token embedding"},
        "Cybersecurity",
        "Transformers / NLP"
    ),
    (
        {"sorting", "quicksort", "mergesort", "binary search", "linked list", "binary tree"},
        {"positional encoding", "self-attention", "gradient descent", "backpropagation", "convolution"},
        "Data Structures / Algorithms",
        "Deep Learning"
    ),
]


def _validate_local_question(
    question: Dict[str, Any],
    expected_type: str,
    expected_topic: str,
    expected_difficulty: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate and normalize one local model question.

    Raises ValueError with a descriptive message when:
    - Required fields are missing or empty.
    - Question text is a placeholder (e.g., "...").
    - MCQ options are generic stubs (e.g., "First", "Second", "Third", "Fourth").
    - Correct_answer is not consistent with the question type.
    - question_type does not match expected_type.
    - Question and answer exhibit an obvious cross-domain mismatch (e.g. cybercrime vs positional encoding).
    """
    if not isinstance(question, dict):
        raise ValueError("Question output must be a JSON object.")

    # ── Field presence ──────────────────────────────────────────────────────
    required = ["question", "question_type", "correct_answer", "explanation"]
    missing = [f for f in required if not str(question.get(f, "")).strip()]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    # ── question_type ────────────────────────────────────────────────────────
    qtype = str(question["question_type"]).strip()
    if qtype != expected_type:
        raise ValueError(f"question_type must be '{expected_type}', got '{qtype}'")

    # ── question text quality ────────────────────────────────────────────────
    q_text = str(question["question"]).strip()
    if _is_placeholder(q_text):
        raise ValueError(
            f"question text is a placeholder ('{q_text}'). "
            "A real question grounded in the source context is required."
        )
    if len(q_text) < _MIN_QUESTION_LEN:
        raise ValueError(
            f"question text is too short ({len(q_text)} chars): '{q_text}'. "
            f"Minimum {_MIN_QUESTION_LEN} characters required."
        )

    # ── explanation quality ──────────────────────────────────────────────────
    explanation = str(question["explanation"]).strip()
    if _is_placeholder(explanation):
        raise ValueError(
            "explanation is a placeholder. A real educational explanation is required."
        )

    # ── MCQ-specific validation ──────────────────────────────────────────────
    options = question.get("options")
    if expected_type == "MCQ":
        if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(
                "MCQ options must be a JSON object with exactly the keys A, B, C, and D."
            )
        # Check every option for placeholder / generic values
        for letter in ("A", "B", "C", "D"):
            opt_text = str(options.get(letter, "")).strip()
            if _is_generic_option(opt_text):
                raise ValueError(
                    f"MCQ option {letter} is a generic placeholder ('{opt_text}'). "
                    "All four options must be distinct, specific answer choices from the source context."
                )
        # Require distinct options (case-insensitive)
        texts = [str(options[k]).strip().lower() for k in ("A", "B", "C", "D")]
        if len(set(texts)) < 4:
            raise ValueError("MCQ options must be distinct — duplicate options found.")
        # Validate correct_answer
        answer = str(question["correct_answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"MCQ correct_answer must be one of A, B, C, or D; got '{question['correct_answer']}'."
            )
        question["correct_answer"] = answer
    else:
        # Non-MCQ must not have options
        if options is not None:
            raise ValueError(f"{expected_type} options must be null.")

    # ── True/False validation ────────────────────────────────────────────────
    if expected_type == "True/False":
        answer = str(question["correct_answer"]).strip().lower()
        if answer not in {"true", "false"}:
            raise ValueError(
                f"True/False correct_answer must be 'True' or 'False'; got '{question['correct_answer']}'."
            )
        question["correct_answer"] = answer.capitalize()

    # ── Short Answer validation ──────────────────────────────────────────────
    if expected_type == "Short Answer":
        sa_ans = str(question["correct_answer"]).strip()
        if _is_placeholder(sa_ans) or len(sa_ans) < _MIN_ANSWER_LEN:
            raise ValueError(
                f"Short Answer correct_answer is empty or a placeholder ('{sa_ans}'). "
                "A specific grounded answer is required."
            )

        q_lower = q_text.lower()
        a_lower = sa_ans.lower()

        # Check for cross-domain collision (e.g. cybercrime vs positional encoding)
        for dom_a_keys, dom_b_keys, dom_a_name, dom_b_name in _CROSS_DOMAIN_PAIRS:
            q_in_a = any(k in q_lower for k in dom_a_keys)
            q_in_b = any(k in q_lower for k in dom_b_keys)
            a_in_a = any(k in a_lower for k in dom_a_keys)
            a_in_b = any(k in a_lower for k in dom_b_keys)

            if q_in_a and not q_in_b and a_in_b and not a_in_a:
                raise ValueError(
                    f"Question-answer domain mismatch detected: Question discusses {dom_a_name} "
                    f"('{q_text[:60]}...'), but expected answer discusses {dom_b_name} ('{sa_ans[:60]}...')."
                )
            if q_in_b and not q_in_a and a_in_a and not a_in_b:
                raise ValueError(
                    f"Question-answer domain mismatch detected: Question discusses {dom_b_name} "
                    f"('{q_text[:60]}...'), but expected answer discusses {dom_a_name} ('{sa_ans[:60]}...')."
                )

        if context:
            ctx_lower = str(context).lower()
            stop = {"that", "this", "with", "from", "have", "were", "what", "which", "will", "when", "does", "been", "into", "more"}
            a_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", a_lower) if w not in stop]
            if len(a_words) >= 2:
                has_overlap = any(w in q_lower or w in ctx_lower for w in a_words)
                if not has_overlap:
                    raise ValueError(
                        f"Ungrounded answer: Expected answer '{sa_ans}' has no conceptual overlap "
                        f"with either the question ('{q_text[:50]}...') or the source context."
                    )

    # ── Normalise fields ─────────────────────────────────────────────────────
    question["question_id"] = str(question.get("question_id") or "").strip()
    question["question"]    = q_text
    question["question_type"] = qtype
    question["explanation"] = explanation
    question["topic"]       = str(question.get("topic") or expected_topic).strip()
    question["difficulty"]  = str(question.get("difficulty") or expected_difficulty).strip()
    return question


# ---------------------------------------------------------------------------
# Token decoding
# ---------------------------------------------------------------------------

def _decode_generated_text(
    outputs,
    input_token_count: int,
    tokenizer,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Decode only newly generated tokens and report truncation metadata."""
    sequence = outputs[0]
    generated_token_count = int(sequence.shape[-1]) - input_token_count
    text = tokenizer.decode(
        sequence[input_token_count:], skip_special_tokens=True
    ).strip()
    return text, {
        "input_tokens": input_token_count,
        "generated_tokens": generated_token_count,
        "hit_token_limit": generated_token_count >= max_new_tokens,
        "empty": not bool(text),
    }


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------

def resolve_adapter_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds an existing adapter directory with adapter_config.json."""
    if custom_path:
        if os.path.isdir(custom_path) and os.path.exists(
            os.path.join(custom_path, "adapter_config.json")
        ):
            return custom_path
        return None

    env_path = os.getenv("LORA_ADAPTER_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        if os.path.exists(os.path.join(env_path, "adapter_config.json")):
            return env_path

    for path in CANDIDATE_ADAPTER_PATHS:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            return path

    return None


def get_adapter_status(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns inspection details about local LoRA adapter availability without
    attempting expensive model loads.
    """
    resolved = resolve_adapter_path(custom_path)
    if resolved:
        has_safetensors = os.path.exists(os.path.join(resolved, "adapter_model.safetensors"))
        has_bin = os.path.exists(os.path.join(resolved, "adapter_model.bin"))
        has_weights = has_safetensors or has_bin

        base_model_name = DEFAULT_BASE_MODEL
        try:
            cfg_path = os.path.join(resolved, "adapter_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                base_model_name = cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
        except Exception:
            pass

        return {
            "is_available": True,
            "has_weights": has_weights,
            "adapter_path": resolved,
            "base_model": base_model_name,
            "message": (
                f"Fine-tuned LoRA adapter detected at '{resolved}' "
                f"(Base: {base_model_name})"
            ),
        }
    else:
        target_path = custom_path or "fine_tuned_adapter"
        return {
            "is_available": False,
            "has_weights": False,
            "adapter_path": target_path,
            "base_model": DEFAULT_BASE_MODEL,
            "message": (
                f"LoRA adapter weights not found at '{target_path}'.\n"
                "Please run 'GenQuiz_FineTuning.ipynb' in Google Colab (free T4 GPU), "
                f"download the zipped adapter, and extract it to '{target_path}/'."
            ),
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(base_model_name: str, adapter_path: str):
    """Internal loader: attaches the LoRA adapter to the base causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float16 on GPU, float32 on CPU; prefer bfloat16 when supported
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_source = (
        adapter_path
        if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json"))
        else base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model ───────────────────────────────────────────────────────────
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # ── LoRA adapter ─────────────────────────────────────────────────────────
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if device == "cpu":
        model.to("cpu")

    return model, tokenizer


def load_cached_peft_model(
    base_model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """
    Loads and caches the fine-tuned model and tokenizer in memory.
    Uses st.cache_resource under Streamlit; module-level dict otherwise.
    """
    resolved_path = resolve_adapter_path(adapter_path)
    if not resolved_path:
        target = adapter_path or "fine_tuned_adapter"
        raise FileNotFoundError(
            f"LORA_ADAPTER_NOT_FOUND: The fine-tuned LoRA adapter was not found at '{target}'.\n"
            "Please train the model using 'GenQuiz_FineTuning.ipynb' in Google Colab, download "
            f"the zip archive, and extract it to '{target}/'."
        )

    status = get_adapter_status(resolved_path)
    effective_base = base_model_name or status["base_model"]
    if base_model_name and status["base_model"] != base_model_name:
        raise ValueError(
            "LORA_BASE_MODEL_MISMATCH: Adapter expects "
            f"'{status['base_model']}', but '{base_model_name}' was requested."
        )
    cache_key = f"{effective_base}::{resolved_path}"

    # ── Streamlit cache ───────────────────────────────────────────────────────
    try:
        import streamlit as st
        if hasattr(st, "cache_resource"):
            @st.cache_resource(show_spinner="Loading Fine-Tuned Local LLM & LoRA Adapter…")
            def _streamlit_cached_loader(b_name: str, a_path: str):
                return _load_model_and_tokenizer(b_name, a_path)
            return _streamlit_cached_loader(effective_base, resolved_path)
    except Exception:
        pass

    # ── Module-level cache (tests / CLI) ──────────────────────────────────────
    if cache_key not in _GLOBAL_MODEL_CACHE:
        _GLOBAL_MODEL_CACHE[cache_key] = _load_model_and_tokenizer(
            effective_base, resolved_path
        )
    return _GLOBAL_MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def clean_local_json_text(text: str) -> str:
    """Strip markdown code fences and isolate the first complete JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    first_brace = cleaned.find("{")
    last_brace  = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Single-question generation
# ---------------------------------------------------------------------------

def generate_single_question_local(
    context: str,
    topic: str,
    qtype: str,
    difficulty: str,
    bloom_level: str,
    model,
    tokenizer,
    temperature: float = 0.4,
    max_new_tokens: int = LOCAL_MAX_NEW_TOKENS,
    correction: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate one structured, grounded question using the fine-tuned model.

    NOTE: temperature is set to 0.4 by default (was 0.2) because the
    fine-tuned model needs more diversity to produce real question text
    rather than echoing training stubs.
    """
    import torch

    system_prompt, user_prompt = _question_prompt(
        context, topic, qtype, difficulty, bloom_level, correction=correction
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=LOCAL_MAX_INPUT_TOKENS,
        )
        device = _model_input_device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        effective_max_new = min(max_new_tokens, LOCAL_MAX_NEW_TOKENS)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=effective_max_new,
                temperature=max(0.05, temperature),
                do_sample=temperature > 0.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw_text, meta = _decode_generated_text(
            outputs,
            inputs["input_ids"].shape[1],
            tokenizer,
            effective_max_new,
        )

        if meta["empty"]:
            return None, "LOCAL_EMPTY_OUTPUT: Model produced an empty response."

        cleaned_json = clean_local_json_text(raw_text)
        if not cleaned_json:
            return None, f"LOCAL_NO_JSON: No JSON object found in model output. Raw: {raw_text[:200]!r}"

        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as jde:
            return None, f"LOCAL_INVALID_JSON: {jde}. Raw: {cleaned_json[:200]!r}"

        return parsed, None

    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            _clear_cuda_cache()
            return None, (
                "LOCAL_CUDA_OOM: GPU memory was exhausted. "
                "Try requesting fewer questions or switching to Demo Mode."
            )
        return None, f"LOCAL_RUNTIME_ERROR: {exc}"
    except Exception as exc:
        return None, f"LOCAL_INFERENCE_ERROR: {exc}"


# ---------------------------------------------------------------------------
# Deduplication & Batch Orchestration
# ---------------------------------------------------------------------------

def normalize_question_for_dedup(text: str) -> str:
    """Strip punctuation, lowercase, and remove common interrogative leading phrases."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    cleaned = re.sub(
        r"^(what\s+is|what\s+are|which\s+of\s+the\s+following|explain\s+how|describe\s+how|how\s+does|why\s+does)\s+",
        "",
        cleaned.strip()
    )
    return " ".join(cleaned.split())


def are_questions_duplicate(q1: str, q2: str, threshold: float = 0.8) -> bool:
    """Check if two questions are duplicates via exact normalized text or high token Jaccard similarity."""
    n1 = normalize_question_for_dedup(q1)
    n2 = normalize_question_for_dedup(q2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    words1 = set(n1.split())
    words2 = set(n2.split())
    if not words1 or not words2:
        return False
    jaccard = len(words1 & words2) / len(words1 | words2)
    return jaccard >= threshold


def split_request_into_batches(
    target_count: int,
    type_counts: Dict[str, int],
    target_types: List[str],
    max_batch_size: int = LOCAL_MAX_QUESTIONS,
) -> List[Dict[str, Any]]:
    """
    Splits a larger quiz request into safe batches, each with count <= max_batch_size.
    Preserves exact requested question-type distribution across the batches.
    """
    if target_count < 1:
        return []

    # 1. Expand items to generate based on requested type distribution
    items_to_generate: List[str] = []
    if type_counts:
        pools = {qt: cnt for qt, cnt in type_counts.items() if cnt > 0}
        while len(items_to_generate) < target_count and any(cnt > 0 for cnt in pools.values()):
            for qt in list(pools.keys()):
                if pools[qt] > 0 and len(items_to_generate) < target_count:
                    items_to_generate.append(qt)
                    pools[qt] -= 1

    type_idx = 0
    fallback_types = target_types or ["MCQ"]
    while len(items_to_generate) < target_count:
        items_to_generate.append(fallback_types[type_idx % len(fallback_types)])
        type_idx += 1
    items_to_generate = items_to_generate[:target_count]

    # 2. Slice into chunks of max_batch_size
    batches: List[Dict[str, Any]] = []
    for start_idx in range(0, len(items_to_generate), max_batch_size):
        chunk = items_to_generate[start_idx : start_idx + max_batch_size]
        chunk_type_counts: Dict[str, int] = {}
        for qt in chunk:
            chunk_type_counts[qt] = chunk_type_counts.get(qt, 0) + 1
        batches.append({
            "count": len(chunk),
            "type_counts": chunk_type_counts,
            "target_types": list(chunk_type_counts.keys()),
            "start_idx": start_idx + 1,
        })

    return batches


# ---------------------------------------------------------------------------
# Full quiz generation
# ---------------------------------------------------------------------------

def generate_quiz_local(
    pdf_context: str,
    target_count: int,
    target_topic: str,
    target_diff: str,
    target_types: List[str],
    type_counts: Dict[str, int],
    adapter_path: Optional[str] = None,
    existing_questions: Optional[List[Dict[str, Any]]] = None,
    start_idx: int = 1,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate a complete quiz batch via grounded local SFT inference.
    Enforces the single-batch safety limit (target_count <= LOCAL_MAX_QUESTIONS).

    Returns (quiz_dict, None) on success, or (None, error_message) on failure.
    """
    if target_count < 1:
        return None, "LOCAL_INVALID_REQUEST: target_count must be at least 1."
    if target_count > LOCAL_MAX_QUESTIONS:
        return None, (
            f"LOCAL_REQUEST_TOO_LARGE: Requested {target_count} questions, "
            f"but the local model limit is {LOCAL_MAX_QUESTIONS}."
        )
    if not target_types:
        return None, "LOCAL_INVALID_REQUEST: At least one question type is required."

    try:
        model, tokenizer = load_cached_peft_model(adapter_path=adapter_path)
    except Exception as exc:
        return None, str(exc)

    # ── Build type distribution ───────────────────────────────────────────────
    items_to_generate: List[str] = []
    if type_counts:
        for qt, cnt in type_counts.items():
            items_to_generate.extend([qt] * cnt)

    type_idx = 0
    while len(items_to_generate) < target_count:
        items_to_generate.append(target_types[type_idx % len(target_types)])
        type_idx += 1
    items_to_generate = items_to_generate[:target_count]

    # ── Split PDF context into passages ───────────────────────────────────────
    paragraphs = [p.strip() for p in pdf_context.split("\n\n") if len(p.strip()) > 60]
    if not paragraphs:
        # Fall back to the whole context as one block
        paragraphs = [pdf_context.strip()] if pdf_context.strip() else [""]

    if not paragraphs or not paragraphs[0]:
        return None, (
            "LOCAL_EMPTY_CONTEXT: The extracted PDF text is empty or too short. "
            "Please upload a text-based PDF with enough content."
        )

    # ── Bloom level mapping ───────────────────────────────────────────────────
    bloom_map = {"Easy": "Remember", "Medium": "Apply", "Hard": "Analyze"}
    bloom = bloom_map.get(target_diff, "Understand")

    # ── Deduplication tracking ────────────────────────────────────────────────
    existing_texts: List[str] = []
    if existing_questions:
        for eq in existing_questions:
            q_txt = str(eq.get("question", "")).strip()
            if q_txt:
                existing_texts.append(q_txt)

    # ── Per-question generation with retries ──────────────────────────────────
    generated_questions: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, qtype in enumerate(items_to_generate, 1):
        q_num = start_idx + len(generated_questions)
        passage = paragraphs[(idx - 1) % len(paragraphs)]

        q_json: Optional[Dict[str, Any]] = None
        last_err: Optional[str] = None
        correction_hint: Optional[str] = None

        for attempt in range(LOCAL_MAX_RETRIES + 1):
            raw_q, raw_err = generate_single_question_local(
                context=passage,
                topic=target_topic,
                qtype=qtype,
                difficulty=target_diff,
                bloom_level=bloom,
                model=model,
                tokenizer=tokenizer,
                correction=correction_hint,
            )

            if raw_q is not None:
                try:
                    validated = _validate_local_question(
                        raw_q, qtype, target_topic, target_diff, context=passage
                    )
                    cand_text = validated["question"].strip()
                    # Check deduplication against previous batches and current batch
                    if any(are_questions_duplicate(cand_text, ex) for ex in existing_texts):
                        last_err = "Duplicate question generated"
                        correction_hint = (
                            f"Your previous question '{cand_text[:50]}...' was already used. "
                            "Generate a completely different question on another aspect of the passage."
                        )
                        continue

                    q_json = validated
                    break
                except ValueError as ve:
                    last_err = str(ve)
                    correction_hint = (
                        f"Your previous answer was rejected: {last_err}. "
                        "Generate a NEW, SPECIFIC question with real content from the context."
                    )
            else:
                last_err = raw_err
                correction_hint = raw_err

        if q_json is not None:
            q_json["question_id"] = f"Q{q_num}"
            existing_texts.append(q_json["question"].strip())
            generated_questions.append(q_json)
        else:
            errors.append(f"Q{q_num} ({qtype}): {last_err or 'Unknown generation error'}")

    # ── Result assembly ───────────────────────────────────────────────────────
    if not generated_questions:
        return None, (
            f"Local model returned 0 of {target_count} requested questions. "
            f"Errors: {'; '.join(errors)}"
        )

    if len(generated_questions) < target_count:
        # Return partial results with a descriptive error rather than none
        return None, (
            f"Local model returned only {len(generated_questions)} of "
            f"{target_count} requested questions. "
            f"Errors: {'; '.join(errors)}"
        )

    quiz_dict = {
        "title": f"{target_topic} Assessment",
        "topic": target_topic,
        "difficulty": target_diff,
        "questions": generated_questions,
    }
    return quiz_dict, None


def generate_quiz_local_batched(
    pdf_context: str,
    target_count: int,
    target_topic: str,
    target_diff: str,
    target_types: List[str],
    type_counts: Dict[str, int],
    adapter_path: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    max_batch_size: int = LOCAL_MAX_QUESTIONS,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Orchestrates multi-batch quiz generation using the local fine-tuned model.
    Enforces per-batch safety limit (max_batch_size <= 8) while allowing users
    to request larger quizzes (e.g. 15 or 20 questions) without manual splitting.
    """
    if target_count < 1:
        return None, "LOCAL_INVALID_REQUEST: target_count must be at least 1."
    if not target_types:
        return None, "LOCAL_INVALID_REQUEST: At least one question type is required."

    batches = split_request_into_batches(
        target_count=target_count,
        type_counts=type_counts,
        target_types=target_types,
        max_batch_size=max_batch_size,
    )

    if not batches:
        return None, "LOCAL_INVALID_REQUEST: No batches could be planned."

    all_questions: List[Dict[str, Any]] = []
    batch_errors: List[str] = []
    total_batches = len(batches)

    for b_idx, batch_plan in enumerate(batches, 1):
        b_count = batch_plan["count"]
        b_types = batch_plan["target_types"]
        b_type_counts = batch_plan["type_counts"]

        if progress_callback:
            try:
                progress_callback(
                    b_idx,
                    total_batches,
                    f"Generating batch {b_idx} of {total_batches} ({b_count} questions)..."
                )
            except Exception:
                pass

        # Execute single batch through generate_quiz_local with deduplication context
        batch_quiz, batch_err = generate_quiz_local(
            pdf_context=pdf_context,
            target_count=b_count,
            target_topic=target_topic,
            target_diff=target_diff,
            target_types=b_types,
            type_counts=b_type_counts,
            adapter_path=adapter_path,
            existing_questions=all_questions,
            start_idx=len(all_questions) + 1,
        )

        if batch_quiz and batch_quiz.get("questions"):
            all_questions.extend(batch_quiz["questions"])
        elif batch_err:
            batch_errors.append(f"Batch {b_idx} of {total_batches} failed: {batch_err}")

    # Renumber all accumulated questions sequentially: Q1, Q2, ..., QN
    for i, q in enumerate(all_questions, 1):
        q["question_id"] = f"Q{i}"

    if not all_questions:
        return None, f"Local model quiz generation failed for all batches: {'; '.join(batch_errors)}"

    quiz_dict = {
        "title": f"{target_topic} Assessment",
        "topic": target_topic,
        "difficulty": target_diff,
        "questions": all_questions,
    }

    err_summary = None
    if len(all_questions) < target_count:
        err_summary = (
            f"Partial generation: Produced {len(all_questions)} of {target_count} requested questions. "
            f"Errors: {'; '.join(batch_errors)}"
        )

    return quiz_dict, err_summary
