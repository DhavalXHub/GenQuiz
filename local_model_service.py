"""
local_model_service.py - Supervised Fine-Tuned (LoRA/QLoRA) Local Model Engine for GenQuiz.

Provides:
- Detection of local fine-tuned LoRA adapter weights (fine_tuned_adapter/ or models/genquiz_lora_adapter/)
- Cached loading of base model (Qwen/Qwen2.5-1.5B-Instruct) + PEFT LoRA adapter
- Memory-efficient CPU and GPU (CUDA) inference support
- Strict separation: never routes to external cloud APIs when local mode is selected
- Robust JSON extraction and grounding on uploaded PDF passages
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

# Global fallback cache for CLI and unit tests
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


def resolve_adapter_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds an existing adapter directory with adapter_config.json."""
    if custom_path and os.path.isdir(custom_path):
        if os.path.exists(os.path.join(custom_path, "adapter_config.json")):
            return custom_path

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

        # Attempt to read base model name from adapter_config.json
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
            "message": f"Fine-tuned LoRA adapter detected at '{resolved}' (Base: {base_model_name})"
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
            )
        }


def _load_model_and_tokenizer(base_model_name: str, adapter_path: str):
    """Internal loader that attaches the LoRA adapter to the base causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")) else base_model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if device == "cpu":
        model.to("cpu")

    return model, tokenizer


def load_cached_peft_model(
    base_model_name: Optional[str] = None,
    adapter_path: Optional[str] = None
):
    """
    Loads and caches the fine-tuned model and tokenizer in memory.
    Leverages st.cache_resource if running under Streamlit, otherwise uses
    module-level dictionary cache.
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
    cache_key = f"{effective_base}::{resolved_path}"

    # Check if Streamlit is active
    try:
        import streamlit as st
        # If Streamlit is running with session, use st.cache_resource
        if hasattr(st, "cache_resource"):
            @st.cache_resource(show_spinner="Loading Fine-Tuned Local LLM & LoRA Adapter...")
            def _streamlit_cached_loader(b_name: str, a_path: str):
                return _load_model_and_tokenizer(b_name, a_path)
            return _streamlit_cached_loader(effective_base, resolved_path)
    except Exception:
        pass

    # Module-level cache fallback for tests/CLI
    if cache_key not in _GLOBAL_MODEL_CACHE:
        _GLOBAL_MODEL_CACHE[cache_key] = _load_model_and_tokenizer(effective_base, resolved_path)

    return _GLOBAL_MODEL_CACHE[cache_key]


def clean_local_json_text(text: str) -> str:
    """Strips markdown formatting and isolates valid JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # If surrounded by extraneous text, extract between first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    return cleaned.strip()


def generate_single_question_local(
    context: str,
    topic: str,
    qtype: str,
    difficulty: str,
    bloom_level: str,
    model,
    tokenizer,
    temperature: float = 0.2,
    max_new_tokens: int = 400
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generates a single structured question strictly grounded in the context
    using the fine-tuned ChatML prompt structure.
    """
    import torch

    system_prompt = (
        "You are an expert assessment generator. Based strictly on the provided context passage "
        "and instructions, generate a pedagogically aligned evaluation question with its answer, "
        "explanation, and metadata. Output valid JSON only conforming to the required schema."
    )

    user_prompt = (
        f"SOURCE CONTEXT:\n{context.strip()}\n\n"
        f"SPECIFICATIONS:\n"
        f"- Topic: {topic.strip()}\n"
        f"- Question Type: {qtype.strip()}\n"
        f"- Difficulty Level: {difficulty.strip()}\n"
        f"- Bloom's Taxonomy Level: {bloom_level.strip()}\n\n"
        f"Generate one {difficulty.strip()}-level {qtype.strip()} question strictly grounded "
        f"in the source context above. Include the correct answer, educational explanation, "
        f"and options (if MCQ). Return valid JSON only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.01, temperature),
                do_sample=temperature > 0.05,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        cleaned_json = clean_local_json_text(raw_text)
        parsed = json.loads(cleaned_json)
        return parsed, None

    except json.JSONDecodeError as jde:
        return None, f"Local model produced invalid JSON: {str(jde)}"
    except Exception as e:
        return None, f"Local model inference error: {str(e)}"


def generate_quiz_local(
    pdf_context: str,
    target_count: int,
    target_topic: str,
    target_diff: str,
    target_types: List[str],
    type_counts: Dict[str, int],
    adapter_path: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generates a full quiz by executing grounded SFT inference across passages.
    Builds the exact number and distribution of question types requested.
    """
    try:
        model, tokenizer = load_cached_peft_model(adapter_path=adapter_path)
    except Exception as e:
        return None, str(e)

    # Determine distribution of question types
    items_to_generate: List[str] = []
    if type_counts:
        for qt, cnt in type_counts.items():
            for _ in range(cnt):
                items_to_generate.append(qt)

    # Fill remainder up to target_count
    type_idx = 0
    while len(items_to_generate) < target_count:
        items_to_generate.append(target_types[type_idx % len(target_types)])
        type_idx += 1

    items_to_generate = items_to_generate[:target_count]

    # Split PDF context into passage chunks to ground different questions on distinct parts
    paragraphs = [p.strip() for p in pdf_context.split("\n\n") if len(p.strip()) > 80]
    if not paragraphs:
        paragraphs = [pdf_context.strip()]

    generated_questions: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, qtype in enumerate(items_to_generate, 1):
        # Pick context passage
        passage = paragraphs[(idx - 1) % len(paragraphs)]
        bloom = "Understand" if target_diff == "Easy" else ("Apply" if target_diff == "Medium" else "Analyze")

        q_json, err = generate_single_question_local(
            context=passage,
            topic=target_topic,
            qtype=qtype,
            difficulty=target_diff,
            bloom_level=bloom,
            model=model,
            tokenizer=tokenizer
        )

        if q_json and isinstance(q_json, dict):
            # Ensure question_id
            q_json["question_id"] = f"Q{idx}"
            generated_questions.append(q_json)
        else:
            errors.append(f"Q{idx} ({qtype}): {err or 'Unknown generation error'}")

    if not generated_questions:
        return None, f"Local model failed to generate questions. Errors: {'; '.join(errors)}"

    quiz_dict = {
        "title": f"{target_topic} Assessment",
        "topic": target_topic,
        "difficulty": target_diff,
        "questions": generated_questions
    }

    return quiz_dict, None
