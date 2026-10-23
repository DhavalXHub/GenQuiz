"""
llm_service.py - Provider-Agnostic LLM Interface for GenQuiz.

Supports:
- Groq (Default fast provider: llama-3.3-70b-versatile / llama-3.1-8b-instant)
- OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- Fine-Tuned Local Model (Open-source LoRA/QLoRA via Hugging Face PEFT)
- Demo Mode (Prewritten Verified Dataset)
- Clear key and adapter detection with masked representation (never logs or reveals secret keys)
- Robust JSON response extraction and single-retry error recovery
- Distinct error reporting when credentials or local weights are missing
"""

import os
import json
import re
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load local .env if present
load_dotenv(override=True)


def mask_secret(secret: str) -> str:
    """Safely mask API key for UI verification without exposing sensitive tokens."""
    if not secret:
        return "Not Set"
    secret = secret.strip()
    if len(secret) <= 8:
        return "••••••••"
    prefix = secret[:4]
    suffix = secret[-3:]
    return f"{prefix}••••••••{suffix}"


class LLMService:
    """Provider-agnostic LLM interface for structured quiz generation."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        # Reload dotenv in case .env was recently modified
        load_dotenv(override=True)

        requested_provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
        if any(kw in requested_provider for kw in ["local", "fine_tuned", "lora", "peft", "qlora"]):
            self.provider = "local"
        elif "openai" in requested_provider:
            self.provider = "openai"
        elif "demo" in requested_provider:
            self.provider = "demo"
        else:
            self.provider = "groq"

        # Key detection
        env_groq = os.getenv("GROQ_API_KEY", "").strip()
        env_openai = os.getenv("OPENAI_API_KEY", "").strip()

        self.groq_key = ""
        self.openai_key = ""
        self.key_source = "None"

        if api_key and api_key.strip():
            if self.provider == "openai":
                self.openai_key = api_key.strip()
                self.key_source = "Session UI Input"
            elif self.provider == "groq":
                self.groq_key = api_key.strip()
                self.key_source = "Session UI Input"
        else:
            if self.provider == "groq" and env_groq:
                self.groq_key = env_groq
                self.key_source = "Local .env file"
            elif self.provider == "openai" and env_openai:
                self.openai_key = env_openai
                self.key_source = "Local .env file"
            elif env_groq:
                self.groq_key = env_groq
                self.key_source = "Local .env file (Groq)"
            elif env_openai:
                self.openai_key = env_openai
                self.key_source = "Local .env file (OpenAI)"

        # Model defaults
        if self.provider == "local":
            self.model = model or os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        elif self.provider == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "groq":
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            self.model = "Prewritten Verified Dataset"

    def has_active_credentials(self) -> bool:
        """Check if active credentials or local adapter weights are present."""
        if self.provider == "local":
            from local_model_service import get_adapter_status
            status = get_adapter_status()
            return status["is_available"]
        if self.provider == "openai" and bool(self.openai_key):
            return True
        if self.provider == "groq" and bool(self.groq_key):
            return True
        if self.provider not in ["demo", "local"] and (bool(self.groq_key) or bool(self.openai_key)):
            return True
        return False

    def get_active_provider_info(self) -> Dict[str, Any]:
        """Return provider status and masked key without exposing raw secrets."""
        if self.provider == "local":
            from local_model_service import get_adapter_status
            status = get_adapter_status()
            return {
                "provider": "Fine-Tuned Local Model (QLoRA)",
                "model": f"{status['base_model']} + LoRA Adapter",
                "is_configured": status["is_available"],
                "has_groq": bool(self.groq_key),
                "has_openai": bool(self.openai_key),
                "masked_key": "None Required (Local Weights)" if status["is_available"] else "Adapter Missing",
                "key_source": f"Local directory ({status['adapter_path']})" if status["is_available"] else "Not Found",
                "adapter_status": status
            }

        active = self.has_active_credentials()
        eff_provider = self.provider

        if active:
            if eff_provider == "groq" and not self.groq_key and self.openai_key:
                eff_provider = "openai"
            elif eff_provider == "openai" and not self.openai_key and self.groq_key:
                eff_provider = "groq"

        active_key = self.groq_key if eff_provider == "groq" else (self.openai_key if eff_provider == "openai" else "")

        return {
            "provider": eff_provider if active else "None (Demo Mode)",
            "model": self.model if active else "Prewritten Verified Dataset",
            "is_configured": active,
            "has_groq": bool(self.groq_key),
            "has_openai": bool(self.openai_key),
            "masked_key": mask_secret(active_key) if active else "No Key Found",
            "key_source": self.key_source if active else "None",
        }

    @staticmethod
    def clean_json_text(text: str) -> str:
        """Strip markdown fences and whitespace from LLM text."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
            cleaned = re.sub(r"```$", "", cleaned)
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        return cleaned.strip()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 3000
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Sends generation request to selected LLM provider and parses structured JSON.
        Includes single retry on malformed JSON output for cloud APIs.
        """
        # Handle Local Model Generation
        if self.provider == "local":
            from local_model_service import (
                LOCAL_MAX_INPUT_TOKENS,
                LOCAL_MAX_NEW_TOKENS,
                _clear_cuda_cache,
                _is_cuda_oom,
                _model_input_device,
                get_adapter_status,
                clean_local_json_text,
                load_cached_peft_model,
            )
            status = get_adapter_status()
            if not status["is_available"]:
                return None, (
                    f"LORA_ADAPTER_NOT_FOUND: The fine-tuned LoRA adapter was not found at '{status['adapter_path']}'. "
                    "Please run 'GenQuiz_FineTuning.ipynb' in Google Colab (free T4 GPU), download 'genquiz_lora_adapter.zip', "
                    f"and extract it into '{status['adapter_path']}/'. Or switch to Groq, OpenAI, or Demo Mode in the sidebar."
                )

            try:
                import torch
                model, tokenizer = load_cached_peft_model(adapter_path=status["adapter_path"])
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=LOCAL_MAX_INPUT_TOKENS,
                )
                inputs = {k: v.to(_model_input_device(model)) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(LOCAL_MAX_NEW_TOKENS, max_tokens),
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                cleaned = clean_local_json_text(raw_text)
                parsed = json.loads(cleaned)
                return parsed, None
            except json.JSONDecodeError as jde:
                return None, f"Local model output could not be parsed as JSON: {str(jde)}"
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    _clear_cuda_cache()
                    return None, (
                        "LOCAL_CUDA_OOM: GPU memory was exhausted during local generation. "
                        f"Try a shorter PDF or fewer questions (limits: {LOCAL_MAX_INPUT_TOKENS} input / "
                        f"{LOCAL_MAX_NEW_TOKENS} output tokens)."
                    )
                return None, f"Local model execution error: {str(exc)}"
            except Exception as e:
                return None, f"Local model execution error: {str(e)}"

        # Handle Cloud LLM Providers (Groq / OpenAI)
        if not self.has_active_credentials():
            return None, (
                "NO_API_KEY: Live LLM generation requires a valid API key. "
                "Please configure GROQ_API_KEY or OPENAI_API_KEY in your local .env file or sidebar, "
                "or switch to Demo Mode to test using verified dataset records."
            )

        eff_provider = self.provider
        if eff_provider == "groq" and not self.groq_key and self.openai_key:
            eff_provider = "openai"
            self.model = "gpt-4o-mini"
        elif eff_provider == "openai" and not self.openai_key and self.groq_key:
            eff_provider = "groq"
            self.model = "llama-3.3-70b-versatile"

        for attempt in range(2):
            try:
                raw_response = ""
                if eff_provider == "groq":
                    from groq import Groq
                    client = Groq(api_key=self.groq_key)
                    completion = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    raw_response = completion.choices[0].message.content or ""

                elif eff_provider == "openai":
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_key)
                    completion = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    raw_response = completion.choices[0].message.content or ""

                else:
                    return None, f"Unsupported LLM provider: {eff_provider}"

                if not raw_response.strip():
                    if attempt == 0:
                        continue
                    return None, "Received empty response from LLM provider."

                cleaned = self.clean_json_text(raw_response)
                parsed = json.loads(cleaned)
                return parsed, None

            except json.JSONDecodeError as jde:
                if attempt == 0:
                    user_prompt += "\nIMPORTANT: Ensure your output is valid, well-formed JSON only."
                    continue
                return None, f"JSON parsing failed: {str(jde)}"
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "unauthorized" in err_str.lower() or "invalid_api_key" in err_str.lower():
                    return None, f"API Authentication failed ({eff_provider}). The configured API key appears invalid or expired."
                if "429" in err_str or "rate limit" in err_str.lower():
                    return None, f"API Rate limit reached ({eff_provider}). Please wait a moment before retrying."
                return None, f"LLM API Error ({eff_provider}): {err_str}"

        return None, "Failed to obtain valid JSON from LLM after 2 attempts."
