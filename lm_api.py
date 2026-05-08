# 
import os
from typing import Optional, Tuple

import openai
import backoff
import warnings


def get_client_from_env() -> Optional[openai.OpenAI]:
    """Create client from OPENAI_API_KEY, DEEPSEEK_API_KEY, or GOOGLE/GEMINI key for default client at import time."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"
    if not api_key and os.getenv("DEEPSEEK_API_KEY"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_client(model_name: str) -> openai.OpenAI:
    """Initialize and return the API client based on the model type"""

    if model_name.startswith("gpt-"):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith(("o1-", "o3-", "o4-")):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
    elif model_name.startswith("deepseek-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
    elif model_name.startswith("ollama/"):
        # https://ollama.com/blog/openai-compatibility
        base_url = 'http://localhost:11434/v1'
        api_key='ollama', # required, but unused
    elif model_name.startswith("vllm/"):
        # vLLM exposes an OpenAI-compatible API.
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")  # required by client, often unused locally
    elif model_name.startswith("gemini"):
        # Gemini via OpenAI-compatible endpoint (Google AI Studio / Gemini API).
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not api_key:
            raise ValueError(
                "Gemini models require GEMINI_API_KEY or GOOGLE_API_KEY "
                "(see https://ai.google.dev/gemini-api/docs/openai)."
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


# NOTE: price per 1M token, recorded on 2025-03-23
API_PRICING = {
    # OpenAI models
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
    "gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "o1-2024-12-17": {"input": 15.0, "output": 60.0},
    "o1-pro-2025-03-19": {"input": 150.0, "output": 600.0},
    "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    # GPT-5 family (reasoning). Output billing includes internal thinking tokens.
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Deepseek models  (using standard price. discount price would be lower)
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Gemini (approximate; verify ai.google.dev pricing)
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    # Gemini 3.1 Pro Preview pricing (paid tier, prompts ≤200K tokens). Output
    # includes thinking tokens. Above 200K, OpenAI-compat layer can't infer
    # tier — billed costs may be higher than this estimate.
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00},
}


def calculate_api_cost(prompt_tokens, completion_tokens, model) -> float:
    if model in API_PRICING:
        input_cost = (prompt_tokens / 1_000_000) * API_PRICING[model]["input"]
        output_cost = (completion_tokens / 1_000_000) * API_PRICING[model]["output"]
        return input_cost + output_cost
    else:
        warnings.warn(f"Unknown model: {model}. No pricing information.", UserWarning)
        return float('nan')
    

def is_reasoning_model(model_name: str) -> bool:
    """True for OpenAI reasoning families (o1, o3, o4, gpt-5) that reject chat-completion kwargs."""
    base = model_name.split("/", 1)[-1]
    return base.startswith(("o1-", "o1", "o3-", "o3", "o4-", "o4", "gpt-5"))


def _prepare_reasoning_call(system_message: str, user_message: str, chat_kwargs: dict):
    """Adapt chat.completions kwargs for o1/o3/o4 reasoning models.

    - merge system into user (o1-mini rejects system role)
    - rename max_tokens -> max_completion_tokens
    - drop top_p / stop (unsupported)
    - drop temperature if not equal to the fixed default (1.0)
    - bump very small timeouts (reasoning models are slow)
    """
    merged_user = user_message
    if system_message:
        merged_user = f"{system_message}\n\n{user_message}"
    messages = [{"role": "user", "content": merged_user}]

    kw = dict(chat_kwargs)
    # Reasoning models count internal reasoning against max_completion_tokens,
    # so small caps (e.g. 128) leave no room for the visible answer.
    reasoning_floor = 8192
    if "max_tokens" in kw:
        kw["max_completion_tokens"] = max(int(kw.pop("max_tokens")), reasoning_floor)
    else:
        kw.setdefault("max_completion_tokens", reasoning_floor)
    kw.pop("top_p", None)
    kw.pop("stop", None)
    if "temperature" in kw and float(kw["temperature"]) != 1.0:
        kw.pop("temperature")
    if "timeout" in kw and kw["timeout"] is not None and float(kw["timeout"]) < 60:
        kw["timeout"] = 120
    return messages, kw


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_llm(client: openai.OpenAI, model: str,
              system_message: str, user_message: str,
              chat_kwargs: dict = {}) -> Tuple[openai.ChatCompletion, float]:

    if model.startswith("ollama/"):
        model_name = model.split("ollama/")[1]  # local ollama models
    elif model.startswith("vllm/"):
        model_name = model.split("vllm/")[1]  # local/remote vLLM served models
    elif model.startswith("gemini/"):
        model_name = model.split("gemini/", 1)[1]  # e.g. gemini/gemini-2.5-flash
    else:
        model_name = model

    if is_reasoning_model(model_name):
        messages, call_kwargs = _prepare_reasoning_call(system_message, user_message, chat_kwargs)
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        call_kwargs = dict(chat_kwargs)
        if model_name == "deepseek-reasoner" and "max_tokens" in call_kwargs:
            call_kwargs["max_tokens"] = max(int(call_kwargs["max_tokens"]), 8192)
        # Gemini 2.5 family are thinking models: reasoning tokens count against
        # max_tokens. Two failure modes were observed:
        #   - low cap (e.g. 512) → thinking eats budget, visible answer cut mid-sentence
        #   - high cap (e.g. 8192) → model has room to continue past the answer and
        #     roleplay extra "env" turns (e.g. "Yes.Okay, let's continue. You are in a room...")
        # Disable thinking via OpenAI-compat reasoning_effort="none" so the visible
        # content is just the answer; keep the caller's max_tokens unchanged. Pro
        # is left thinking-on but with an 8192 floor since it benefits more.
        if model_name.startswith("gemini-2.5-flash"):
            extra = dict(call_kwargs.get("extra_body") or {})
            extra.setdefault("reasoning_effort", "none")
            call_kwargs["extra_body"] = extra
        elif model_name.startswith("gemini-2.5-pro"):
            floor = 8192
            if "max_tokens" in call_kwargs:
                call_kwargs["max_tokens"] = max(int(call_kwargs["max_tokens"]), floor)
            else:
                call_kwargs["max_tokens"] = floor
        # Local models (ollama / vllm) are slower than cloud APIs; the 12s
        # action-call timeout tuned for cloud was forcing massive retry storms.
        if model.startswith(("ollama/", "vllm/")):
            t = call_kwargs.get("timeout")
            if t is None or float(t) < 120:
                call_kwargs["timeout"] = 180
        # Gemini's OpenAI-compat shim rejects fields explicitly set to null
        # (e.g. "stop": None → "Value is not a string: null"), so strip them.
        if model.startswith("gemini") or "generativelanguage.googleapis.com" in str(getattr(client, "base_url", "")):
            call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **call_kwargs,
    )

    if model.startswith(("ollama/", "vllm/")):
        cost = 0.0  # locally hosted model
    else:
        cost = calculate_api_cost(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=model_name
        )

    return response, cost