"""Single entry point for all LLM calls: `call(group, messages, schema)`.

A LiteLLM Router (in-process) maps two model groups to deployments:

    generator -> Groq large model   --fallback-->  Gemini flash        (SQL, answers)
    judge     -> Groq small model   --fallback-->  Gemini flash-lite   (validation, M3)

The router tracks each Groq deployment's requests/tokens per minute and routes to the
Gemini fallback *before* Groq would return a 429 (docs/DESIGN.md §8).
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm
    from litellm import Router

litellm.suppress_debug_info = True
# LiteLLM builds some pydantic types lazily on first call; this warning is about its internals.
warnings.filterwarnings("ignore", message=".*ReadOnly.*", category=UserWarning)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

Group = Literal["generator", "judge"]
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """All providers failed, or the model kept returning invalid output."""


@dataclass
class LLMResult(Generic[T]):
    data: T
    model: str          # the deployment that actually answered, e.g. "groq/openai/gpt-oss-120b"
    fell_back: bool     # True if the fallback provider answered
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


def call(group: Group, messages: list[dict], schema: type[T], *, router: Router | None = None) -> LLMResult[T]:
    """Call a model group and parse its reply into `schema`.

    One repair attempt is made if the reply isn't valid JSON for the schema.
    """
    router = router or get_router()
    start = time.perf_counter()
    response = _complete(router, group, messages, schema)
    content = response.choices[0].message.content or ""
    try:
        data = schema.model_validate_json(_strip_fences(content))
    except ValidationError as e:
        repair = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": (
                f"Your reply did not match the required JSON schema: {_short(e)}\n"
                "Reply again with ONLY a valid JSON object matching the schema."
            )},
        ]
        response = _complete(router, group, repair, schema)
        content = response.choices[0].message.content or ""
        try:
            data = schema.model_validate_json(_strip_fences(content))
        except ValidationError as e2:
            raise LLMError(f"Model returned invalid output twice: {_short(e2)}") from e2

    model = _deployment_name(router, response)
    usage = getattr(response, "usage", None)
    return LLMResult(
        data=data,
        model=model,
        fell_back=model != _primary_model(group),
        latency_ms=int((time.perf_counter() - start) * 1000),
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _complete(router: Router, group: Group, messages: list[dict], schema: type[BaseModel]):
    try:
        return router.completion(model=group, messages=messages, response_format=schema)
    except Exception as e:  # LiteLLM raises many provider-specific types
        raise LLMError(f"All providers failed for '{group}': {type(e).__name__}: {_short(e)}") from e


@lru_cache
def get_router() -> Router:
    return build_router(get_settings())


def build_router(s: Settings) -> Router:
    if not s.groq_api_key or not s.gemini_api_key:
        raise LLMError("GROQ_API_KEY and GEMINI_API_KEY must be set in .env")

    def deployment(group: str, model: str) -> dict:
        is_groq = model.startswith("groq/")
        params: dict = {
            "model": model,
            "api_key": s.groq_api_key if is_groq else s.gemini_api_key,
            "timeout": s.llm_timeout_s,
        }
        if is_groq:
            params |= {"temperature": 0, "rpm": s.groq_rpm, "tpm": s.groq_tpm}
        else:
            # Gemini 3+ deprecates sampling params; it is deterministic enough without them.
            params |= {"rpm": s.gemini_rpm}
        return {"model_name": group, "litellm_params": params}

    return Router(
        model_list=[
            deployment("generator", s.generator_model),
            deployment("generator-fallback", s.generator_fallback_model),
            deployment("judge", s.judge_model),
            deployment("judge-fallback", s.judge_fallback_model),
        ],
        fallbacks=[{"generator": ["generator-fallback"]}, {"judge": ["judge-fallback"]}],
        routing_strategy="usage-based-routing-v2",
        num_retries=0,          # don't hammer a rate-limited provider; fall back instead
        allowed_fails=1,
        cooldown_time=60,
    )


def _primary_model(group: Group) -> str:
    s = get_settings()
    return s.generator_model if group == "generator" else s.judge_model


def _deployment_name(router: Router, response) -> str:
    """Map the response back to our configured model string (e.g. 'gemini/gemini-3.5-flash')."""
    model_id = (getattr(response, "_hidden_params", None) or {}).get("model_id")
    if model_id:
        for d in router.model_list:
            if d.get("model_info", {}).get("id") == model_id:
                return d["litellm_params"]["model"]
    return getattr(response, "model", "unknown")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _short(e: Exception, n: int = 300) -> str:
    s = " ".join(str(e).split())
    return s if len(s) <= n else s[: n - 1] + "…"
