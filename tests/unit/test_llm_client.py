from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import get_settings
from app.llm.client import LLMError, build_router, call


class Out(BaseModel):
    answer: int


class FakeRouter:
    """Stands in for litellm.Router: returns queued replies, records requests."""

    def __init__(self, replies, model_id="primary-id"):
        self.replies = list(replies)
        self.requests = []
        self.model_list = [
            {"model_info": {"id": "primary-id"}, "litellm_params": {"model": get_settings().generator_model}},
            {"model_info": {"id": "fallback-id"}, "litellm_params": {"model": "gemini/some-flash"}},
        ]
        self.model_id = model_id

    def completion(self, model, messages, response_format):
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            _hidden_params={"model_id": self.model_id},
        )


MSGS = [{"role": "user", "content": "17 + 25?"}]


def test_parses_structured_reply():
    r = call("generator", MSGS, Out, router=FakeRouter(['{"answer": 42}']))
    assert r.data.answer == 42
    assert r.model == get_settings().generator_model
    assert not r.fell_back
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)


def test_reports_fallback_deployment():
    r = call("generator", MSGS, Out, router=FakeRouter(['{"answer": 42}'], model_id="fallback-id"))
    assert r.model == "gemini/some-flash"
    assert r.fell_back


def test_strips_markdown_fences():
    r = call("generator", MSGS, Out, router=FakeRouter(['```json\n{"answer": 7}\n```']))
    assert r.data.answer == 7


def test_repairs_invalid_json_once():
    router = FakeRouter(["not json", '{"answer": 1}'])
    r = call("generator", MSGS, Out, router=router)
    assert r.data.answer == 1
    repair_prompt = router.requests[1][-1]["content"]
    assert "did not match the required JSON schema" in repair_prompt


def test_gives_up_after_second_invalid_reply():
    with pytest.raises(LLMError, match="invalid output twice"):
        call("generator", MSGS, Out, router=FakeRouter(["nope", '{"answer": "x"}']))


def test_provider_errors_become_llm_error():
    with pytest.raises(LLMError, match="All providers failed"):
        call("generator", MSGS, Out, router=FakeRouter([RuntimeError("429 everywhere")]))


class TestRouterConfig:
    def test_groups_and_fallbacks(self):
        s = get_settings().model_copy(update={"groq_api_key": "g", "gemini_api_key": "m"})
        router = build_router(s)
        groups = {d["model_name"]: d["litellm_params"] for d in router.model_list}
        assert set(groups) == {"generator", "generator-fallback", "judge", "judge-fallback"}
        assert groups["generator"]["model"] == s.generator_model
        assert groups["generator"]["tpm"] == s.groq_tpm
        assert groups["generator"]["temperature"] == 0
        # Gemini 3+ deprecates sampling params.
        assert "temperature" not in groups["generator-fallback"]

    def test_missing_keys(self):
        s = get_settings().model_copy(update={"groq_api_key": None})
        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            build_router(s)
