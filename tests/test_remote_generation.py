from __future__ import annotations

import signal
from types import SimpleNamespace

from albedo_eval_service.remote.generation import VllmProcessGenerator


def _make_gen() -> VllmProcessGenerator:
    return VllmProcessGenerator(
        model="m", gpu_ids=["0"], max_new_tokens=1, temperature=0.0, top_p=1.0
    )


def test_kill_process_tree_kills_the_whole_group(monkeypatch):
    """Teardown must SIGKILL the worker's process group, so the vLLM EngineCore/Workers it
    spawned (which share the group via os.setsid) are reaped instead of orphaned holding GPU mem."""
    calls = []
    killed = {}
    gen = _make_gen()
    gen._process = SimpleNamespace(pid=4321, terminate=lambda: calls.append("terminate"))
    monkeypatch.setattr("albedo_eval_service.remote.generation.os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        "albedo_eval_service.remote.generation.os.killpg",
        lambda pgid, sig: killed.update(pgid=pgid, sig=sig),
    )
    gen._kill_process_tree()
    assert killed == {"pgid": 4321, "sig": signal.SIGKILL}
    assert calls == []  # did not need the terminate fallback


def test_kill_process_tree_falls_back_to_terminate_when_group_gone(monkeypatch):
    calls = []
    gen = _make_gen()
    gen._process = SimpleNamespace(pid=4321, terminate=lambda: calls.append("terminate"))
    monkeypatch.setattr("albedo_eval_service.remote.generation.os.getpgid", lambda pid: pid)

    def boom(pgid, sig):
        raise ProcessLookupError

    monkeypatch.setattr("albedo_eval_service.remote.generation.os.killpg", boom)
    gen._kill_process_tree()
    assert calls == ["terminate"]


def test_kill_process_tree_terminates_when_pid_missing():
    calls = []
    gen = _make_gen()
    gen._process = SimpleNamespace(pid=None, terminate=lambda: calls.append("terminate"))
    gen._kill_process_tree()
    assert calls == ["terminate"]


class _FakeTok:
    def __call__(self, prompt):
        return SimpleNamespace(input_ids=list(range(len(prompt))))  # 1 token per char


class _FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"
        self.token_ids = [1, 2, 3]


class _FakeLLM:
    def __init__(self):
        self.seen_prompts = None

    def get_tokenizer(self):
        return _FakeTok()

    def generate(self, prompts, params):
        self.seen_prompts = list(prompts)
        return [SimpleNamespace(outputs=[_FakeCompletion(f"out:{p[:4]}")]) for p in prompts]


def test_generate_payload_sidelines_context_exhausted_prompts():
    from albedo_eval_service.remote.generation import (
        _CONTEXT_SAFETY_MARGIN_TOKENS,
        _generate_payload,
    )

    llm = _FakeLLM()
    max_model_len = 100 + _CONTEXT_SAFETY_MARGIN_TOKENS
    prompts = ["a" * 20, "b" * 150, "c" * 30]  # middle one exceeds the 100-token budget
    payload = _generate_payload(
        llm, None, prompts, ["s1", "s2", "s3"], 4096, max_model_len=max_model_len
    )
    results = {r["sample_id"]: r for r in payload["results"]}
    assert llm.seen_prompts == ["a" * 20, "c" * 30]  # over-budget prompt never sent
    assert results["s2"] == {"sample_id": "s2", "text": "", "error": None, "truncated": True}
    assert results["s1"]["text"] == "out:aaaa" and not results["s1"]["truncated"]
    assert results["s3"]["text"] == "out:cccc" and not results["s3"]["truncated"]
    assert [r["sample_id"] for r in payload["results"]] == ["s1", "s2", "s3"]  # order kept


def test_generate_payload_all_exhausted_skips_generate():
    from albedo_eval_service.remote.generation import _generate_payload

    llm = _FakeLLM()
    payload = _generate_payload(llm, None, ["x" * 500], ["s1"], 4096, max_model_len=100)
    assert llm.seen_prompts is None  # generate never called
    assert payload["results"][0]["truncated"] is True


def test_generate_payload_no_max_model_len_keeps_old_behavior():
    from albedo_eval_service.remote.generation import _generate_payload

    llm = _FakeLLM()
    payload = _generate_payload(llm, None, ["y" * 500], ["s1"], 4096, max_model_len=None)
    assert llm.seen_prompts == ["y" * 500]
    assert payload["results"][0]["truncated"] is False
