from __future__ import annotations

import json
import signal
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from albedo_eval_service.remote.dataset import EvalSample
from albedo_eval_service.remote.generation import (
    _CONTEXT_SAFETY_MARGIN_TOKENS,
    VllmServerGenerator,
)


def _make_gen(**overrides) -> VllmServerGenerator:
    kwargs = dict(
        model="/models/m",
        gpu_ids=["4", "5", "6", "7"],
        port=9302,
        max_new_tokens=16,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_model_len=100 + _CONTEXT_SAFETY_MARGIN_TOKENS,
        compile_cache_dir="/cache",
    )
    kwargs.update(overrides)
    return VllmServerGenerator(**kwargs)


class _Tok:
    def __call__(self, prompt):
        return SimpleNamespace(input_ids=list(range(len(prompt))))  # 1 token per char


class _Running:
    pid = 4321

    def poll(self):
        return None


class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


class _Client:
    """Stand-in for httpx.Client: `handler(body)` returns (status, json)."""

    def __init__(self, handler):
        self.handler = handler
        self.bodies = []

    def post(self, path, json):
        self.bodies.append(json)
        return _Response(*self.handler(json))


def _completion(text, finish_reason, tokens):
    return 200, {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": tokens},
    }


def _ready(gen, handler):
    gen._process = _Running()
    gen._tokenizer = _Tok()
    gen._client = _Client(handler)
    return gen


def test_command_mirrors_the_engine_settings():
    cmd = _make_gen()._command()
    assert cmd[1:3] == ["serve", "/models/m"]
    expected = {
        "--port": "9302",
        "--tensor-parallel-size": "4",
        "--max-model-len": "164",
        "--max-num-seqs": "256",
        "--generation-config": "vllm",
        "--kv-cache-dtype": "auto",
        "--gpu-memory-utilization": "0.95",
    }
    for flag, value in expected.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert "--enable-prefix-caching" in cmd
    assert "--enforce-eager" not in cmd
    assert json.loads(cmd[cmd.index("--compilation-config") + 1]) == {"cache_dir": "/cache"}
    assert "--enforce-eager" in _make_gen(enforce_eager=True)._command()


def test_truncation_only_when_the_per_response_cap_is_hit():
    answers = iter(
        [_completion("a", "length", 16), _completion("b", "length", 9), _completion("c", "stop", 3)]
    )
    gen = _ready(_make_gen(), lambda body: next(answers))

    results = gen.generate([EvalSample(f"s{i}", "p" * 10) for i in range(3)])

    assert [r.truncated for r in results] == [True, False, False]
    assert [r.text for r in results] == ["a", "b", "c"]
    body = gen._client.bodies[0]
    assert body["stop_token_ids"] == [248046]
    assert (body["max_tokens"], body["temperature"], body["top_p"], body["top_k"]) == (
        16,
        1.0,
        0.95,
        20,
    )


def test_context_exhausted_prompt_is_sidelined_without_a_request():
    gen = _ready(_make_gen(), lambda body: _completion("x", "stop", 1))

    results = gen.generate([EvalSample("s1", "a" * 20), EvalSample("s2", "b" * 150)])

    assert [r.truncated for r in results] == [False, True]
    assert [r.text for r in results] == ["x", ""]
    assert len(gen._client.bodies) == 1  # the over-budget prompt never reached the server


def test_request_failure_becomes_an_error_result():
    gen = _ready(_make_gen(max_model_len=None), lambda body: (503, {}))

    (result,) = gen.generate([EvalSample("s1", "p")])

    assert result.text == ""
    assert result.error and "HTTPStatusError" in result.error


def test_close_kills_the_server_group_and_escalates(monkeypatch):
    killed = []
    monkeypatch.setattr(
        "albedo_eval_service.remote.generation.os.killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    waits = iter([subprocess.TimeoutExpired("vllm", 30), None])

    def wait(timeout):
        outcome = next(waits)
        if outcome:
            raise outcome

    gen = _make_gen()
    gen._process = SimpleNamespace(pid=4321, wait=wait)
    gen.close()

    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert gen._process is None


def test_start_waits_for_health_then_loads_the_tokenizer(monkeypatch):
    spawned = {}

    def popen(cmd, env, start_new_session):
        spawned.update(cmd=cmd, env=env)
        return _Running()

    monkeypatch.setattr("albedo_eval_service.remote.generation.subprocess.Popen", popen)
    monkeypatch.setattr("albedo_eval_service.remote.generation._load_tokenizer", lambda m: _Tok())
    monkeypatch.setattr("albedo_eval_service.remote.generation.time.sleep", lambda s: None)
    health = iter([httpx.ConnectError("not yet"), SimpleNamespace(status_code=200)])

    def get(path, timeout):
        outcome = next(health)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    gen = _make_gen()
    gen._client = SimpleNamespace(get=get)
    gen._start()

    assert spawned["env"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert spawned["cmd"][1] == "serve"
    assert isinstance(gen._tokenizer, _Tok)


def test_start_fails_fast_when_the_server_dies(monkeypatch):
    class _Dead:
        pid = 7
        returncode = 3

        def poll(self):
            return 3

    monkeypatch.setattr(
        "albedo_eval_service.remote.generation.subprocess.Popen", lambda *a, **k: _Dead()
    )
    with pytest.raises(RuntimeError, match="exited 3"):
        _make_gen()._start()
