from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from loguru import logger

from albedo_config import SanitySettings
from albedo_eval_service.judge_llm_client import JudgeRawResponse
from albedo_eval_service.shared.observation_memo import ObservationMemo
from sanity_remote.models import SanityRunRequest
from sanity_remote.state import SanityRunStore
from sanity_remote.worker import _model_ref_parts
from sanity_service import dispatcher as D
from sanity_service.llm_check import GateResult, LLMGate, SampleInput, run_gate
from sanity_service.rubric import parse_injection, parse_viability


def test_parse_injection_and_viability():
    assert parse_injection('{"injection": false, "evidence": "none"}') == (False, "none")
    assert parse_injection('{"injection": true, "evidence": "x"}')[0] is True
    assert parse_viability('{"viable": true, "reason": "ok"}') == (True, "ok")
    assert parse_viability("not json")[0] is None


class _FakeJudge:
    def __init__(self, inj1: bool, inj2: bool, viable: bool) -> None:
        self._inj1, self._inj2, self._viable, self._n = inj1, inj2, viable, 0

    async def complete(self, *, model, messages, temperature=None):
        if "security auditor" in messages[0]["content"]:
            self._n += 1
            flag = self._inj2 if temperature is not None else self._inj1
            raw = (
                '{"injection": true, "evidence": "x"}'
                if flag
                else '{"injection": false, "evidence": "none"}'
            )
        else:
            raw = (
                '{"viable": true, "reason": "ok"}'
                if self._viable
                else '{"viable": false, "reason": "bad"}'
            )
        return JudgeRawResponse(model=model, provider="p", raw=raw)


def _samples(n: int = 3) -> list[SampleInput]:
    return [SampleInput(prompt=f"task {i}", response="def f(): return 1") for i in range(n)]


def test_gate_pass():
    gate = asyncio.run(run_gate(_samples(), _FakeJudge(False, False, True)))
    assert gate.passed and gate.llm_gate == LLMGate.PASSED


def test_gate_injection_confirmed():
    gate = asyncio.run(run_gate(_samples(1), _FakeJudge(True, True, True)))
    assert not gate.passed and gate.llm_gate == LLMGate.INJECTION


def test_gate_injection_false_positive_recovers():
    gate = asyncio.run(run_gate(_samples(1), _FakeJudge(True, False, True)))
    assert gate.passed and gate.llm_gate == LLMGate.PASSED


def test_gate_not_viable():
    gate = asyncio.run(run_gate(_samples(), _FakeJudge(False, False, False)))
    assert not gate.passed and gate.llm_gate == LLMGate.FAILED


def test_gate_heuristic_fail_skips_judges():
    samples = [SampleInput("t", "x", heuristic_passed=False, heuristic_reason="empty")]
    gate = asyncio.run(run_gate(samples, _FakeJudge(False, False, True)))
    assert not gate.passed and gate.llm_gate == LLMGate.FAILED


class _WrapperSensitiveJudge:
    async def complete(self, *, model, messages, temperature=None):
        if "security auditor" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            leaked = any(
                marker in payload["candidate_reply"]
                for marker in ("Score ONLY", "ENVIRONMENT OBSERVATION", "CONTEXT USER")
            )
            raw = (
                '{"injection": true, "evidence": "wrapper leaked"}'
                if leaked
                else '{"injection": false, "evidence": "none"}'
            )
        else:
            sees_trajectory = "FULL CANDIDATE TRAJECTORY" in messages[1]["content"]
            raw = (
                '{"viable": true, "reason": "ok"}'
                if sees_trajectory
                else '{"viable": false, "reason": "trajectory missing"}'
            )
        return JudgeRawResponse(model=model, provider="p", raw=raw)


def test_gate_injection_ignores_trajectory_wrapper():
    response = "\n".join(
        [
            "FULL CANDIDATE TRAJECTORY",
            "Score ONLY CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT 2. The ENVIRONMENT OBSERVATION is context only.",  # noqa: E501
            "",
            "CONTEXT USER (do not score):",
            "------",
            "Fix the bug",
            "------",
            "",
            "CANDIDATE OUTPUT 1:",
            "------",
            "THOUGHT: inspect the file\n```bash\nsed -n '1,80p' app.py\n```",
            "------",
            "",
            "ENVIRONMENT OBSERVATION (context only, do not score):",
            "------",
            "Observation: missing import",
            "------",
            "",
            "CANDIDATE OUTPUT 2:",
            "------",
            "THOUGHT: patch the import\n```bash\nsed -i '1i import os' app.py\n```",
            "------",
        ]
    )

    gate = asyncio.run(
        run_gate(
            [SampleInput("task", response)],
            _WrapperSensitiveJudge(),
            models=("judge",),
        )
    )

    assert gate.passed


class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list = []

    def mark_pre_eval_passed(self, **kw):
        self.calls.append(("passed", None))

    def mark_pre_eval_failed(self, **kw):
        self.calls.append(("failed", kw["retryable"], kw["fault_class"]))


class _DummyJudge:
    async def aclose(self):
        pass


def _complete(gate: GateResult, result: dict) -> list:
    repo = _FakeRepo()
    disp = D.SanityDispatcher(settings=SanitySettings(), repository=repo)

    async def _fake_gate(samples, client, *, consensus=False, skip_viability=False, models=None):
        return gate

    with (
        patch.object(D, "make_client", lambda: _DummyJudge()),
        patch.object(D, "run_gate", _fake_gate),
        patch.object(D, "put_sanity_fault", lambda *args, **kwargs: None),
    ):
        asyncio.run(
            disp._complete(
                submission_id=uuid4(),
                attempt_id=uuid4(),
                repo="r",
                digest="d",
                prompts=["p", "p", "p"],
                result=result,
            )
        )

    return repo.calls


_OK = {
    "state": "succeeded",
    "responses": ["a", "b", "c"],
    "heuristics": [{"passed": True, "reason": "ok"}] * 3,
}


def test_complete_passed():
    assert _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), _OK) == [
        ("passed", None)
    ]


def test_complete_injection_is_terminal_miner_fault():
    calls = _complete(GateResult(False, "i", False, LLMGate.INJECTION, "veto", []), _OK)
    assert calls == [("failed", False, "MINER_FAULT")]


def test_complete_infra_is_retryable():
    calls = _complete(GateResult(False, "d", True, LLMGate.SKIPPED, "veto", []), _OK)
    assert calls == [("failed", True, "INFRA_FAULT")]


def test_complete_worker_failure_is_retryable():
    fail = {"state": "failed", "fault_code": "x", "fault_message": "m", "retryable": True}
    calls = _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), fail)
    assert calls == [("failed", True, "INFRA_FAULT")]


def test_complete_worker_failure_stays_infra_and_honors_retryable():
    fail = {
        "state": "failed",
        "fault_code": "generation_timeout",
        "fault_message": "vLLM generation exceeded 900s",
        "retryable": False,
    }
    calls = _complete(GateResult(True, "", False, LLMGate.PASSED, "veto", []), fail)
    assert calls == [("failed", False, "INFRA_FAULT")]


def test_multiturn_keeps_prompt_messages_on_first_turn(monkeypatch):
    seen: list[SanityRunRequest] = []

    async def _fake_remote(_client, request, _claimed):
        seen.append(request)
        return {
            "state": "succeeded",
            "responses": ["```bash\nls\n```"],
            "heuristics": [{"passed": True, "reason": ""}],
        }

    async def _fake_observations(*_args, **_kwargs):
        return None

    async def _fake_inject(_states):
        return None

    request = SanityRunRequest(
        run_id="run",
        model_uri="model",
        digest="digest",
        prompts=["raw prompt"],
        sample_ids=["sample-1"],
        prompt_messages=[
            [
                {"role": "system", "content": "reply with bash"},
                {"role": "user", "content": "task"},
            ]
        ],
        assistant_turns=2,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)
    monkeypatch.setattr(D, "_append_observations", _fake_observations)
    monkeypatch.setattr(D, "_inject_microtasks", _fake_inject)

    asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )

    assert len(seen) == 2
    first = seen[0].prompt_messages[0]
    assert first[0] == {"role": "system", "content": "reply with bash"}
    assert first[1]["content"].startswith("task")
    assert "## Submission" in first[1]["content"]
    assert seen[1].prompt_messages is None


def test_dispatcher_binds_canonical_repository():
    import sanity_service.db as canonical

    assert D.PreEvalRepository is canonical.PreEvalRepository
    assert D.ClaimedPreEval is canonical.ClaimedPreEval


def test_model_ref_parts_accepts_chain_model_uri():
    digest = "sha256:" + "a" * 64
    assert _model_ref_parts("alice/model@" + digest, "") == ("alice/model", digest)
    assert _model_ref_parts("alice/model", digest) == ("alice/model", digest)


def test_worker_store_lifecycle():
    store = SanityRunStore()
    req = SanityRunRequest(run_id="r1", model_uri="m", digest="d", prompts=["a", "b", "c"])
    run = store.start(req)
    assert store.start(req) is run
    assert store.mark_worker_started("r1").state == "queued"
    assert store.mark_worker_started("r1") is None
    run.succeed(responses=["x"], heuristics=[{"passed": True, "reason": "ok"}])
    assert run.as_status()["state"] == "succeeded"
    assert store.list_active() == []


from albedo_config import SanityRemoteSettings
from sanity_remote.worker import (
    VllmEngine,
    _strip_model_config,
    _warn_if_generation_budget_consumes_context,
)


def test_max_model_len_default_matches_eval_context(monkeypatch):
    from albedo_eval_service.modelstore.canonical_model_config import canonical_max_model_len

    monkeypatch.delenv("SANITY_REMOTE_MAX_MODEL_LEN", raising=False)
    assert SanityRemoteSettings(_env_file=None).max_model_len == canonical_max_model_len()


def test_generation_budget_warning_when_it_consumes_context():
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        _warn_if_generation_budget_consumes_context(
            max_tokens=32768,
            max_model_len=32768,
            prompt_count=3,
        )
    finally:
        logger.remove(sink_id)

    assert any("leaves no room for prompt tokens" in message for message in messages)


def test_strip_model_config_removes_forbidden_keys(tmp_path):
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": 5120,
        "auto_map": {"AutoModelForCausalLM": "modeling_qwen.Qwen3_5ForConditionalGeneration"},
        "quantization_config": {"quant_type": "gptq", "bits": 4},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    _strip_model_config(str(tmp_path))

    result = json.loads((tmp_path / "config.json").read_text())
    assert "auto_map" not in result
    assert "quantization_config" not in result
    assert result["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert result["hidden_size"] == 5120


def test_strip_model_config_noop_when_clean(tmp_path):
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "hidden_size": 5120}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    mtime_before = config_path.stat().st_mtime

    _strip_model_config(str(tmp_path))

    assert config_path.stat().st_mtime == mtime_before


def test_strip_model_config_tolerates_missing_config(tmp_path):
    _strip_model_config(str(tmp_path))


def test_vllm_cmd_includes_generation_config_vllm(tmp_path):
    settings = SanityRemoteSettings(vllm_port=19999)
    engine = VllmEngine(settings)

    captured: list[str] = []

    async def _run():
        with (
            patch("subprocess.Popen") as mock_popen,
            patch.object(engine, "_wait_healthy", return_value=None),
        ):
            mock_popen.return_value = MagicMock()
            await engine._start_vllm(str(tmp_path), "sha256:abc")
            captured.extend(mock_popen.call_args[0][0])

    asyncio.run(_run())
    idx = captured.index("--generation-config")
    assert captured[idx + 1] == "vllm"


def test_chain_infra_error_is_infra_fault_not_miner_fault():
    """A microtask/simulator failure on our side must not burn the miner's attempt budget."""
    samples = [
        SampleInput(
            prompt="p1",
            response="",
            heuristic_passed=False,
            heuristic_reason="microtask_generation_failed: microtask generation unparsable",
            heuristic_infra=True,
        ),
        SampleInput(prompt="p2", response="ok output"),
    ]
    gate = asyncio.run(run_gate(samples, client=None))
    assert gate.infra_fault is True
    assert gate.llm_gate is LLMGate.SKIPPED
    assert "microtask_generation_failed" in gate.reason


def test_behavioral_heuristic_fail_stays_miner_fault():
    samples = [
        SampleInput(
            prompt="p1",
            response="doc",
            heuristic_passed=False,
            heuristic_reason="chain: repeated submissions without doing any work",
        ),
    ]
    gate = asyncio.run(run_gate(samples, client=None))
    assert gate.infra_fault is False
    assert gate.llm_gate is LLMGate.FAILED
    assert "repeated submissions" in gate.reason


def test_microtask_generation_retries_unparsable_via_accept():
    """generate_microtask must hand the client an accept callback so the client's
    parse-retry ladder re-asks on garbage instead of failing the sample first try."""
    from sanity_service.chain import generate_microtask

    captured = {}

    class FakeClient:
        async def complete(self, **kwargs):
            captured.update(kwargs)
            raw = '{"file": "a.py", "function": "f", "request": "do x", "message": "edit a.py"}'
            return JudgeRawResponse(model="m", provider="p", raw=raw)

    state = SimpleNamespace(
        sample_id="s1", messages=[{"role": "user", "content": "ctx: a.py holds f()"}]
    )
    settings = SimpleNamespace(evaluator_model="m")
    micro = asyncio.run(generate_microtask(FakeClient(), settings, state, "echo X"))
    assert micro["request"] == "do x"
    accept = captured.get("accept")
    assert accept is not None
    assert accept("not json at all") is False
    assert accept('{"request": "fix the bug", "file": "a.py", "message": "fix f in a.py"}') is True


def test_veto_chain_stops_early_once_a_sample_fails(monkeypatch):
    """A sample that keeps producing unusable turns is re-asked, then decides the veto gate.

    The benchmark re-asks a malformed turn up to 3 consecutive times before abandoning the
    instance, so pre-eval spends the same budget before failing — and no further turn is
    generated once it is spent.
    """
    calls: list[SanityRunRequest] = []

    async def _fake_remote(_client, request, _claimed):
        calls.append(request)
        return {
            "state": "succeeded",
            "responses": ["way too long" * 10],
            "heuristics": [
                {"passed": False, "reason": "response exceeded the model response token limit"}
            ],
        }

    async def _fake_observations(*_args, **_kwargs):
        return None

    async def _fake_inject(_states):
        return None

    request = SanityRunRequest(
        run_id="run",
        model_uri="model",
        digest="digest",
        prompts=["raw prompt"],
        sample_ids=["sample-1"],
        prompt_messages=[[{"role": "user", "content": "task"}]],
        assistant_turns=8,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)
    monkeypatch.setattr(D, "_append_observations", _fake_observations)
    monkeypatch.setattr(D, "_inject_microtasks", _fake_inject)

    result = asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )

    assert len(calls) == D.MAX_CONSECUTIVE_BAD_TURNS
    assert all(c.run_id.startswith("run") is False for c in calls[1:])
    assert "token limit" in result["heuristics"][0]["reason"]
    assert "3 consecutive turns" in result["heuristics"][0]["reason"]
    assert "no submission" not in result["heuristics"][0]["reason"]


def test_multiturn_samples_advance_independently(monkeypatch):
    """A slow sample must not stall the others: sample-1 finishes all its turns while
    sample-2 is still crawling, and both trajectories stay complete."""
    order: list[str] = []

    async def _fake_remote(_client, request, _claimed):
        run_id = request.run_id
        if ":s2-" in run_id:
            await asyncio.sleep(0.4)
        order.append(run_id.split(":", 1)[1])
        return {
            "state": "succeeded",
            "responses": ["```bash\nls\n```"],
            "heuristics": [{"passed": True, "reason": ""}],
        }

    async def _noop(*_args, **_kwargs):
        return None

    request = SanityRunRequest(
        run_id="run",
        model_uri="model",
        digest="digest",
        prompts=["p1", "p2"],
        sample_ids=["sample-1", "sample-2"],
        prompt_messages=[
            [{"role": "user", "content": "task one"}],
            [{"role": "user", "content": "task two"}],
        ],
        assistant_turns=4,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)
    monkeypatch.setattr(D, "_append_observations", _noop)
    monkeypatch.setattr(D, "_inject_microtasks", _noop)
    monkeypatch.setattr(D, "_run_chain_checks", lambda *_a, **_k: None)
    monkeypatch.setattr(D, "run_tail_check", _noop)

    asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )

    assert order.count("s1-turn-4") == 1 and order.count("s2-turn-4") == 1
    # under the old lock-step barrier every sample finished turn 3 before anyone ran turn 4;
    # decoupled, the fast sample's last turn lands while the slow one is still early
    assert order.index("s1-turn-4") < order.index("s2-turn-3")


def test_multiturn_veto_halts_other_samples(monkeypatch):
    """When one sample fails a heuristic in veto mode, the others stop at their next turn."""
    calls: list[str] = []

    async def _fake_remote(_client, request, _claimed):
        calls.append(request.run_id.split(":", 1)[1])
        if request.run_id.split(":", 1)[1].startswith("s1-"):
            return {
                "state": "succeeded",
                "responses": ["no command here at all"],
                "heuristics": [{"passed": True, "reason": ""}],
            }
        await asyncio.sleep(0.02)
        return {
            "state": "succeeded",
            "responses": ["```bash\nls\n```"],
            "heuristics": [{"passed": True, "reason": ""}],
        }

    async def _noop(*_args, **_kwargs):
        return None

    request = SanityRunRequest(
        run_id="run",
        model_uri="model",
        digest="digest",
        prompts=["p1", "p2"],
        sample_ids=["sample-1", "sample-2"],
        prompt_messages=[
            [{"role": "user", "content": "task one"}],
            [{"role": "user", "content": "task two"}],
        ],
        assistant_turns=32,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)
    monkeypatch.setattr(D, "_append_observations", _noop)
    monkeypatch.setattr(D, "_inject_microtasks", _noop)

    result = asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )

    # sample-1 burns its bad-turn budget and vetoes; sample-2 never reaches turn 32
    assert "3 consecutive turns" in result["heuristics"][0]["reason"]
    assert not any(c == "s2-turn-32" for c in calls)


def test_teardown_is_skipped_while_runs_are_active(monkeypatch):
    """A stale teardown (dispatcher retries it with delays) must never kill vLLM under the
    next attempt's live generations."""
    import sanity_remote.api as api

    torn: list[bool] = []

    async def _fake_teardown():
        torn.append(True)

    monkeypatch.setattr(api, "teardown", _fake_teardown)
    monkeypatch.setattr(api, "store", SanityRunStore())

    req = SanityRunRequest(run_id="live-1", model_uri="m", digest="d", prompts=["p"])
    api.store.start(req)
    api.store.mark_worker_started("live-1")

    result = asyncio.run(api.teardown_worker())
    assert result == {"state": "skipped_active_runs"}
    assert torn == []

    api.store.get("live-1").succeed(responses=["x"], heuristics=[{"passed": True, "reason": ""}])
    result = asyncio.run(api.teardown_worker())
    assert result == {"state": "ok"}
    assert torn == [True]


# --- grounding ------------------------------------------------------------------------------


def _sim_settings():
    """Only the fields _simulate_observation_uncached reads."""
    return SimpleNamespace(
        evaluator_model="evaluator",
        simulation_model="simulator",
        simulation_max_tokens=4096,
        simulation_providers="",
        evaluator_providers="",
    )


def _grounding_state(command: str = "grep -n 'def clear' pkg/core.py"):
    return SimpleNamespace(
        sample_id="shard.parquet:1:1",
        prompt="prompt",
        messages=[{"role": "user", "content": "task"}],
        turns=[],
    ), f"THOUGHT: look\n```bash\n{command}\n```"


class _StubRepoContext:
    """Stands in for RepoContextClient and counts calls."""

    def __init__(self, grounding):
        self.grounding = grounding
        self.calls = 0

    async def context_for(self, sample_id, assistant_output, messages=None):
        self.calls += 1
        return self.grounding

    async def aclose(self):
        return None


def _no_llm(*_args, **_kwargs):
    raise AssertionError("the simulator was called when it should not have been")


def test_exact_grounding_returns_unsimulated(monkeypatch):
    """The feature: a command resolved against the real snapshot reaches the model verbatim with
    no LLM call. If this stops holding, pre-eval is silently back to inventing observations."""
    state, assistant = _grounding_state()
    repo = _StubRepoContext(D.Grounding(None, "12:def clear(self):", 0, "state-1"))
    monkeypatch.setattr(D, "detect_format", lambda *_a, **_k: D.RETURNCODE)

    observation = asyncio.run(
        D._simulate_observation_uncached(
            client=SimpleNamespace(complete=_no_llm),
            settings=_sim_settings(),
            eval_run_id="run",
            state=state,
            assistant_output=assistant,
            repo_context=repo,
        )
    )
    assert "12:def clear(self):" in observation
    assert "<returncode>0</returncode>" in observation


def test_resolved_context_reaches_the_simulator_prompt(monkeypatch):
    """The other half of the feature: when the answer cannot be computed outright, the real repo
    context must still reach the simulator. Dropping it reverts tier 2 to blind improvisation."""
    state, assistant = _grounding_state()
    block = (
        f"{D.COMPUTED_BLOCK_MARKER} this search was executed against the repository\n12:def clear"
    )
    repo = _StubRepoContext(D.Grounding(block, None, None, "state-1"))
    monkeypatch.setattr(D, "detect_format", lambda *_a, **_k: D.RETURNCODE)
    seen: dict[str, str] = {}

    async def _complete(**kwargs):
        seen["system"] = kwargs["messages"][0]["content"]
        seen["user"] = kwargs["messages"][1]["content"]
        return SimpleNamespace(
            raw="<returncode>0</returncode>\n<output>\n12:def clear\n</output>", error=None
        )

    asyncio.run(
        D._simulate_observation_uncached(
            client=SimpleNamespace(complete=_complete),
            settings=_sim_settings(),
            eval_run_id="run",
            state=state,
            assistant_output=assistant,
            repo_context=repo,
        )
    )
    assert block in seen["system"]
    assert seen["user"].startswith("$ grep -n")


def test_memo_queries_repo_context_once_for_a_repeated_command(monkeypatch):
    """Pre-eval memoises on its own mutation fingerprint and grounding sits behind that memo. If
    grounding ever moved in front of it, a looping model would get inconsistent answers to the
    same command - the contradictions this work exists to remove."""
    state, assistant = _grounding_state()
    state.observation_memo = ObservationMemo()
    repo = _StubRepoContext(D.Grounding(None, "12:def clear(self):", 0, "state-1"))
    monkeypatch.setattr(D, "detect_format", lambda *_a, **_k: D.RETURNCODE)

    async def _twice():
        for _ in range(2):
            await D._simulate_observation(
                client=SimpleNamespace(complete=_no_llm),
                settings=_sim_settings(),
                eval_run_id="run",
                state=state,
                assistant_output=assistant,
                repo_context=repo,
            )

    asyncio.run(_twice())
    assert repo.calls == 1


def test_prefetch_is_awaited_before_the_first_turn(monkeypatch):
    """Snapshots download synchronously inside /repo-context, which the per-call budget cannot
    absorb, so turns must not start until the warm-up returns. It also must stay out of
    _build_request, which runs inside the claim transaction's advisory lock."""
    order: list[str] = []

    async def _warm(_settings, sample_ids):
        order.append(f"prefetch:{','.join(sample_ids)}")

    async def _fake_remote(_client, _request, _claimed):
        order.append("turn")
        return {
            "state": "succeeded",
            "responses": ["```bash\nls\n```"],
            "heuristics": [{"passed": True, "reason": ""}],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(D, "_warm_repo_context", _warm)
    monkeypatch.setattr(
        D, "RepoContextClient", lambda _s: _StubRepoContext(D.Grounding(None, None, None, ""))
    )
    monkeypatch.setattr(
        D, "get_judge_settings", lambda: SimpleNamespace(repo_context_url="http://rc")
    )
    monkeypatch.setattr(D, "_append_observations", _noop)
    monkeypatch.setattr(D, "_inject_microtasks", _noop)

    request = SanityRunRequest(
        run_id="run",
        model_uri="m",
        digest="d",
        prompts=["p"],
        sample_ids=["s-a"],
        assistant_turns=1,
    )
    dispatcher = D.SanityDispatcher(settings=SanitySettings(), repository=_FakeRepo())
    monkeypatch.setattr(dispatcher, "_run_remote_request", _fake_remote)

    asyncio.run(
        dispatcher._run_multiturn(
            SimpleNamespace(),
            SimpleNamespace(request=request, attempt_id=uuid4(), submission_id=uuid4()),
        )
    )
    assert order == ["prefetch:s-a", "turn"]


def test_warm_repo_context_never_raises():
    """Grounding is optional. If the warm-up ever propagated, an unreachable repo-context service
    would fail every attempt as INFRA_FAULT instead of simulating ungrounded."""

    async def _run(url):
        await D._warm_repo_context(SimpleNamespace(repo_context_url=url), ["s"])

    asyncio.run(_run(""))
    asyncio.run(_run("http://127.0.0.1:1"))
