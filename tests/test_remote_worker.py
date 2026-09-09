from __future__ import annotations

import json
import types
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_config import RemoteSettings
from albedo_eval_service.modelstore.canonical_model_config import canonical_max_model_len
from albedo_eval_service.modelstore.resolver import ResolvedModel
from albedo_eval_service.remote.dataset import EvalSample
from albedo_eval_service.remote.generation import GenerationResult, format_scored_trajectory
from albedo_eval_service.remote.state import RemoteRun
from albedo_eval_service.remote.worker import (
    ObservationResult,
    RemoteEvalWorker,
    _completion_observation,
    _generate_retrying_bad_turns,
    _merge_trajectory_results,
    _missing_command_observation,
    _next_turn_samples,
)
from albedo_eval_service.scoring.scoring_client import ScoringResult
from albedo_eval_service.shared.models import (
    Challenger,
    DatasetConfig,
    EvalRequest,
    PreviousKing,
    ScoringConfig,
)
from albedo_eval_service.shared.observation_format import (
    MAX_CONSECUTIVE_BAD_TURNS,
    TRUNCATION_SENTINEL,
    abandonment_notice,
    is_abandoned,
)


class _Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(message["content"] for message in messages) + " assistant:"


class RecordingGenerator:
    def __init__(self, *, side: str, calls: list[dict[str, object]]):
        self.side = side
        self.calls = calls

    def generate(self, samples):
        self.calls.append(
            {"side": self.side, "sample_ids": [sample.sample_id for sample in samples]}
        )
        suffix = " challenger output" if self.side == "challenger" else " king"
        # a real turn always carries a bash block; without one the worker now short-circuits to a
        # missing-command observation instead of asking the simulator
        return [
            GenerationResult(
                sample_id=sample.sample_id,
                text=f"{sample.sample_id}{suffix}\n```bash\nls\n```",
            )
            for sample in samples
        ]

    def close(self):
        self.calls.append({"side": self.side, "closed": True})


def _write_dataset(root):
    shard_dir = root / "data"
    shard_dir.mkdir()
    rows = []
    for idx in range(2):
        rows.append(
            json.dumps(
                [
                    {"role": "user", "content": f"Task {idx}"},
                    {"role": "assistant", "content": f"Answer {idx}"},
                ]
            )
        )
    pq.write_table(pa.table({"messages": rows}), shard_dir / "train-00000.parquet")


def _request():
    return EvalRequest(
        eval_run_id=uuid4(),
        submission_id=uuid4(),
        challenger=Challenger(model_uri="s3-or-hippius-uri/challenger", model_hash="sha256:chal"),
        previous_king=PreviousKing(
            model_uri="s3-or-hippius-uri/king", model_hash="sha256:king", king_version=7
        ),
        dataset=DatasetConfig(
            version="AlienKevin/SWE-ZERO-12M-trajectories",
            manifest_uri="s3://albedo-artifacts/datasets/swe-zero/manifest.json",
            manifest_hash="982a92bd85d122d287b15f2ddb4e2050b9e345fb3921aa9a63382c7af022bd7f",
            sample_count=2,
            sample_seed="0xabc",
            sampling_algo="swe-zero-manifest-sample-v1",
            generation_batch_size=1,
            scoring_batch_size=1,
            sample_ids=["data/train-00000.parquet:0:0", "data/train-00000.parquet:1:0"],
        ),
        scoring=ScoringConfig(judge_config_hash="sha256:judge"),
        artifact_prefix="s3://albedo-artifacts/submissions/sub/eval/run",
    )


def test_scored_trajectory_marks_only_candidate_outputs():
    text = format_scored_trajectory(
        [
            {"role": "user", "content": "Fix it"},
            {"role": "assistant", "content": "first", "score_target": True},
            {"role": "user", "content": "Observation: ok", "environment_observation": True},
            {"role": "assistant", "content": "second", "score_target": True},
            {"role": "user", "content": "Observation: still ok", "environment_observation": True},
            {"role": "assistant", "content": "third", "score_target": True},
        ]
    )

    assert "Score ONLY CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT 3" in text
    assert "CONTEXT USER (do not score)" in text
    assert "CANDIDATE OUTPUT 1" in text
    assert "ENVIRONMENT OBSERVATION (context only, do not score)" in text
    assert "CANDIDATE OUTPUT 2" in text
    assert "CANDIDATE OUTPUT 3" in text


def test_remote_worker_loads_parquet_and_runs_paired_generation(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
    calls: list[dict[str, object]] = []

    def factory(side, gpu_ids, model):
        calls.append({"side": side, "gpu_ids": gpu_ids, "model": model})
        return RecordingGenerator(side=side, calls=calls)

    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        generation_backend="vllm",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
        trajectory_assistant_turns=2,
        rollouts_per_sample=1,
    )

    RemoteEvalWorker(settings, generator_factory=factory).execute(run)

    assert run.state == "succeeded"
    verdict = run.final_verdict()
    assert verdict is not None
    assert set(verdict["artifacts"]) == {
        "generated_samples",
        "progress",
        "remote_logs",
        "request",
        "scoring_results",
        "verdict",
    }
    assert verdict["artifact_metadata"]["generated_samples"]["sha256"].startswith("sha256:")
    assert verdict["valid_turns"] == 2
    assert verdict["gpu_topology"]["previous_king"] == ["0", "1", "2", "3"]
    assert verdict["gpu_topology"]["challenger"] == ["4", "5", "6", "7"]
    generation_events = [event for event in run.events if event["type"] == "generation_batch_done"]
    scoring_events = [event for event in run.events if event["type"] == "scoring_batch_done"]
    assert [event["batch_id"] for event in generation_events] == ["gen-0001", "gen-0002"]
    assert [event["batch_id"] for event in scoring_events] == ["score-0001", "score-0002"]
    assert {call["side"] for call in calls if "gpu_ids" in call} == {"previous_king", "challenger"}
    generate_calls = [call for call in calls if "sample_ids" in call]
    # horizons 12 and 16, one request per trajectory turn
    assert [call["side"] for call in generate_calls].count("previous_king") == 28
    assert [call["side"] for call in generate_calls].count("challenger") == 28
    assert all(len(call["sample_ids"]) == 1 for call in generate_calls)
    assert [call["side"] for call in calls if call.get("closed")].count("previous_king") == 1
    assert [call["side"] for call in calls if call.get("closed")].count("challenger") == 1


class RecordingModelResolver:
    def __init__(self, calls: list[object]):
        self.calls = calls

    def resolve(self, model_ref: str) -> ResolvedModel:
        self.calls.append(f"resolve:{model_ref}")
        return ResolvedModel(model_ref, model_ref, "test", True, 0, 0)


class RecordingScorer:
    def __init__(self, calls: list[object]):
        self.calls = calls

    def start_category_prep(self, *, request, samples):
        self.calls.append("category_prep")
        return "prep-1"

    def simulate_observation(self, *, request, sample, assistant_output):
        self.calls.append(f"simulate:{sample.sample_id}")
        return f"Observation: saw {assistant_output[-20:]}"

    def score(self, *, request, samples, king_results, challenger_results, category_prep_id=None):
        self.calls.append(f"score:{category_prep_id}")
        records = [
            {
                "sample_id": sample.sample_id,
                "order": ["previous_king", "challenger"],
                "judge_results": [],
                "judge_scores": [],
                "sample_score": 0.5,
                "scored": True,
                "scoring_mode": "test",
            }
            for sample in samples
        ]
        return ScoringResult(
            records=records,
            summary={
                "state": "succeeded",
                "score_challenger": 0.5,
                "score_king": 0.5,
                "challenger_won": False,
                "valid_turns": len(records),
                "total_turns": len(records),
                "judge_errors": 0,
                "scored_sample_count": len(records),
                "scoring_mode": "test",
            },
        )


def test_remote_worker_starts_category_prep_before_model_resolution(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
    calls: list[object] = []

    def factory(side, gpu_ids, model):
        calls.append({"side": side, "model": model})
        return RecordingGenerator(side=side, calls=calls)

    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        generation_backend="vllm",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
    )

    RemoteEvalWorker(
        settings,
        generator_factory=factory,
        model_resolver=RecordingModelResolver(calls),
        scorer=RecordingScorer(calls),
    ).execute(run)

    assert calls.index("category_prep") < calls.index("resolve:s3-or-hippius-uri/king")
    assert any(str(call).startswith("simulate:") for call in calls)


def test_submit_echo_stops_future_trajectory_turns(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages", lambda messages, **_kwargs: "next"
    )
    sample = types.SimpleNamespace(
        sample_id="sample-1",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
        submit_marker="",
        submit_command="",
    )
    observation = ObservationResult(
        "sample-1", "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )

    assert (
        _next_turn_samples(
            [sample],
            [
                GenerationResult(
                    "sample-1", "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                )
            ],
            {("challenger", "sample-1"): observation},
            side="challenger",
        )
        == []
    )

    merged = _merge_trajectory_results(
        [sample],
        [
            [
                GenerationResult(
                    "sample-1", "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                )
            ],
            [],
        ],
        [{("challenger", "sample-1"): observation}],
        side="challenger",
        token_limit=16384,
    )

    assert merged[0].error is None
    assert "CANDIDATE OUTPUT 2" not in merged[0].text
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in merged[0].text


def _trajectory_sample(sample_id: str = "sample-1"):
    return types.SimpleNamespace(
        sample_id=sample_id,
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )


def test_truncated_response_ends_trajectory_and_stays_valid(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages", lambda messages, **_kwargs: "next"
    )
    sample = _trajectory_sample()
    oversized = "x" * 200
    truncated = GenerationResult("sample-1", oversized, truncated=True)
    observation = ObservationResult("sample-1", "")

    assert (
        _next_turn_samples(
            [sample],
            [truncated],
            {("challenger", "sample-1"): observation},
            side="challenger",
        )
        == []
    )

    merged = _merge_trajectory_results(
        [sample],
        [[truncated], []],
        [{("challenger", "sample-1"): observation}],
        side="challenger",
        token_limit=16384,
    )

    assert merged[0].error is None
    assert merged[0].truncated is True
    assert TRUNCATION_SENTINEL in merged[0].text
    assert "16384" in merged[0].text
    assert oversized not in merged[0].text
    assert "CANDIDATE OUTPUT 2" not in merged[0].text


class _ScriptedGenerator:
    """Returns the scripted response per call; records every prompt it was asked."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompts: list[list[str]] = []

    def generate(self, samples):
        self.prompts.append([sample.prompt for sample in samples])
        text = self.responses.pop(0)
        return [GenerationResult(sample.sample_id, text) for sample in samples]


def _eval_sample(sample_id: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sample_id, prompt="Task", messages=[{"role": "user", "content": "Task"}]
    )


def test_bad_turn_is_retried_with_accumulated_feedback(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages",
        lambda messages, **_kwargs: "\n".join(m["content"] for m in messages),
    )
    good = "THOUGHT: ok\n\n```bash\nls\n```"
    generator = _ScriptedGenerator(["no command here", "", good])

    results = _generate_retrying_bad_turns(generator, [_eval_sample()])

    assert results[0].text == good
    assert len(results[0].retry_feedbacks) == 2
    assert all("Format error" in feedback for feedback in results[0].retry_feedbacks)
    # the second re-ask carries BOTH earlier feedbacks, like the bench conversation would
    assert generator.prompts[2][0].count("Format error") == 2


def test_turn_still_unusable_after_all_attempts_becomes_abandonment(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages",
        lambda messages, **_kwargs: "\n".join(m["content"] for m in messages),
    )
    generator = _ScriptedGenerator(["prose one", "prose two", "prose three"])

    results = _generate_retrying_bad_turns(generator, [_eval_sample()])

    assert len(generator.prompts) == MAX_CONSECUTIVE_BAD_TURNS
    assert is_abandoned(results[0].text)
    assert "no bash command" in results[0].text
    assert results[0].error is None
    assert len(results[0].retry_feedbacks) == MAX_CONSECUTIVE_BAD_TURNS - 1


def test_abandoned_turn_ends_trajectory_and_hides_feedback_from_judges(monkeypatch):
    monkeypatch.setattr(
        "albedo_eval_service.remote.worker.format_messages", lambda messages, **_kwargs: "next"
    )
    sample = _trajectory_sample()
    abandoned = GenerationResult(
        "sample-1",
        abandonment_notice("no bash command found in the response"),
        retry_feedbacks=("FB-ONE", "FB-TWO"),
    )
    observation = ObservationResult("sample-1", "")

    assert (
        _next_turn_samples(
            [sample],
            [abandoned],
            {("challenger", "sample-1"): observation},
            side="challenger",
        )
        == []
    )

    merged = _merge_trajectory_results(
        [sample],
        [[abandoned], []],
        [{("challenger", "sample-1"): observation}],
        side="challenger",
        token_limit=16384,
    )

    assert merged[0].error is None
    assert is_abandoned(merged[0].text)
    assert "CANDIDATE OUTPUT 2" not in merged[0].text
    assert "FB-ONE" not in merged[0].text
    feedback_turns = [t for t in merged[0].turns if t.get("retry_feedback")]
    assert [t["content"] for t in feedback_turns] == ["FB-ONE", "FB-TWO"]


def test_retry_feedback_persists_into_next_turn_context(monkeypatch):
    captured: list[list[dict[str, str]]] = []

    def _fake_format(messages, **_kwargs):
        captured.append(messages)
        return "next"

    monkeypatch.setattr("albedo_eval_service.remote.worker.format_messages", _fake_format)
    sample = _eval_sample("sample-1")
    recovered = GenerationResult("sample-1", "```bash\nls\n```", retry_feedbacks=("FB-ONE",))
    observation = ObservationResult("sample-1", "ok")

    (next_sample,) = _next_turn_samples(
        [sample],
        [recovered],
        {("challenger", "sample-1"): observation},
        side="challenger",
    )

    roles_and_contents = [(m["role"], m["content"]) for m in next_sample.messages]
    assert roles_and_contents == [
        ("user", "Task"),
        ("user", "FB-ONE"),
        ("assistant", "```bash\nls\n```"),
        ("user", "ok"),
    ]


def _pairs_worker(scorer):
    return RemoteEvalWorker(
        RemoteSettings(
            scoring_backend="mock", upload_artifacts=False, resolve_model_artifacts=False
        ),
        scorer=scorer,
    )


def test_score_pairs_retries_when_our_side_lost_the_pairs():
    class NeverCalled:
        def score(self, **_kwargs):
            raise AssertionError("scorer must not run when too few pairs are valid")

        def simulate_observation(self, **_kwargs):
            return ""

    samples = [_trajectory_sample(f"s{index}") for index in range(10)]
    healthy = [GenerationResult(sample.sample_id, "out") for sample in samples]
    # the king box failed: the miner's model produced every trajectory
    king_down = [
        GenerationResult(s.sample_id, "", "vllm timed out") for s in samples[:9]
    ] + healthy[9:]
    # the bridge dropped: the challenger's errors are ours, not the miner's
    bridge_down = [
        GenerationResult(s.sample_id, "", "ScoreBridgeUnavailable: score bridge disconnected")
        for s in samples[:9]
    ] + healthy[9:]
    for king_results, challenger_results in ((king_down, healthy), (healthy, bridge_down)):
        summary = _pairs_worker(NeverCalled())._score_pairs(
            request=_request(),
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
        )["summary"]
        assert summary["fault_class"] == "REMOTE_EVAL_FAULT"
        assert summary["fault_code"] == "insufficient_valid_samples"
        assert summary["retryable"] is True
        assert summary["valid_turns"] == 1


def test_score_pairs_is_terminal_when_too_few_valid_pairs():
    class NeverCalled:
        def score(self, **_kwargs):
            raise AssertionError("scorer must not run when too few pairs are valid")

        def simulate_observation(self, **_kwargs):
            return ""

    samples = [_trajectory_sample(f"s{index}") for index in range(10)]
    king_results = [GenerationResult(sample.sample_id, "king") for sample in samples]
    challenger_results = [
        GenerationResult(sample.sample_id, "", "vllm timed out") for sample in samples[:9]
    ] + [GenerationResult("s9", "challenger")]

    summary = _pairs_worker(NeverCalled())._score_pairs(
        request=_request(),
        samples=samples,
        king_results=king_results,
        challenger_results=challenger_results,
    )["summary"]

    assert summary["state"] == "failed"
    assert summary["fault_class"] == "MINER_FAULT"
    assert summary["fault_code"] == "insufficient_valid_samples"
    assert summary["retryable"] is False
    assert summary["valid_turns"] == 1
    assert summary["total_turns"] == 10


def test_score_pairs_counts_truncated_pairs_as_valid():
    scored = {}

    class Recording:
        def score(
            self, *, request, samples, king_results, challenger_results, category_prep_id=None
        ):
            scored["samples"] = len(samples)
            return ScoringResult(records=[], summary={"state": "succeeded"})

        def simulate_observation(self, **_kwargs):
            return ""

    samples = [_trajectory_sample(f"s{index}") for index in range(10)]
    king_results = [GenerationResult(sample.sample_id, "king") for sample in samples]
    challenger_results = [
        GenerationResult(sample.sample_id, "notice", truncated=True) for sample in samples
    ]

    result = _pairs_worker(Recording())._score_pairs(
        request=_request(),
        samples=samples,
        king_results=king_results,
        challenger_results=challenger_results,
    )

    assert scored["samples"] == 10
    assert result["summary"]["state"] == "succeeded"


def test_submit_echo_bypasses_observation_simulator(tmp_path):
    class FailingScorer:
        def simulate_observation(self, **_kwargs):
            raise AssertionError("simulator should not run for submit echo")

    sample = types.SimpleNamespace(
        sample_id="mini-coder/data/train-00000.parquet:1:0",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
        submit_marker="",
        submit_command="",
    )
    result = GenerationResult(
        sample.sample_id, "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
    )
    worker = RemoteEvalWorker(
        RemoteSettings(dataset_root=str(tmp_path), scoring_backend="mock"),
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
        scorer=FailingScorer(),
    )

    observations = worker._simulate_observations(
        request=_request(),
        samples_by_side={"challenger": [sample]},
        results_by_side={"challenger": [result]},
    )

    assert observations[("challenger", sample.sample_id)].observation == _completion_observation(
        sample
    )


def test_remote_worker_rejects_overlapping_gpu_groups(tmp_path):
    _write_dataset(tmp_path)
    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        previous_king_gpu_ids="0,1,2,3",
        challenger_gpu_ids="3,4,5,6",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
    )

    RemoteEvalWorker(
        settings,
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
    ).execute(run)

    verdict = run.final_verdict()
    assert run.state == "failed"
    assert verdict is not None
    assert verdict["fault_code"] == "remote_worker_failed"
    assert "GPU groups overlap" in verdict["fault_message"]


def test_vllm_generator_uses_canonical_max_model_len_even_when_env_is_lower(tmp_path):
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        upload_artifacts=False,
        max_model_len=4096,
        scoring_backend="mock",
    )

    worker = RemoteEvalWorker(settings, generator_factory=None)
    generator = worker._vllm_generator("challenger", ["4", "5", "6", "7"], "/models/challenger")

    assert generator.max_model_len == canonical_max_model_len()
    assert generator.max_new_tokens == settings.max_new_tokens
    assert generator.port == settings.challenger_vllm_port
    assert (
        worker._vllm_generator("previous_king", ["0"], "/m").port
        == settings.previous_king_vllm_port
    )


def test_rollouts_share_the_row_horizon():
    from albedo_eval_service.remote import worker as W

    samples = [_eval_sample("a"), _eval_sample("b")]
    base = W.assign_horizons(samples)
    expanded = W._rollouts(samples, 2)

    assert [sample.sample_id for sample in expanded] == ["a#r1", "b#r1", "a#r2", "b#r2"]
    assert W._rollout_horizons(expanded) == {
        "a#r1": base["a"],
        "a#r2": base["a"],
        "b#r1": base["b"],
        "b#r2": base["b"],
    }
    assert W._rollouts(samples, 1) is samples


def test_rollouts_double_the_trajectories_of_every_row(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
    calls: list[dict[str, object]] = []
    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
        rollouts_per_sample=2,
    )

    RemoteEvalWorker(
        settings,
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=calls),
    ).execute(run)

    verdict = run.final_verdict()
    assert run.state == "succeeded"
    assert verdict["valid_turns"] == 4
    assert verdict["generated_sample_count"] == 4
    started = next(event for event in run.events if event["type"] == "generation_started")
    assert started["sample_count"] == 2  # the dataset draw, not the trajectory count
    king_ids = [
        call["sample_ids"][0]
        for call in calls
        if call.get("side") == "previous_king" and "sample_ids" in call
    ]
    assert sorted(set(king_ids)) == [
        "data/train-00000.parquet:0:0#r1",
        "data/train-00000.parquet:0:0#r2",
        "data/train-00000.parquet:1:0#r1",
        "data/train-00000.parquet:1:0#r2",
    ]
    assert len(king_ids) == 2 * (12 + 16)


def test_prefetch_repo_context_fires_only_when_configured(monkeypatch):
    import threading

    import albedo_eval_service.remote.worker as remote_worker_module
    from albedo_eval_service.remote.dataset import EvalSample

    recorded: dict[str, object] = {}
    posted = threading.Event()

    def fake_post(url, json=None, timeout=None):
        recorded["url"] = url
        recorded["json"] = json
        posted.set()

    monkeypatch.setattr(remote_worker_module.httpx, "post", fake_post)
    samples = [EvalSample(sample_id="data/train-00000.parquet:0:0", prompt="p")]

    enabled = RemoteEvalWorker(
        RemoteSettings(
            repo_context_url="http://127.0.0.1:8093/",
            upload_artifacts=False,
            scoring_backend="mock",
        )
    )
    enabled._prefetch_repo_context(_request(), samples)
    assert posted.wait(2.0)
    assert recorded["url"] == "http://127.0.0.1:8093/prefetch"
    assert recorded["json"] == {"sample_ids": ["data/train-00000.parquet:0:0"]}

    posted.clear()
    disabled = RemoteEvalWorker(RemoteSettings(upload_artifacts=False, scoring_backend="mock"))
    disabled._prefetch_repo_context(_request(), samples)
    assert not posted.wait(0.2)


def test_missing_bash_command_bypasses_observation_simulator(tmp_path):
    """A turn with no bash block must get a real command error, never a simulated observation.

    Previously _command_only() fell back to the whole assistant message, so the simulator
    invented a filesystem for models that emit a JSON tool call instead of a bash fence.
    """

    class FailingScorer:
        def simulate_observation(self, **_kwargs):
            raise AssertionError("simulator must not run when there is no bash command")

    sample = types.SimpleNamespace(
        sample_id="mini-coder-rs/data/train-00000.parquet:1156:2",
        submit_marker="",
        submit_command="",
        prompt="Task",
        target=None,
        messages=[{"role": "user", "content": "Task"}],
    )
    result = GenerationResult(
        sample.sample_id,
        'THOUGHT: read the file\n{"command": "sed -n \'590,670p\' /testbed/src/naive/date/mod.rs"}',
    )
    worker = RemoteEvalWorker(
        RemoteSettings(dataset_root=str(tmp_path), scoring_backend="mock"),
        generator_factory=lambda side, gpu_ids, model: RecordingGenerator(side=side, calls=[]),
        scorer=FailingScorer(),
    )

    observations = worker._simulate_observations(
        request=_request(),
        samples_by_side={"challenger": [sample]},
        results_by_side={"challenger": [result]},
    )

    observation = observations[("challenger", sample.sample_id)].observation
    assert observation == _missing_command_observation(sample)
    assert "No bash command found" in observation
    assert "<returncode>2</returncode>" in observation


def test_eval_reasks_a_bad_turn_instead_of_ending_the_trajectory():
    """Scoring mirrors the benchmark: a cut-off or malformed turn is re-asked, not fatal."""
    from albedo_eval_service.remote import worker as W
    from albedo_eval_service.remote.dataset import EvalSample
    from albedo_eval_service.remote.generation import GenerationResult
    from albedo_eval_service.shared.observation_format import truncation_notice

    good = "THOUGHT: look\n\n```bash\nls -la\n```"
    bad_by_attempt = [
        [GenerationResult("s1", truncation_notice(4096), truncated=True)],
        [GenerationResult("s1", "")],
        [GenerationResult("s1", good)],
    ]

    class _Gen:
        def __init__(self):
            self.calls = []

        def generate(self, samples):
            self.calls.append(samples[0].messages[-1]["content"] if samples[0].messages else "")
            return bad_by_attempt[len(self.calls) - 1]

    gen = _Gen()
    sample = EvalSample(sample_id="s1", prompt="p", messages=[{"role": "user", "content": "task"}])
    out = W._generate_retrying_bad_turns(gen, [sample])

    assert out[0].text == good, "the usable third attempt must win"
    assert len(gen.calls) == 3
    assert "reached the output token limit" in gen.calls[1], "truncation -> be concise"
    assert "Format error" in gen.calls[2], "empty response -> format feedback"


def test_eval_gives_up_after_three_consecutive_bad_turns():
    from albedo_eval_service.remote import worker as W
    from albedo_eval_service.remote.dataset import EvalSample
    from albedo_eval_service.remote.generation import GenerationResult

    class _Gen:
        def __init__(self):
            self.calls = 0

        def generate(self, samples):
            self.calls += 1
            return [GenerationResult("s1", "")]

    gen = _Gen()
    sample = EvalSample(sample_id="s1", prompt="p", messages=[{"role": "user", "content": "t"}])
    out = W._generate_retrying_bad_turns(gen, [sample])
    assert gen.calls == W.MAX_CONSECUTIVE_BAD_TURNS
    assert is_abandoned(out[0].text)
    assert "empty response" in out[0].text


def test_eval_retry_fires_inside_the_real_turn_loop(tmp_path, monkeypatch):
    """End to end through the worker: a bad first turn is re-asked, the run still succeeds."""
    _write_dataset(tmp_path)
    monkeypatch.setattr(
        "albedo_eval_service.remote.dataset._load_tokenizer", lambda _path: _Tokenizer()
    )
    seen: list[str] = []

    class _FlakyGenerator:
        """Emits one unusable turn per sample, then behaves."""

        def __init__(self, side):
            self.side = side
            self.bad_done: set[str] = set()

        def generate(self, samples):
            out = []
            for sample in samples:
                last = (sample.messages or [{}])[-1].get("content", "")
                if "Format error" in last or "output token limit" in last:
                    seen.append(last[:40])
                if sample.sample_id in self.bad_done:
                    text = f"{sample.sample_id} ok\n```bash\nls\n```"
                else:
                    self.bad_done.add(sample.sample_id)
                    text = ""  # empty response -> must be re-asked, not fatal
                out.append(GenerationResult(sample_id=sample.sample_id, text=text))
            return out

        def close(self):
            return None

    settings = RemoteSettings(
        dataset_root=str(tmp_path),
        generation_backend="vllm",
        upload_artifacts=False,
        artifact_spool_dir=str(tmp_path / "artifacts"),
        scoring_backend="mock",
        trajectory_assistant_turns=2,
    )
    request = _request()
    run = RemoteRun(remote_run_id=str(request.eval_run_id), request=request, state="accepted")
    RemoteEvalWorker(
        settings, generator_factory=lambda side, gpu_ids, model: _FlakyGenerator(side)
    ).execute(run)

    assert run.state == "succeeded"
    assert seen, "the retry feedback must have reached the generator"
    assert all("Format error" in s for s in seen)
