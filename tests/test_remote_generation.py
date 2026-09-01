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
