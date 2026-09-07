from __future__ import annotations

from fastapi.testclient import TestClient

from albedo_config import RepoContextSettings
from repo_context_service.api import create_app
from repo_context_service.core import GroundingContext


class FakeService:
    def __init__(self, result: GroundingContext | Exception):
        self.result = result
        self.prefetched: list[str] | None = None

    def context_for(self, sample_id, assistant_output, messages=None):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def prefetch(self, sample_ids):
        self.prefetched = list(sample_ids)
        return {"samples": len(sample_ids), "instances": 0, "ready": 0}

    def _auth_headers(self):
        return {}

    def close(self):
        pass


def make_client(result, service: FakeService | None = None) -> TestClient:
    # every field this file asserts on is pinned rather than defaulted: `_env_file=None` stops
    # pydantic reading .env but not os.environ, and any test that touches a settings getter has
    # already copied the developer's .env into it via load_dotenv's setdefault
    settings = RepoContextSettings(
        _env_file=None,
        cache_dir="/tmp/unused",
        dataset_manifest_path="",
        dataset_root="",
        github_token="",
    )
    return TestClient(create_app(settings, service=service or FakeService(result)))


def test_repo_context_happy_path():
    client = make_client(GroundingContext(context="BLOCK", kind="repo"))
    response = client.post("/repo-context", json={"sample_id": "swe-zero/data/train-0.parquet:0:0"})
    assert response.status_code == 200
    assert response.json() == {
        "sample_id": "swe-zero/data/train-0.parquet:0:0",
        "context": "BLOCK",
        "kind": "repo",
        "exact_output": None,
        "exact_returncode": None,
        "state": "",
    }


def test_repo_context_returns_none_kind_on_failure():
    client = make_client(RuntimeError("boom"))
    response = client.post("/repo-context", json={"sample_id": "x", "assistant_output": "y"})
    assert response.status_code == 200
    assert response.json() == {
        "sample_id": "x",
        "context": None,
        "kind": "none",
        "exact_output": None,
        "exact_returncode": None,
        "state": "",
    }


def test_prefetch_endpoint_accepts_and_runs_in_background():
    service = FakeService(GroundingContext(context=None, kind="none"))
    client = make_client(None, service=service)
    response = client.post("/prefetch", json={"sample_ids": ["a:0:0", "b:1:0"]})
    assert response.status_code == 200
    assert response.json() == {"accepted": 2}
    assert service.prefetched == ["a:0:0", "b:1:0"]


def test_healthz_reports_configuration():
    client = make_client(GroundingContext(context=None, kind="none"))
    payload = client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["cache_dir"] == "/tmp/unused"
    assert payload["manifest_configured"] is False
    assert payload["github_token_set"] is False


def test_prefetch_wait_blocks_and_returns_the_summary():
    """Pre-eval must not start driving turns on a cold snapshot cache: a cold `/repo-context`
    cannot finish inside its 20s budget, so the caller needs a completion signal."""
    import threading

    started = threading.Event()
    released = threading.Event()

    class SlowService(FakeService):
        def prefetch(self, sample_ids):
            started.set()
            released.wait(timeout=5)
            self.prefetched = list(sample_ids)
            return {"samples": len(sample_ids), "instances": 1, "ready": 1}

    service = SlowService(GroundingContext(context=None, kind="none"))
    client = TestClient(create_app(RepoContextSettings(cache_dir="/tmp/x"), service))

    result: dict = {}

    def call():
        result["body"] = client.post(
            "/prefetch", json={"sample_ids": ["s:1:1"], "wait": True}
        ).json()

    caller = threading.Thread(target=call)
    caller.start()
    assert started.wait(timeout=5), "prefetch was never invoked"
    assert "body" not in result, "wait=true returned before prefetch finished"
    released.set()
    caller.join(timeout=5)

    assert result["body"] == {"samples": 1, "instances": 1, "ready": 1}
    assert service.prefetched == ["s:1:1"]
