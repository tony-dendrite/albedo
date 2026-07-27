"""Repo-context API: serves grounding blocks for the env simulator (runs on the eval host).

Judge-api POSTs {sample_id, assistant_output} per observation simulation; the response carries
either a repo-snapshot grounding block, a dataset-trajectory fallback block, or null (caller
keeps the ungrounded prompt)."""

from __future__ import annotations

import asyncio

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from loguru import logger
from pydantic import BaseModel

from .core import RepoContextService
from .settings import RepoContextSettings, get_repo_context_settings


class RepoContextRequest(BaseModel):
    sample_id: str
    assistant_output: str = ""


class RepoContextResponse(BaseModel):
    sample_id: str
    context: str | None
    kind: str  # "repo" | "trajectory" | "none"


class PrefetchRequest(BaseModel):
    sample_ids: list[str]


def create_app(
    settings: RepoContextSettings | None = None, service: RepoContextService | None = None
) -> FastAPI:
    settings = settings or get_repo_context_settings()
    service = service or RepoContextService(settings)
    app = FastAPI(title="Albedo Repo Context API")

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "cache_dir": settings.cache_dir,
            "manifest_configured": bool(settings.dataset_manifest_path),
            "dataset_root_configured": bool(settings.dataset_root),
            "github_token_set": bool(service._auth_headers()),
        }

    @app.post("/prefetch")
    async def prefetch(request: PrefetchRequest, background_tasks: BackgroundTasks) -> dict:
        # Called by the eval worker when model resolution starts: snapshots download in the
        # background alongside the model download; already-cached snapshots are skipped.
        background_tasks.add_task(service.prefetch, request.sample_ids)
        return {"accepted": len(request.sample_ids)}

    @app.post("/repo-context", response_model=RepoContextResponse)
    async def repo_context(request: RepoContextRequest) -> RepoContextResponse:
        try:
            result = await asyncio.to_thread(
                service.context_for, request.sample_id, request.assistant_output
            )
        except Exception as exc:  # noqa: BLE001 - grounding must never fail the caller
            logger.warning(
                "repo_context_endpoint_error sample_id={} error={}",
                request.sample_id, f"{type(exc).__name__}: {exc}",
            )
            return RepoContextResponse(sample_id=request.sample_id, context=None, kind="none")
        return RepoContextResponse(
            sample_id=request.sample_id, context=result.context, kind=result.kind
        )

    return app


def main() -> None:
    settings = get_repo_context_settings()
    if not settings.cache_dir:
        raise SystemExit(
            "ALBEDO_REPO_CONTEXT_CACHE_DIR is required: set the snapshot download directory"
        )
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
