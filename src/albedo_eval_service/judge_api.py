from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from .judge_config import JudgeSettings, get_judge_settings
from .observation_format import (
    CommandContract,
    command_contract,
    contract_violation,
    detect_format,
    empty_output,
    first_bash_block,
    format_block,
    has_content,
    is_truncated,
    missing_command_output,
    repair_output,
    repair_to_contract,
    requires_output,
    valid_output,
    wrap,
)
from .judge_core import (
    JUDGE_MODELS,
    aggregate_scores,
    answer_schema,
    build_judge_messages,
    build_question_messages,
    apply_measurement_gate,
    candidate_turn_texts_from_merged,
    enforce_question_labels,
    filter_reference_leaks,
    format_reference_trajectory,
    trajectory_made_edit,
    judge_yes_rate,
    parse_answers,
    parse_questions,
    question_floor,
    question_schema,
    response_score,
)
from .judge_openrouter import OpenRouterJudgeClient
from .notifications import EvalErrorNotification, notify_eval_error


class QuestionPrepSample(BaseModel):
    sample_id: str
    prompt: str
    sample_index: int = 0
    messages: list[dict[str, str]] | None = None
    assistant_turns: int = 0


class QuestionPrepRequest(BaseModel):
    eval_run_id: str
    batch_id: str = "category-prep"
    samples: list[QuestionPrepSample]
    total_sample_count: int


class QuestionPrepResponse(BaseModel):
    eval_run_id: str
    category_prep_id: str
    accepted_sample_count: int


class JudgeSample(BaseModel):
    sample_id: str
    prompt: str
    previous_king_output: str
    challenger_output: str
    sample_index: int = 0
    messages: list[dict[str, str]] | None = None
    assistant_turns: int = 0


class ScoreBatchRequest(BaseModel):
    eval_run_id: str
    batch_id: str
    samples: list[JudgeSample]
    total_sample_count: int
    judge_models: list[str] = Field(default_factory=lambda: list(JUDGE_MODELS))
    category_prep_id: str | None = None


class ScoreBatchResponse(BaseModel):
    eval_run_id: str
    batch_id: str
    scoring_records: list[dict[str, Any]]
    summary: dict[str, Any]


class SimulateObservationRequest(BaseModel):
    eval_run_id: str
    sample_id: str
    prompt: str
    assistant_output: str
    messages: list[dict[str, str]] | None = None


class SimulateObservationResponse(BaseModel):
    eval_run_id: str
    sample_id: str
    observation: str


@dataclass(frozen=True)
class QuestionPrepResult:
    questions: list[dict[str, str]]
    source: dict[str, object]
    error: str | None = None


@dataclass(frozen=True)
class QuestionPrepLookup:
    result: QuestionPrepResult | None
    reason: str


class QuestionScoringUnavailable(RuntimeError):
    pass


class ObservationSimulationUnavailable(RuntimeError):
    pass


BASE_PROMPT = """You are the ENVIRONMENT (execution harness) in a SWE-agent session. You are NOT the assistant and you must never act as the assistant.

You will receive a transcript with "### system", "### user" and "### assistant" section markers.
The transcript ends with the assistant's first message containing one command. Mentally execute
that command against the repository state implied by the task description and reply with the
environment's next message: the terminal output of that command.

STRICT RULES:
- Reply ONLY with the environment message in the exact format specified below — nothing else.
- NEVER write "THOUGHT:", never write a bash command, never write "### user" or "### assistant"
  headers, never use markdown code fences, never explain or comment. You are not solving the
  task; you are only the terminal returning the command's output.
- NEVER give task tips, hints, suggestions, next steps, encouragement, or any part of the
  solution. A terminal has no opinion: it only prints what the command outputs, even if the
  assistant is on the wrong track or asked a question.
- Emulate realistic tool behavior: sed -i, cp, mv, mkdir, rm print nothing on success; echo
  prints its argument; cat/sed -n print file content; grep -n prefixes matches with "NN:"
  (context lines with "NN-"); find/ls list paths one per line; failed commands print realistic
  error messages.
- If the assistant message contains MORE THAN ONE bash code block, only the FIRST block is
  executed — simulate the first command and ignore all later blocks.
- Respect pipe limits exactly: "| head -N" outputs at most N lines, "| tail -N" the last N.
  Count your output lines before replying.
- Anchor on evidence: file, directory and symbol names mentioned in the task description are
  real — build your output around them and the standard layout for the project's language.
  When you cannot infer paths with confidence, prefer FEWER lines over invented ones; if the
  command's filters plausibly match nothing in this project (e.g. a file extension foreign to
  its language), the output is empty.
"""

def _evaluator_provider(settings: JudgeSettings) -> dict[str, Any]:
    block: dict[str, Any] = {"allow_fallbacks": True, "quantizations": ["fp8"]}
    order = [p.strip() for p in settings.evaluator_providers.split(",") if p.strip()]
    if order:
        block["order"] = order
        block["allow_fallbacks"] = False
    return block


def _simulation_provider(settings: JudgeSettings) -> dict[str, Any] | None:
    allowed = [p.strip() for p in settings.simulation_providers.split(",") if p.strip()]
    if not allowed:
        return None
    return {"only": allowed}


_COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_REROLL_WINDOW_TURNS = 5


def _reference_completion_observation(fmt: str) -> str:
    return wrap(_COMPLETE_MARKER, fmt)


class ReferenceTrajectoryService:

    def __init__(
        self,
        settings: JudgeSettings,
        client: OpenRouterJudgeClient,
        simulator: "ObservationSimulationService",
    ):
        self.settings = settings
        self.client = client
        self.simulator = simulator

    def _model_for(self, sample_id: str, *, offset: int = 0) -> str:
        pool = [m.strip() for m in self.settings.sota_models.split(",") if m.strip()]
        if not pool:
            raise QuestionScoringUnavailable("ALBEDO_JUDGE_SOTA_MODELS is empty")
        index = random.Random(sample_id).randrange(len(pool))
        return pool[(index + offset) % len(pool)]

    async def generate(
        self, sample: QuestionPrepSample, *, eval_run_id: str = ""
    ) -> tuple[str, str, bool]:
        reference, model, made_edit, _ = await self._generate_once(
            sample, eval_run_id, extra_turns=0
        )
        if made_edit:
            return reference, model, made_edit
        try:
            longer = await self._generate_once(sample, eval_run_id, extra_turns=2)
        except QuestionScoringUnavailable:
            return reference, model, made_edit
        if longer[2]:
            return longer[0], longer[1], longer[2]
        return reference, model, made_edit

    async def reroll_for_material(
        self, sample: QuestionPrepSample, *, eval_run_id: str = "", exclude_model: str
    ) -> tuple[str, str, bool] | None:
        window = min(_REROLL_WINDOW_TURNS, self.settings.sota_trajectory_turns)
        extra = window - max(
            1, sample.assistant_turns or self.settings.sota_trajectory_turns
        )
        try:
            reference, model, made_edit, steps = await self._generate_once(
                sample, eval_run_id, extra_turns=extra, model_offset=1
            )
        except QuestionScoringUnavailable as exc:
            logger.warning(
                "reference_reroll_failed sample_id={} error={}", sample.sample_id, exc
            )
            return None
        if steps < 2 or model == exclude_model:
            return None
        logger.info(
            "reference_reroll_used sample_id={} replaced={} with={}/{}steps window={}",
            sample.sample_id, exclude_model, model, steps, window,
        )
        return reference, model, made_edit

    async def _generate_once(
        self, sample: QuestionPrepSample, eval_run_id: str, *, extra_turns: int,
        model_offset: int = 0,
    ) -> tuple[str, str, bool, int]:
        model = self._model_for(sample.sample_id, offset=model_offset)
        turn_count = (
            max(1, sample.assistant_turns or self.settings.sota_trajectory_turns) + extra_turns
        )
        convo = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in (sample.messages or [])
        ]
        fmt = detect_format(sample.sample_id, sample.messages)
        turns: list[dict[str, Any]] = []
        for turn_index in range(turn_count):
            response = await self.client.complete(
                purpose="reference",
                model=model,
                messages=convo,
                temperature=0.0,
                max_tokens=self.settings.sota_max_tokens,
                provider=_evaluator_provider(self.settings)
                if model == self.settings.evaluator_model
                else None,
                accept=lambda raw: bool(raw.strip()),
            )
            if response.error or not response.raw.strip():
                raise QuestionScoringUnavailable(
                    f"reference generation failed: {response.error or 'empty output'}"
                )
            text = response.raw.strip()
            turns.append({"role": "assistant", "content": text, "score_target": True})
            last = turn_index == turn_count - 1
            if _COMPLETE_MARKER in text:
                if not last:
                    turns.append(
                        {
                            "role": "user",
                            "content": _reference_completion_observation(fmt),
                            "environment_observation": True,
                        }
                    )
                break
            if last:
                break
            observation = await self.simulator.simulate(
                SimulateObservationRequest(
                    eval_run_id=eval_run_id,
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    assistant_output=text,
                    messages=convo,
                )
            )
            turns.append(
                {"role": "user", "content": observation, "environment_observation": True}
            )
            convo = convo + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": observation},
            ]
        reference = format_reference_trajectory(turns)
        if not reference.strip():
            raise QuestionScoringUnavailable("reference trajectory rendered empty")
        generated = [t["content"] for t in turns if t.get("score_target")]
        made_edit = trajectory_made_edit(generated)
        return reference, model, made_edit, len(generated)


class QuestionService:

    def __init__(
        self,
        settings: JudgeSettings,
        client: OpenRouterJudgeClient,
        reference_service: ReferenceTrajectoryService | None = None,
    ):
        self.settings = settings
        self.client = client
        self.reference_service = reference_service

    async def prepare(
        self, sample: QuestionPrepSample | JudgeSample, *, eval_run_id: str = ""
    ) -> QuestionPrepResult:
        reference: str | None = None
        reference_model: str | None = None
        reference_made_edit = False
        anchoring_intended = (
            self.reference_service is not None and bool(getattr(sample, "messages", None))
        )
        if anchoring_intended:
            try:
                reference, reference_model, reference_made_edit = (
                    await self.reference_service.generate(sample, eval_run_id=eval_run_id)
                )
            except Exception as exc:
                logger.warning(
                    "reference_trajectory_failed sample_id={} error={} retrying=reference_reroll",
                    sample.sample_id, f"{type(exc).__name__}: {exc}",
                )
                rerolled = await self.reference_service.reroll_for_material(
                    sample, eval_run_id=eval_run_id, exclude_model=""
                )
                if rerolled is None:
                    raise QuestionScoringUnavailable(
                        f"reference unavailable: {type(exc).__name__}: {exc}"
                    ) from exc
                reference, reference_model, reference_made_edit = rerolled
        if reference is not None:
            try:
                return await self._prepare_once(
                    sample, reference, reference_model, reference_made_edit
                )
            except QuestionScoringUnavailable as exc:
                if self.reference_service is None:
                    raise
                logger.warning(
                    "anchored_questions_failed sample_id={} error={} retrying=reference_reroll",
                    sample.sample_id, exc,
                )
                rerolled = await self.reference_service.reroll_for_material(
                    sample, eval_run_id=eval_run_id, exclude_model=reference_model or ""
                )
                if rerolled is None:
                    raise
                return await self._prepare_once(sample, *rerolled)
        return await self._prepare_once(sample, None, None, False)

    async def _prepare_once(
        self,
        sample: QuestionPrepSample | JudgeSample,
        reference: str | None,
        reference_model: str | None,
        reference_made_edit: bool,
    ) -> QuestionPrepResult:
        n = self.settings.num_questions

        def _accept(raw: str) -> bool:
            questions, ok = parse_questions(raw, n)
            if reference is not None:
                questions = filter_reference_leaks(questions)
                questions, _ = enforce_question_labels(
                    questions, reference_made_edit=reference_made_edit
                )
            return ok and len(questions) >= question_floor(n)

        response = await self.client.complete(
            purpose="questions",
            model=self.settings.evaluator_model,
            messages=build_question_messages(
                task=sample.prompt, n=n, reference=reference,
                reference_made_edit=reference_made_edit if reference is not None else None,
            ),
            temperature=self.settings.temperature,
            max_tokens=self.settings.question_max_tokens,
            provider=_evaluator_provider(self.settings),
            response_schema=question_schema(n),
            accept=_accept,
        )
        if response.error:
            raise QuestionScoringUnavailable(response.error)
        questions, ok = parse_questions(response.raw, n)
        drops: dict[str, int] = {}
        if reference is not None:
            questions = filter_reference_leaks(questions)
            questions, drops = enforce_question_labels(
                questions, reference_made_edit=reference_made_edit
            )
        if not ok or len(questions) < question_floor(n):
            raise QuestionScoringUnavailable(
                f"evaluator returned {len(questions)}/{n} well-formed questions"
            )
        source: dict[str, object] = {
            "provider": response.provider,
            "model": self.settings.evaluator_model,
            "n_questions": len(questions),
            "question_mode": "sota_anchored" if reference is not None else "task_only",
            "reference_made_edit": reference_made_edit if reference is not None else None,
            "enforcement_drops": drops,
        }
        if reference_model:
            source["reference_model"] = reference_model
        if reference is not None:
            source["reference_trajectory"] = reference
        return QuestionPrepResult(questions=questions, source=source)


class RepoContextClient:

    def __init__(self, settings: JudgeSettings):
        self._client = httpx.AsyncClient(
            base_url=settings.repo_context_url.rstrip("/"),
            timeout=settings.repo_context_timeout_seconds,
        )
        self._last_warning = 0.0

    async def context_for(self, sample_id: str, assistant_output: str) -> str | None:
        try:
            response = await self._client.post(
                "/repo-context",
                json={"sample_id": sample_id, "assistant_output": assistant_output},
            )
            response.raise_for_status()
            context = response.json().get("context")
            return context if isinstance(context, str) and context else None
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_warning > 60.0:
                self._last_warning = now
                logger.warning(
                    "repo_context_unavailable sample_id={} error={}",
                    sample_id, f"{type(exc).__name__}: {exc}",
                )
            return None

    async def aclose(self) -> None:
        await self._client.aclose()


class ObservationSimulationService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: OpenRouterJudgeClient,
        repo_context: RepoContextClient | None = None,
    ):
        self.settings = settings
        self.client = client
        self.repo_context = repo_context

    async def simulate(self, request: SimulateObservationRequest) -> str:
        command = first_bash_block(request.assistant_output)
        if not command:
            fmt = detect_format(request.sample_id, request.messages)
            logger.warning(
                "observation_simulation_no_command eval_run_id={} sample_id={} fmt={} chars={}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                len(request.assistant_output or ""),
            )
            return missing_command_output(fmt)
        context_block = None
        if self.repo_context is not None:
            context_block = await self.repo_context.context_for(
                request.sample_id, request.assistant_output
            )
        transcript = _simulation_transcript(
            messages=request.messages,
            prompt=request.prompt,
            assistant_output=request.assistant_output,
        )
        fmt = detect_format(request.sample_id, request.messages)
        require_content = requires_output(command)
        contract = command_contract(command)
        primary = self.settings.simulation_model or self.settings.evaluator_model
        fallback_model = self.settings.evaluator_model
        attempts: list[tuple[str, int]] = [
            (primary, self.settings.simulation_loop_reruns + 1)
        ]
        if primary != fallback_model:
            attempts.append((fallback_model, 1))

        observation = ""
        best_rank = -1
        for model, tries in attempts:
            single_shot = model == primary and primary != fallback_model
            messages = [
                {
                    "role": "system",
                    "content": _simulation_system_prompt(fmt, context_block),
                },
                {"role": "user", "content": transcript},
            ]
            single_shot_kwargs = {"parse_retries": 1, "retry_count": 0} if single_shot else {}
            for attempt in range(tries):
                response = await self.client.complete(
                    purpose="simulate",
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.settings.simulation_max_tokens,
                    provider=(_evaluator_provider(self.settings)
                              if model == fallback_model
                              else _simulation_provider(self.settings)),
                    accept=lambda raw: _usable_simulation_output(
                        repair_to_contract(repair_output(raw, fmt), fmt, contract), fmt,
                        require_content=require_content, contract=contract,
                    ),
                    **single_shot_kwargs,
                )
                if response.error:
                    if model != fallback_model:
                        break
                    raise ObservationSimulationUnavailable(response.error)
                candidate = repair_to_contract(
                    repair_output(response.raw, fmt), fmt, contract
                )
                rank = _candidate_rank(
                    candidate, fmt, require_content=require_content, contract=contract
                )
                if rank > best_rank:
                    best_rank, observation = rank, candidate
                if rank == _RANK_USABLE:
                    if model != primary:
                        logger.info(
                            "observation_simulation_fallback_used eval_run_id={} sample_id={} "
                            "primary={} fallback={}",
                            request.eval_run_id, request.sample_id, primary, model,
                        )
                    break
                logger.warning(
                    "observation_simulation_unusable eval_run_id={} sample_id={} model={} "
                    "attempt={}/{} reason={} kept_rank={}",
                    request.eval_run_id, request.sample_id, model, attempt + 1, tries,
                    _unusable_reason(
                        candidate, fmt, require_content=require_content, contract=contract
                    ),
                    best_rank,
                )
            if best_rank == _RANK_USABLE:
                break
        if _looping_output(observation):
            collapsed = _collapse_looping(observation).strip()
            logger.warning(
                "observation_simulation_looping_collapsed eval_run_id={} sample_id={} chars={}->{}",
                request.eval_run_id,
                request.sample_id,
                len(observation),
                len(collapsed),
            )
            observation = collapsed
        if not valid_output(observation, fmt):
            fallback = empty_output(fmt)
            logger.warning(
                "observation_simulation_invalid_format eval_run_id={} sample_id={} fmt={} "
                "fallback={!r}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                fallback,
            )
            return fallback
        return observation


class QuestionPrepStore:

    def __init__(self, settings: JudgeSettings, service: QuestionService):
        self.settings = settings
        self.service = service
        self._preps: dict[str, dict[str, asyncio.Task[QuestionPrepResult]]] = {}
        self._created_at: dict[str, float] = {}

    def start(self, request: QuestionPrepRequest) -> str:
        self._sweep_expired()
        prep_id = f"{request.eval_run_id}:{uuid4()}"
        self._created_at[prep_id] = time.monotonic()
        self._preps[prep_id] = {
            sample.sample_id: asyncio.create_task(self._prepare_sample(prep_id, request, sample))
            for sample in request.samples
        }
        return prep_id

    async def get_with_reason(self, prep_id: str, sample: JudgeSample) -> QuestionPrepLookup:
        self._sweep_expired()
        tasks = self._preps.get(prep_id)
        if not tasks:
            return QuestionPrepLookup(None, "unknown_or_expired_prep_id")
        task = tasks.get(sample.sample_id)
        if task is None:
            return QuestionPrepLookup(None, "sample_not_in_prep")
        return QuestionPrepLookup(await task, "prepared")

    async def _prepare_sample(
        self, prep_id: str, request: QuestionPrepRequest, sample: QuestionPrepSample
    ) -> QuestionPrepResult:
        try:
            return await self.service.prepare(sample, eval_run_id=request.eval_run_id)
        except Exception as exc:
            logger.warning(
                "question_prep_sample_failed eval_run_id={} prep_id={} sample_id={} error={}",
                request.eval_run_id, prep_id, sample.sample_id, f"{type(exc).__name__}: {exc}",
            )
            raise

    def _sweep_expired(self) -> None:
        ttl = self.settings.question_prep_ttl_seconds
        now = time.monotonic()
        for prep_id in [pid for pid, created in self._created_at.items() if now - created > ttl]:
            for task in self._preps.get(prep_id, {}).values():
                if not task.done():
                    task.cancel()
            self._preps.pop(prep_id, None)
            self._created_at.pop(prep_id, None)


def create_app(settings: JudgeSettings | None = None) -> FastAPI:
    settings = settings or get_judge_settings()
    app = FastAPI(title="Albedo Judge API")

    @app.on_event("startup")
    async def startup() -> None:
        client = OpenRouterJudgeClient(settings)
        app.state.eval_client = client
        repo_context = RepoContextClient(settings) if settings.repo_context_url else None
        app.state.repo_context_client = repo_context
        app.state.observation_service = ObservationSimulationService(settings, client, repo_context)
        app.state.question_service = QuestionService(
            settings, client,
            ReferenceTrajectoryService(settings, client, app.state.observation_service),
        )
        app.state.question_prep_store = QuestionPrepStore(settings, app.state.question_service)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        client = getattr(app.state, "eval_client", None)
        if client is not None:
            await client.aclose()
        repo_context = getattr(app.state, "repo_context_client", None)
        if repo_context is not None:
            await repo_context.aclose()

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_auth_token:
            return
        if authorization != f"Bearer {settings.api_auth_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def prep_store() -> QuestionPrepStore:
        store = getattr(app.state, "question_prep_store", None)
        if store is None:
            client = OpenRouterJudgeClient(settings)
            app.state.eval_client = client
            repo_context = RepoContextClient(settings) if settings.repo_context_url else None
            app.state.repo_context_client = repo_context
            app.state.observation_service = ObservationSimulationService(
                settings, client, repo_context
            )
            app.state.question_service = QuestionService(
                settings, client,
                ReferenceTrajectoryService(settings, client, app.state.observation_service),
            )
            app.state.question_prep_store = QuestionPrepStore(settings, app.state.question_service)
        return app.state.question_prep_store

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(_: None = Depends(require_auth)) -> dict[str, object]:
        return {
            "status": "ready",
            "judge_models": list(JUDGE_MODELS),
            "evaluator_model": settings.evaluator_model,
            "num_questions": settings.num_questions,
        }

    @app.post("/category-prep", response_model=QuestionPrepResponse)
    async def category_prep(
        request: QuestionPrepRequest, _: None = Depends(require_auth)
    ) -> QuestionPrepResponse:
        prep_id = prep_store().start(request)
        return QuestionPrepResponse(
            eval_run_id=request.eval_run_id,
            category_prep_id=prep_id,
            accepted_sample_count=len(request.samples),
        )

    @app.post("/simulate-observation", response_model=SimulateObservationResponse)
    async def simulate_observation(
        request: SimulateObservationRequest, _: None = Depends(require_auth)
    ) -> SimulateObservationResponse:
        service: ObservationSimulationService = app.state.observation_service
        observation = await service.simulate(request)
        return SimulateObservationResponse(
            eval_run_id=request.eval_run_id,
            sample_id=request.sample_id,
            observation=observation,
        )

    @app.post("/score-batch", response_model=ScoreBatchResponse)
    async def score_batch(
        request: ScoreBatchRequest, _: None = Depends(require_auth)
    ) -> ScoreBatchResponse:
        unknown = [model for model in request.judge_models if model not in JUDGE_MODELS]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unsupported judge model(s): {', '.join(unknown)}")
        client: OpenRouterJudgeClient = app.state.eval_client
        try:
            records = await _score_samples(
                client=client, request=request, settings=settings, prep_store=prep_store()
            )
        except Exception as exc:
            _notify(
                settings, request, severity="ERROR",
                message="Scoring failed", fault_code="scoring_failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            logger.exception(
                f"[judge-api] scoring failed eval_run={request.eval_run_id} batch={request.batch_id}: {exc}"
            )
            raise HTTPException(status_code=502, detail=f"scoring failed: {exc}")
        summary = aggregate_scores(records, min_valid_fraction=settings.min_valid_fraction)
        if summary.get("state") != "succeeded":
            _notify(
                settings, request, severity="WARNING",
                message="Scoring produced too few valid samples",
                fault_code=str(summary.get("fault_code") or "scoring_invalid"),
                retryable=bool(summary.get("retryable")),
            )
        return ScoreBatchResponse(
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            scoring_records=records,
            summary=summary,
        )

    return app


async def _questions_for(
    request: ScoreBatchRequest, sample: JudgeSample, prep_store: QuestionPrepStore
) -> QuestionPrepResult:
    if request.category_prep_id:
        try:
            lookup = await prep_store.get_with_reason(request.category_prep_id, sample)
        except Exception as exc:
            reason = f"prep_failed:{type(exc).__name__}"
        else:
            if lookup.result is not None:
                return lookup.result
            reason = lookup.reason
    else:
        reason = "missing_prep_id"
    logger.warning(
        "score_batch_question_sync_generation eval_run_id={} batch_id={} sample_id={} reason={}",
        request.eval_run_id, request.batch_id, sample.sample_id, reason,
    )
    return await prep_store.service.prepare(sample)


_COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)


def _command_only(text: str) -> str:
    match = _COMMAND_BLOCK_RE.search(text or "")
    if match:
        return f"```bash\n{match.group(1).strip()}\n```"
    return text


def _simulation_transcript(
    *,
    messages: list[dict[str, str]] | None,
    prompt: str,
    assistant_output: str,
) -> str:
    transcript_messages = messages or [{"role": "user", "content": prompt}]
    sections = []
    for message in transcript_messages + [{"role": "assistant", "content": assistant_output}]:
        role = str(message.get("role") or "user").lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(message.get("content") or "").rstrip()
        if role == "assistant":
            content = _command_only(content)
        sections.append(f"### {role}\n{content}")
    return "\n\n".join(sections).rstrip()


def _simulation_system_prompt(fmt: str, context_block: str | None = None) -> str:
    block = format_block(fmt)
    if not context_block:
        return f"{BASE_PROMPT}\n{block}"
    return f"{BASE_PROMPT}\n{context_block}\n{block}"


_LOOP_LINE_RUN = 25
_LOOP_TAIL_WINDOW = 512
_LOOP_MIN_REPEATS = 4


def _looping_output(text: str) -> bool:
    run = 1
    prev: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped == prev:
            run += 1
            if run >= _LOOP_LINE_RUN:
                return True
        elif stripped:
            run = 1
            prev = stripped
    return _trailing_cycle_period(text) > 0


def _trailing_cycle_period(text: str) -> int:
    tail = text.rstrip()[-_LOOP_TAIL_WINDOW:]
    if len(tail) < _LOOP_TAIL_WINDOW:
        return 0
    for period in range(1, _LOOP_TAIL_WINDOW // _LOOP_MIN_REPEATS + 1):
        if tail[period:] == tail[:-period]:
            return period
    return 0


def _collapse_looping(text: str) -> str:
    out: list[str] = []
    run = 1
    prev: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped == prev:
            run += 1
            if run == _LOOP_LINE_RUN:
                out.append("... (output repeats)")
            if run >= _LOOP_LINE_RUN:
                continue
        elif stripped:
            run = 1
            prev = stripped
        out.append(line)
    collapsed = "\n".join(out)
    period = _trailing_cycle_period(collapsed)
    if period:
        stripped_text = collapsed.rstrip()
        index = len(stripped_text) - period - 1
        while index >= 0 and stripped_text[index] == stripped_text[index + period]:
            index -= 1
        keep = min(len(stripped_text), index + 1 + 2 * period)
        collapsed = stripped_text[:keep].rstrip() + "\n... (output repeats)"
    return collapsed


_ROLE_LEAK_RE = re.compile(r"(?:^|\n)\s*(?:THOUGHT:|### (?:assistant|user|system)\b)")


def _role_violation(raw: str) -> bool:
    return bool(_ROLE_LEAK_RE.search(raw or ""))


_RANK_INVALID = 0
_RANK_VALID = 1
_RANK_HAS_CONTENT = 2
_RANK_USABLE = 3


def _candidate_rank(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> int:
    """How good an attempt is, so escalation keeps the best one rather than the last.

    Escalating used to overwrite a usable primary result with whatever the fallback produced. In
    practice the fallback often answers in the wrong dialect, which then collapsed to an empty
    observation, or returned a worse contract violation — both strictly worse than the primary.
    """
    if not valid_output(raw, fmt):
        return _RANK_INVALID
    if _usable_simulation_output(
        raw, fmt, require_content=require_content, contract=contract
    ):
        return _RANK_USABLE
    if has_content(raw, fmt):
        return _RANK_HAS_CONTENT
    return _RANK_VALID


def _usable_simulation_output(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> bool:
    return (
        valid_output(raw, fmt)
        and not _role_violation(raw)
        and not _looping_output(raw)
        and (not require_content or has_content(raw, fmt))
        and (contract is None or contract_violation(raw, fmt, contract) is None)
    )


def _unusable_reason(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
) -> str:
    if not valid_output(raw, fmt):
        return "invalid_format"
    if _role_violation(raw):
        return "role_violation"
    if _looping_output(raw):
        return "looping"
    if require_content and not has_content(raw, fmt):
        return "no_content_for_read"
    if contract is not None and (breach := contract_violation(raw, fmt, contract)):
        return breach
    return "ok"


def _corrupted_side(
    *,
    side: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    per_judge_answers: dict[str, dict[str, str | None]] = {
        model: {q["id"]: "0" for q in questions} for model in judge_models
    }
    records = [
        {
            "side": side,
            "judge_model": model,
            "provider": None,
            "answers": per_judge_answers[model],
            "explanations": {},
            "yes_rate": judge_yes_rate(per_judge_answers[model], questions),
            "parse_ok": True,
            "error": None,
            "corrupted": True,
        }
        for model in judge_models
    ]
    return per_judge_answers, records


async def _judge_side(
    *,
    client: OpenRouterJudgeClient,
    settings: JudgeSettings,
    side: str,
    response_text: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
    reference_made_edit: bool | None = None,
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    question_ids = [q["id"] for q in questions]
    schema = answer_schema(question_ids)
    messages = build_judge_messages(response=response_text, questions=questions)
    raws = await asyncio.gather(
        *[
            client.score(
                model=model,
                messages=messages,
                response_schema=schema,
                schema_name="albedo_answers",
                max_tokens=settings.answer_max_tokens,
                accept=lambda raw: parse_answers(raw, question_ids)[2],
            )
            for model in judge_models
        ]
    )
    per_judge_answers: dict[str, dict[str, str | None]] = {}
    records: list[dict[str, Any]] = []
    gate_turns = (
        candidate_turn_texts_from_merged(response_text)
        if reference_made_edit is not None
        else None
    )
    for raw, model in zip(raws, judge_models):
        answers, explanations, parse_ok = parse_answers(raw.raw, question_ids)
        if gate_turns is not None:
            answers = apply_measurement_gate(
                answers, questions,
                candidate_turn_texts=gate_turns,
                reference_made_edit=bool(reference_made_edit),
            )
        per_judge_answers[model] = answers
        records.append(
            {
                "side": side,
                "judge_model": model,
                "provider": raw.provider,
                "answers": answers,
                "explanations": explanations,
                "yes_rate": judge_yes_rate(answers, questions),
                "parse_ok": parse_ok and not raw.error,
                "error": raw.error,
            }
        )
    return per_judge_answers, records


async def _score_samples(
    *,
    client: OpenRouterJudgeClient,
    request: ScoreBatchRequest,
    settings: JudgeSettings,
    prep_store: QuestionPrepStore,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    completed = 0
    progress_lock = asyncio.Lock()
    logger.info(
        "score_batch_started eval_run_id={} batch_id={} samples={} judges={} prep_id={}",
        request.eval_run_id, request.batch_id, len(request.samples),
        len(request.judge_models), request.category_prep_id or "",
    )

    async def _score_one(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        try:
            return await _score_one_inner(sample)
        except Exception as exc:
            async with progress_lock:
                completed += 1
            logger.warning(
                "score_batch_sample_failed eval_run_id={} batch_id={} completed={}/{} sample_id={} error={}",
                request.eval_run_id, request.batch_id, completed, len(request.samples),
                sample.sample_id, f"{type(exc).__name__}: {exc}",
            )
            return {
                "sample_id": sample.sample_id,
                "questions": [],
                "king_score": None,
                "challenger_score": None,
                "judge_results": [],
                "scored": False,
                "scoring_mode": "binary",
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _score_one_inner(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        prepared = await _questions_for(request, sample, prep_store)
        if prepared.error:
            raise QuestionScoringUnavailable(prepared.error)
        questions = prepared.questions
        gate_flag = prepared.source.get("reference_made_edit")
        gate_flag = bool(gate_flag) if gate_flag is not None else None
        async def _side(side: str, response_text: str):
            if is_truncated(response_text):
                return _corrupted_side(
                    side=side, questions=questions, judge_models=request.judge_models
                )
            return await _judge_side(
                client=client, settings=settings, side=side,
                response_text=response_text, questions=questions,
                judge_models=request.judge_models, reference_made_edit=gate_flag,
            )

        (king_answers, king_recs), (chal_answers, chal_recs) = await asyncio.gather(
            _side("previous_king", sample.previous_king_output),
            _side("challenger", sample.challenger_output),
        )
        king_score = response_score(king_answers, questions)
        chal_score = response_score(chal_answers, questions)
        king_ok = all(r["parse_ok"] for r in king_recs) and king_score is not None
        chal_ok = all(r["parse_ok"] for r in chal_recs) and chal_score is not None
        scored = king_ok and chal_ok
        async with progress_lock:
            completed += 1
            logger.info(
                "score_batch_sample_done eval_run_id={} batch_id={} completed={}/{} sample_id={} "
                "scored={} king={} chal={} elapsed_s={:.1f}",
                request.eval_run_id, request.batch_id, completed, len(request.samples),
                sample.sample_id, scored, king_score, chal_score, time.monotonic() - started_at,
            )
        return {
            "sample_id": sample.sample_id,
            "questions": questions,
            "king_score": king_score,
            "challenger_score": chal_score,
            "judge_results": king_recs + chal_recs,
            "scored": scored,
            "scoring_mode": "binary",
            "question_source": prepared.source,
        }

    records = await asyncio.gather(*[_score_one(sample) for sample in request.samples])
    logger.info(
        "score_batch_done eval_run_id={} batch_id={} scored={}/{} elapsed_s={:.1f}",
        request.eval_run_id, request.batch_id,
        sum(1 for r in records if r.get("scored")), len(records), time.monotonic() - started_at,
    )
    return list(records)


def _notify(
    settings: JudgeSettings,
    request: ScoreBatchRequest,
    *,
    severity: str,
    message: str,
    fault_code: str,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    notify_eval_error(
        EvalErrorNotification(
            component="judge_api",
            severity=severity,
            message=message,
            eval_run_id=request.eval_run_id,
            batch_id=request.batch_id,
            fault_class="PROVIDER_FAULT",
            fault_code=fault_code,
            scoring_mode="binary",
            retryable=retryable,
            details=details,
        ),
        webhook_url=settings.slack_error_webhook_url,
    )


def main() -> None:
    settings = get_judge_settings()
    uvicorn.run(
        "albedo_eval_service.judge_api:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
