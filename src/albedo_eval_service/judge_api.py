from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from albedo_config import JudgeSettings, get_judge_settings
from albedo_config.models import JUDGE_MODELS

from .control.notifications import EvalErrorNotification, notify_eval_error
from .evaluator.reference.prompt_ladder import (
    LADDER_MIN,
    build_ladder_messages,
    ladder_schema,
    parse_ladder,
    project_vector,
)
from .evaluator.reference.prompt_milestones import (
    build_extractor_messages,
    extractor_schema,
    milestone_tag,
    validate_vector,
)
from .evaluator.reference.questions import (
    filter_reference_leaks,
    format_reference_trajectory,
)
from .evaluator.reference.vector_merge import (
    build_merge_messages,
    build_question_merge_messages,
    exact_groups,
    merge_schema,
    merge_vectors,
    needs_alignment,
    parse_clusters,
    select_questions,
)
from .evaluator.shared.questions import (
    enforce_question_labels,
    sample_phase,
    trajectory_made_edit,
)
from .judge_core import (
    AMPUTATED_THINKING_MULTIPLIER,
    aggregate_scores,
    amputated_thinking,
    answer_schema,
    build_judge_messages,
    judge_yes_rate,
    majority_answers,
    parse_answers,
    question_weight,
    reserved_token_leak,
    response_score,
)
from .judge_llm_client import JudgeLLMClient
from .remote.generation import format_scored_trajectory
from .repo_context_client import Grounding, RepoContextClient
from .shared.edit_detection import any_shows_work, named_in_removal
from .shared.json_extract import extract_json
from .shared.loop_check import LoopVerdict, loop_explanation, loop_verdict_for_document
from .shared.observation_format import (
    NOT_DERIVABLE,
    ROLE_MARKER_RE,
    CommandContract,
    absent_tool_output,
    canonical_empty,
    claims_tracked_change,
    command_contract,
    contract_violation,
    correct_returncode,
    degenerate_observation,
    deleted_files,
    detect_format,
    echoed_command,
    empty_output,
    first_bash_block,
    grounded_observation,
    has_content,
    impossible_success,
    is_abandoned,
    is_file_read,
    is_truncated,
    no_output_notice,
    output_expectation,
    pipeline_returncode_override,
    renumbered_view,
    repair_output,
    repair_to_contract,
    requires_output,
    stuttered_lines,
    valid_output,
    with_body,
    without_tracked_changes,
    wrap,
)
from .shared.observation_memo import ObservationMemo
from .shared.pip_check import fabricated_pip_error
from .shared.sed_check import fabricated_sed_error, misdiagnosed_sed
from .shared.submit_protocol import first_bash_command, is_exact_submission
from .simulator.prompt_simulator import (
    COMPLETE_MARKER,
    COMPUTED_BLOCK_MARKER,
    MUST_PRINT_RETRY,
    missing_command_output,
    reference_completion_observation,
    simulation_system_prompt,
)


class QuestionPrepSample(BaseModel):
    sample_id: str
    prompt: str
    sample_index: int = 0
    messages: list[dict[str, str]] | None = None
    assistant_turns: int = 0
    submit_marker: str = ""
    submit_command: str = ""


def _sample_submitted(sample: QuestionPrepSample, text: str) -> bool:
    if sample.submit_command:
        return is_exact_submission(text, sample.submit_command)
    return COMPLETE_MARKER in text


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
    submit_marker: str = ""
    submit_command: str = ""


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
    return {"order": allowed, "allow_fallbacks": False}


_REROLL_WINDOW_TURNS = 5
_MEMO_MAX_ENTRIES = 8192


class ReferenceTrajectoryService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
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
    ) -> tuple[str, str, bool, list[dict[str, Any]]]:
        reference, model, made_edit, _, turns = await self._generate_once(
            sample, eval_run_id, extra_turns=0
        )
        return reference, model, made_edit, turns

    async def generate_many(
        self, sample: QuestionPrepSample, n: int, *, eval_run_id: str = ""
    ) -> list[tuple[str, str, bool, list[dict[str, Any]]]]:
        """N runs of the same task, walking the SOTA pool where there is more than one model in it.

        The runs diverge on their own: the model is not deterministic at a fixed temperature, and
        the simulator answers each run's own commands. Several runs are what keeps one degenerate
        simulation from deciding the whole checklist, and what lets the extractor tell a fact the
        task requires from one run's detour.

        Run concurrently. The observation memo keys on (sample_id, format, repo state, command) and
        dedupes in-flight work, so two runs issuing the same command in the same state still share
        one simulation rather than racing for two - the shared world survives the parallelism.

        One run may fail without taking the sample with it: two runs still support consensus, and
        `validate_vector` records which runs reached each milestone either way. Below two there is
        nothing left to compare and the sample is unusable.
        """
        results = await asyncio.gather(
            *(
                self._generate_once(sample, eval_run_id, extra_turns=0, model_offset=index)
                for index in range(max(1, n))
            ),
            return_exceptions=True,
        )
        runs: list[tuple[str, str, bool, list[dict[str, Any]]]] = []
        for index, result in enumerate(results, start=1):
            if isinstance(result, BaseException):
                logger.warning(
                    "reference_run_failed sample_id={} run={} error={}",
                    sample.sample_id,
                    index,
                    f"{type(result).__name__}: {result}",
                )
                continue
            reference, model, made_edit, _, turns = result
            runs.append((reference, model, made_edit, turns))
        if len(runs) < min(2, max(1, n)):
            raise QuestionScoringUnavailable(
                f"reference generation produced {len(runs)}/{n} usable runs"
            )
        return runs

    async def _generate_once(
        self,
        sample: QuestionPrepSample,
        eval_run_id: str,
        *,
        extra_turns: int,
        model_offset: int = 0,
    ) -> tuple[str, str, bool, int, list[dict[str, Any]]]:
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
                eval_run_id=eval_run_id,
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
            if _sample_submitted(sample, text):
                if not last:
                    turns.append(
                        {
                            "role": "user",
                            "content": reference_completion_observation(fmt, sample.submit_marker),
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
            turns.append({"role": "user", "content": observation, "environment_observation": True})
            convo = convo + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": observation},
            ]
        reference = format_reference_trajectory(turns)
        if not reference.strip():
            raise QuestionScoringUnavailable("reference trajectory rendered empty")
        generated = [t["content"] for t in turns if t.get("score_target")]
        made_edit = trajectory_made_edit(generated)

        return reference, model, made_edit, len(generated), turns


# Fewest questions a WHOLE checklist can have
QUESTION_FLOOR = 6


def _reference_document(prefix: list[dict[str, str]] | None, turns: list[dict[str, Any]]) -> str:
    """A reference run rendered as the judgeable document a candidate would be.

    The prefix goes in as unscored context and every reference step becomes a CANDIDATE OUTPUT, so
    a reference is judged in the same shape a candidate is - otherwise a difference in answer could
    be a difference in framing rather than in the work.
    """
    context = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in (prefix or [])
        if m.get("content")
    ]
    return format_scored_trajectory(context + turns)


def _steps_from_turns(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The turns a reference run produced, as the {assistant, observation} pairs the vector wants.

    Built from the turns themselves rather than by re-splitting the rendered `REFERENCE STEP n:`
    text: the structure is already present here, and recovering it from the rendering loses any
    step whose observation happens to contain the marker.
    """
    steps: list[dict[str, str]] = []
    for turn in turns:
        if turn.get("score_target"):
            steps.append({"assistant": str(turn.get("content") or ""), "observation": ""})
        elif turn.get("environment_observation") and steps:
            steps[-1]["observation"] = str(turn.get("content") or "")
    return steps


class QuestionService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
        reference_service: ReferenceTrajectoryService,
    ):
        self.settings = settings
        self.client = client
        self.reference_service = reference_service

    async def prepare(
        self, sample: QuestionPrepSample | JudgeSample, *, eval_run_id: str = ""
    ) -> QuestionPrepResult:
        if not getattr(sample, "messages", None):
            raise QuestionScoringUnavailable(
                "sample carries no prior context to anchor a reference trajectory"
            )
        runs = await self.reference_service.generate_many(
            sample, self.settings.reference_runs, eval_run_id=eval_run_id
        )
        return await self._prepare_once(sample, runs)

    async def _extract_vector(
        self, task: str, references: list[str], candidate_turns: int
    ) -> tuple[list[dict[str, Any]], str]:
        """The runs, read against each other, become an ordered vector of milestones."""
        response = await self.client.complete(
            purpose="questions",
            model=self.settings.evaluator_model,
            messages=build_extractor_messages(
                task=task, references=references, candidate_turns=candidate_turns
            ),
            temperature=self.settings.temperature,
            max_tokens=self.settings.question_max_tokens,
            provider=_evaluator_provider(self.settings),
            response_schema=extractor_schema(),
        )
        if response.error:
            raise QuestionScoringUnavailable(f"milestone extraction failed: {response.error}")
        payload = extract_json(response.raw or "", prefer_keys=("milestones",))
        if isinstance(payload, list):
            payload = {"milestones": payload}
        milestones = (payload or {}).get("milestones") if isinstance(payload, dict) else None
        return list(milestones or []), response.provider or ""

    async def _extract_validated_vector(
        self,
        task: str,
        references: list[str],
        candidate_turns: int,
        trajectories: list[dict[str, Any]],
        problem: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], str, int, int]:
        """`milestone_readings` independent readings of the same runs, merged by fact.

        One reading is the degenerate case and needs no special path: a single vector clusters to
        itself, `needs_alignment` is false so no aligner is called, and `merge_vectors` returns
        that reading in causal order.
        """
        readings = max(1, int(getattr(self.settings, "milestone_readings", 1) or 1))

        async def read() -> tuple[
            list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], str
        ]:
            raw, provider = await self._extract_vector(task, references, candidate_turns)
            milestones, dropped = validate_vector(raw, trajectories, task)
            return raw, milestones, dropped, provider

        results = await asyncio.gather(*[read() for _ in range(readings)], return_exceptions=True)
        held = [r for r in results if not isinstance(r, BaseException)]
        if not held:
            first = next(r for r in results if isinstance(r, BaseException))
            raise QuestionScoringUnavailable(
                f"milestone extraction failed in all {readings} readings: {first}"
            )
        vectors = [milestones for _, milestones, _, _ in held]
        dropped = [
            {**record, "reading": index}
            for index, (_, _, records, _) in enumerate(held, 1)
            for record in records
        ]
        provider = next((p for _, _, _, p in held if p), "")
        emitted = sum(len(raw) for raw, _, _, _ in held)

        clusters = exact_groups(vectors)
        aligned = "exact"
        if needs_alignment(clusters, vectors):
            response = await self.client.complete(
                purpose="questions",
                model=self.settings.evaluator_model,
                messages=build_merge_messages(problem=problem or task[-1500:], readings=vectors),
                temperature=self.settings.temperature,
                max_tokens=4000,
                provider=_evaluator_provider(self.settings),
                response_schema=merge_schema(),
            )
            parsed = (
                None
                if response.error
                else parse_clusters(
                    extract_json(response.raw or "", prefer_keys=("clusters",)), vectors
                )
            )
            if parsed is None:
                logger.warning("milestone_alignment_unparseable readings={}", len(held))
                aligned = "exact_fallback"
            else:
                clusters, aligned = parsed, "model"
        merged = merge_vectors(vectors, clusters)
        logger.info(
            "milestone_readings held={}/{} kept_per_reading={} merged={} alignment={}",
            len(held),
            readings,
            [len(v) for v in vectors],
            len(merged),
            aligned,
        )
        return merged, dropped, provider, len(held), emitted

    async def _write_ladders(
        self, milestones: list[dict[str, Any]], approach: dict[int, list[str]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """`question_readings` calls for the whole vector, merged by what each question tests.

        Every reading is sent the WHOLE vector. Approach windows are a partition across the
        milestone sequence, so projecting a subset would silently widen them - every surviving
        milestone would look like the first in its run - and a reading over a slice would be
        answering a different question from the one that was asked.

        Milestones left under LADDER_MIN are returned for reporting; independent readings are what
        covers a writer that stops after the opening milestones, so nothing is re-asked.
        """
        vector = project_vector(milestones, approach)
        known = {str(m.get("id")) for m in milestones}

        async def ask(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
            response = await self.client.complete(
                purpose="questions",
                model=self.settings.evaluator_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.question_max_tokens,
                provider=_evaluator_provider(self.settings),
                response_schema=ladder_schema(),
            )
            return [] if response.error else parse_ladder(response.raw, known)

        order = [str(m.get("id")) for m in milestones]
        readings = max(1, int(getattr(self.settings, "question_readings", 1) or 1))
        lists = await asyncio.gather(
            *[ask(build_ladder_messages(vector=vector)) for _ in range(readings)]
        )
        lists = [q for q in lists if q]
        clusters = exact_groups(lists)
        if lists and needs_alignment(clusters, lists):
            response = await self.client.complete(
                purpose="questions",
                model=self.settings.evaluator_model,
                messages=build_question_merge_messages(lists),
                temperature=self.settings.temperature,
                max_tokens=6000,
                provider=_evaluator_provider(self.settings),
                response_schema=merge_schema(),
            )
            parsed = (
                None
                if response.error
                else parse_clusters(
                    extract_json(response.raw or "", prefer_keys=("clusters",)), lists
                )
            )
            if parsed is None:
                logger.warning("question_alignment_unparseable readings={}", len(lists))
            else:
                clusters = parsed
        by_id = select_questions(lists, clusters, LADDER_MIN)
        thin = [i for i in order if len(by_id.get(i, [])) < LADDER_MIN]
        logger.info(
            "question_readings held={}/{} per_reading={} kept={} thin={}",
            len(lists),
            readings,
            [len(q) for q in lists],
            sum(len(g) for g in by_id.values()),
            len(thin),
        )

        tags = {str(m.get("id")): milestone_tag(str(m.get("category") or "")) for m in milestones}
        statements = {str(m.get("id")): str(m.get("statement") or "") for m in milestones}
        merged = [
            {**question, "rung": index, "tag": tags[mid], "milestone_statement": statements[mid]}
            for mid in order
            for index, question in enumerate(by_id.get(mid, []), start=1)
        ]
        return merged, thin

    async def _prune_unreachable(
        self,
        questions: list[dict[str, Any]],
        runs: list[tuple[str, str, bool, list[dict[str, Any]]]],
        prefix: list[dict[str, str]] | None,
        discarded: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Drop questions that no reference run can answer.

        A milestone is extracted FROM these runs and its questions are written from that milestone,
        so a run should be able to answer them: it is the trajectory that did the work being asked
        about. A question none of them earns is asking for something no run did, or asking in a way
        the judge cannot see.
        """
        judges = [self.settings.evaluator_model]
        results = await asyncio.gather(
            *[
                _judge_side(
                    client=self.client,
                    settings=self.settings,
                    side=f"reference_{index}",
                    response_text=_reference_document(prefix, turns),
                    questions=questions,
                    judge_models=judges,
                )
                for index, (_, _, _, turns) in enumerate(runs, start=1)
            ]
        )
        earned: set[str] = set()
        unreadable = 0
        for _, records in results:
            for record in records:
                if not record.get("parse_ok"):
                    unreadable += 1
                    continue
                earned.update(
                    qid for qid, value in (record.get("answers") or {}).items() if value == "1"
                )
        if unreadable == len(results):
            # no usable verdict from any run: keep the checklist rather than delete it blind
            logger.warning("reference_prune_unreadable runs={} keeping_unpruned", len(results))
            return questions
        kept = [q for q in questions if q["id"] in earned]
        for question in questions:
            if question["id"] not in earned:
                discarded.append(
                    {
                        "stage": "reference_prune",
                        "reason": "no_reference_earned_it",
                        "text": question.get("text", ""),
                        "origin": "content",
                    }
                )
        return kept

    async def _prepare_once(
        self,
        sample: QuestionPrepSample | JudgeSample,
        runs: list[tuple[str, str, bool, list[dict[str, Any]]]],
    ) -> QuestionPrepResult:
        prefix = getattr(sample, "messages", None)
        phase = sample_phase(prefix)
        # what the extractor calls the TASK: the system prompt, the problem, and the conversation
        # that already ran. All of it is free to the candidate, which is why validate_vector drops
        # a milestone whose evidence comes from it.
        task = "\n\n".join(str(m.get("content") or "") for m in (prefix or []))
        references = [text for text, _, _, _ in runs]
        made_edit = any(edit for _, _, edit, _ in runs)
        discarded: list[dict[str, str]] = []

        trajectories = [
            {"run": index, "steps": _steps_from_turns(turns)}
            for index, (_, _, _, turns) in enumerate(runs, start=1)
        ]
        (
            milestones,
            dropped,
            provider,
            readings_held,
            milestones_emitted,
        ) = await self._extract_validated_vector(
            task,
            references,
            int(getattr(sample, "assistant_turns", 0) or self.settings.sota_trajectory_turns),
            trajectories,
            problem=str(getattr(sample, "prompt", "") or ""),
        )
        discarded.extend(dropped)
        if not milestones:
            raise QuestionScoringUnavailable(
                f"no milestone survived validation in any of {readings_held} reading(s)"
            )

        questions, thin = await self._write_ladders(
            milestones,
            {
                int(t["run"]): [str(step.get("assistant") or "") for step in t["steps"]]
                for t in trajectories
            },
        )
        # build_judge_messages shows the judge id/tag/text/example_bad and nothing else, so the
        # near-miss has to arrive under the name it reads
        for question in questions:
            question["example_bad"] = question.pop("unearned", "")

        questions = filter_reference_leaks(questions, discards=discarded)
        questions, drops = enforce_question_labels(questions, discards=discarded)
        pruned_from = len(questions)
        if self.settings.reference_prune:
            questions = await self._prune_unreachable(questions, runs, prefix, discarded)
        pruned_out = pruned_from - len(questions)
        if len(questions) < QUESTION_FLOOR:
            raise QuestionScoringUnavailable(
                f"evaluator returned {len(questions)}/{QUESTION_FLOOR}+ well-formed questions"
            )
        for position, question in enumerate(questions, start=1):
            question["id"] = f"q_{position:02d}"

        logger.info(
            "reference_ladder sample_id={} runs={} milestones={} readings={} questions={} thin={}",
            sample.sample_id,
            len(runs),
            len(milestones),
            readings_held,
            len(questions),
            len(thin),
        )
        source: dict[str, object] = {
            "provider": provider,
            "model": self.settings.evaluator_model,
            "n_questions": len(questions),
            "question_mode": "milestone_ladder",
            "sample_phase": phase,
            "reference_made_edit": made_edit,
            "reference_runs": len(runs),
            "reference_models": [model for _, model, _, _ in runs],
            "readings_held": readings_held,
            "milestones_emitted": milestones_emitted,
            "milestones_kept": len(milestones),
            "pruned_unreachable": pruned_out,
            "milestones_thin": thin,
            "enforcement_drops": drops,
            "reference_trajectory": references[0] if references else "",
            "reference_trajectories": references,
            "discarded_questions": discarded,
        }
        return QuestionPrepResult(questions=questions, source=source)


class ObservationSimulationService:
    def __init__(
        self,
        settings: JudgeSettings,
        client: JudgeLLMClient,
        repo_context: RepoContextClient | None = None,
    ):
        self.settings = settings
        self.client = client
        self.repo_context = repo_context
        self._observations = ObservationMemo(max_entries=_MEMO_MAX_ENTRIES)

    async def _observe(
        self,
        key: str,
        produce: Callable[[], Awaitable[str]],
    ) -> str:
        return await self._observations.observe(key, produce)

    async def _retry_for_output(
        self,
        request: SimulateObservationRequest,
        command: str,
        fmt: str,
        context_block: str | None,
        transcript: str,
        contract: CommandContract,
        observation: str,
    ) -> str:
        """One more ask when a command that must print came back silent anyway."""
        primary = self.settings.simulation_model or self.settings.evaluator_model
        response = await self.client.complete(
            purpose="simulate",
            model=primary,
            messages=[
                {"role": "system", "content": simulation_system_prompt(fmt, context_block)},
                {"role": "user", "content": f"{transcript}\n\n{MUST_PRINT_RETRY}"},
            ],
            temperature=0.0,
            eval_run_id=request.eval_run_id,
            max_tokens=self.settings.simulation_max_tokens,
            provider=_simulation_provider(self.settings),
            hedge_after_seconds=self.settings.simulation_hedge_seconds or None,
        )
        candidate = (
            ""
            if response.error
            else correct_returncode(
                repair_to_contract(repair_output(response.raw, fmt), fmt, contract),
                fmt,
                command,
            )
        )
        recovered = valid_output(candidate, fmt) and has_content(candidate, fmt)
        logger.info(
            "observation_simulation_must_print_retry eval_run_id={} sample_id={} command={!r} "
            "recovered={}",
            request.eval_run_id,
            request.sample_id,
            command[:80],
            recovered,
        )
        return candidate if recovered else observation

    async def simulate(self, request: SimulateObservationRequest) -> str:
        """Produce the environment's answer to the assistant's command."""
        command = first_bash_block(request.assistant_output)
        fmt = detect_format(request.sample_id, request.messages)
        if not command:
            logger.warning(
                "observation_simulation_no_command eval_run_id={} sample_id={} fmt={} chars={}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                len(request.assistant_output or ""),
            )
            return missing_command_output(fmt)
        absent = absent_tool_output(command)
        if absent is not None:
            body, returncode = absent
            # a refusal reached through a head/tail pipe still exits with the filter's code, so
            # the shape has to be settled here: this path returns before correct_returncode runs
            override = pipeline_returncode_override(command)
            if override is not None:
                returncode = override
            logger.info(
                "observation_simulation_absent_tool eval_run_id={} sample_id={} command={!r}",
                request.eval_run_id,
                request.sample_id,
                command[:80],
            )
            return wrap(body, fmt, returncode=returncode)
        resolved = Grounding(None, None, None, "")
        if self.repo_context is not None:
            resolved = await self.repo_context.context_for(
                request.sample_id, request.assistant_output, request.messages
            )
        context_block = resolved.context
        exact_output = resolved.exact_output
        exact_returncode = resolved.exact_returncode
        state = resolved.state
        if exact_output is not None:
            observation = grounded_observation(
                fmt, exact_output, exact_returncode, command, request.messages
            )
            if observation is not None:
                # a computed status still passes the shape check: only the last stage of a
                # pipeline owns the exit code, and a read failure fixes it regardless
                observation = correct_returncode(observation, fmt, command)
                logger.info(
                    "observation_simulation_exact eval_run_id={} sample_id={} fmt={} chars={}",
                    request.eval_run_id,
                    request.sample_id,
                    fmt,
                    len(observation),
                )
                return observation
            logger.info(
                "observation_simulation_exact_unwrapped eval_run_id={} sample_id={} fmt={} "
                "rc={} command={!r}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                exact_returncode,
                command[:80],
            )
        if not state:
            return await self._simulate_uncached(
                request, command, fmt, context_block, exact_output, exact_returncode
            )
        key = hashlib.sha1(
            "\0".join((request.sample_id, fmt, state, command)).encode("utf-8", "replace")
        ).hexdigest()
        return await self._observe(
            key,
            lambda: self._simulate_uncached(
                request, command, fmt, context_block, exact_output, exact_returncode
            ),
        )

    async def _simulate_uncached(
        self,
        request: SimulateObservationRequest,
        command: str,
        fmt: str,
        context_block: str | None,
        exact_output: str | None,
        exact_returncode: int | None = None,
    ) -> str:
        computed = bool(context_block) and context_block.lstrip().startswith(COMPUTED_BLOCK_MARKER)
        transcript = (
            f"$ {command}"
            if computed
            else _simulation_transcript(
                messages=request.messages,
                prompt=request.prompt,
                assistant_output=request.assistant_output,
            )
        )
        require_content = requires_output(command)
        contract = command_contract(command)
        primary = self.settings.simulation_model or self.settings.evaluator_model
        fallback_model = self.settings.evaluator_model
        attempts: list[tuple[str, int, dict[str, Any] | None]] = []
        if primary != fallback_model:
            sim_provider = _simulation_provider(self.settings)
            order = (sim_provider or {}).get("order") or []
            rungs = [
                {**sim_provider, "order": order[i:] + order[:i]} for i in range(len(order))
            ] or [sim_provider]
            attempts = [
                (primary, self.settings.simulation_loop_reruns + 1, rung, index > 0)
                for index, rung in enumerate(rungs)
            ]
            attempts.append((fallback_model, 1, _evaluator_provider(self.settings), True))
        else:
            attempts = [
                (
                    primary,
                    self.settings.simulation_loop_reruns + 1,
                    _evaluator_provider(self.settings),
                    False,
                )
            ]

        observation = ""
        best_rank = -1
        for model, tries, provider_block, or_only in attempts:
            capped = model == primary and primary != fallback_model
            messages = [
                {
                    "role": "system",
                    "content": simulation_system_prompt(fmt, context_block),
                },
                {"role": "user", "content": transcript},
            ]
            # one parse attempt per rung: the ladder itself is the retry mechanism, and
            # every extra in-rung attempt lands on the turn barrier's critical path
            capped_kwargs = {"parse_retries": 1, "retry_count": 1} if capped else {}
            for attempt in range(tries):
                response = await self.client.complete(
                    purpose="simulate",
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    eval_run_id=request.eval_run_id,
                    max_tokens=self.settings.simulation_max_tokens,
                    provider=provider_block,
                    force_openrouter=or_only,
                    hedge_after_seconds=self.settings.simulation_hedge_seconds or None,
                    accept=lambda raw: _usable_simulation_output(
                        repair_to_contract(repair_output(raw, fmt), fmt, contract),
                        fmt,
                        require_content=require_content,
                        contract=contract,
                        command=command,
                    ),
                    **capped_kwargs,
                )
                if response.error:
                    if model != fallback_model:
                        break
                    raise ObservationSimulationUnavailable(response.error)
                candidate = correct_returncode(
                    repair_to_contract(repair_output(response.raw, fmt), fmt, contract),
                    fmt,
                    command,
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
                            request.eval_run_id,
                            request.sample_id,
                            primary,
                            model,
                        )
                    break
                logger.warning(
                    "observation_simulation_unusable eval_run_id={} sample_id={} model={} "
                    "provider={} attempt={}/{} reason={} kept_rank={}",
                    request.eval_run_id,
                    request.sample_id,
                    model,
                    ((provider_block or {}).get("order") or ["auto"])[0],
                    attempt + 1,
                    tries,
                    _unusable_reason(
                        candidate,
                        fmt,
                        require_content=require_content,
                        contract=contract,
                        command=command,
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
        observation = canonical_empty(observation, fmt)
        if diagnostic := misdiagnosed_sed(command, observation):
            logger.info(
                "observation_simulation_sed_rediagnosed eval_run_id={} sample_id={}: {!r}",
                request.eval_run_id,
                request.sample_id,
                diagnostic,
            )
            observation = wrap(diagnostic, fmt, returncode=1)
        if echoed_command(command, observation):
            logger.warning(
                "observation_simulation_echoed eval_run_id={} sample_id={} command={!r}",
                request.eval_run_id,
                request.sample_id,
                command[:80],
            )
            observation = empty_output(fmt)
        if exact_output is None:
            assistant = [
                str(m.get("content") or "")
                for m in (request.messages or [])
                if str(m.get("role") or "").lower() == "assistant"
            ]
            phantom = (
                not any_shows_work(assistant) and claims_tracked_change(command, observation)
            ) or any(
                not named_in_removal(assistant, [first_bash_command(a) for a in assistant], path)
                for path in deleted_files(command, observation)
            )
            if phantom:
                logger.warning(
                    "observation_simulation_phantom_change eval_run_id={} sample_id={} "
                    "command={!r}",
                    request.eval_run_id,
                    request.sample_id,
                    command[:80],
                )
                observation = without_tracked_changes(observation, fmt)
        observation = renumbered_view(command, observation)
        if (
            require_content
            and exact_output is None
            and is_file_read(command)
            and not has_content(observation, fmt)
        ):
            observation = await self._retry_for_output(
                request, command, fmt, context_block, transcript, contract, observation
            )
        if not valid_output(observation, fmt):
            fallback = (
                wrap(exact_output, fmt, returncode=exact_returncode or 0)
                if exact_output
                else empty_output(fmt)
            )
            logger.warning(
                "observation_simulation_invalid_format eval_run_id={} sample_id={} fmt={} "
                "exact={} fallback={!r}",
                request.eval_run_id,
                request.sample_id,
                fmt,
                bool(exact_output),
                fallback[:120],
            )
            return fallback
        if exact_output is not None:
            corrected = with_body(observation, fmt, exact_output)
            if corrected != observation:
                logger.info(
                    "observation_simulation_body_corrected eval_run_id={} sample_id={} fmt={} "
                    "chars={}->{}",
                    request.eval_run_id,
                    request.sample_id,
                    fmt,
                    len(observation),
                    len(corrected),
                )
            return corrected
        if output_expectation(command) == NOT_DERIVABLE and not has_content(observation, fmt):
            body, returncode = no_output_notice(command)
            logger.info(
                "observation_simulation_no_output_notice eval_run_id={} sample_id={} rc={} "
                "command={!r}",
                request.eval_run_id,
                request.sample_id,
                returncode,
                command[:80],
            )
            return wrap(body, fmt, returncode=returncode)
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
                request.eval_run_id,
                prep_id,
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
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
        client = JudgeLLMClient(settings)
        app.state.eval_client = client
        repo_context = RepoContextClient(settings) if settings.repo_context_url else None
        app.state.repo_context_client = repo_context
        app.state.observation_service = ObservationSimulationService(settings, client, repo_context)
        app.state.question_service = QuestionService(
            settings,
            client,
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
            client = JudgeLLMClient(settings)
            app.state.eval_client = client
            repo_context = RepoContextClient(settings) if settings.repo_context_url else None
            app.state.repo_context_client = repo_context
            app.state.observation_service = ObservationSimulationService(
                settings, client, repo_context
            )
            app.state.question_service = QuestionService(
                settings,
                client,
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
            raise HTTPException(
                status_code=400, detail=f"unsupported judge model(s): {', '.join(unknown)}"
            )
        client: JudgeLLMClient = app.state.eval_client
        try:
            records = await _score_samples(
                client=client, request=request, settings=settings, prep_store=prep_store()
            )
        except Exception as exc:
            _notify(
                settings,
                request,
                severity="ERROR",
                message="Scoring failed",
                fault_code="scoring_failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            logger.exception(
                f"[judge-api] scoring failed eval_run={request.eval_run_id} batch={request.batch_id}: {exc}"  # noqa: E501
            )
            raise HTTPException(status_code=502, detail=f"scoring failed: {exc}")
        summary = aggregate_scores(records, min_valid_fraction=settings.min_valid_fraction)
        if summary.get("state") != "succeeded":
            _notify(
                settings,
                request,
                severity="WARNING",
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
        request.eval_run_id,
        request.batch_id,
        sample.sample_id,
        reason,
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


def _role_violation(raw: str) -> bool:
    text = raw or ""
    return bool(ROLE_MARKER_RE.search(text)) or text.count("<returncode>") > 1


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
    if _usable_simulation_output(raw, fmt, require_content=require_content, contract=contract):
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
    command: str = "",
) -> bool:
    return (
        valid_output(raw, fmt)
        and not _role_violation(raw)
        and not _looping_output(raw)
        and not degenerate_observation(raw)
        and not impossible_success(raw, fmt, command)
        and not stuttered_lines(raw)
        and not (command and fabricated_sed_error(command, raw))
        and not (command and fabricated_pip_error(command, raw))
        and (not require_content or has_content(raw, fmt))
        and (contract is None or contract_violation(raw, fmt, contract) is None)
    )


def _unusable_reason(
    raw: str,
    fmt: str,
    *,
    require_content: bool = False,
    contract: CommandContract | None = None,
    command: str = "",
) -> str:
    if not valid_output(raw, fmt):
        return "invalid_format"
    if _role_violation(raw):
        return "role_violation"
    if _looping_output(raw):
        return "looping"
    if degenerate_observation(raw):
        return "degenerate_lines"
    if impossible_success(raw, fmt, command):
        return "shell_error_with_rc_0"
    if reason := stuttered_lines(raw):
        return f"stuttered: {reason}"
    if command and fabricated_sed_error(command, raw):
        return "fabricated_sed_error"
    if command and fabricated_pip_error(command, raw):
        return "fabricated_pip_error"
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
    reason: str,
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
            "explanations": {q["id"]: reason for q in questions},
            "yes_rate": judge_yes_rate(per_judge_answers[model], questions),
            "parse_ok": True,
            "error": None,
            "corrupted": True,
            "corruption_reason": reason,
        }
        for model in judge_models
    ]
    return per_judge_answers, records


def _looped_side(
    *,
    side: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
    verdict: LoopVerdict,
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    explanation = loop_explanation(verdict)
    per_judge_answers: dict[str, dict[str, str | None]] = {
        model: {q["id"]: "0" for q in questions} for model in judge_models
    }
    records = [
        {
            "side": side,
            "judge_model": model,
            "provider": None,
            "answers": per_judge_answers[model],
            "explanations": {q["id"]: explanation for q in questions},
            "yes_rate": judge_yes_rate(per_judge_answers[model], questions),
            "parse_ok": True,
            "error": None,
            "looped": True,
            "loop_reasons": list(verdict.reasons),
            "loop_commands": [
                {
                    "command": entry.command,
                    "count": entry.count,
                    "longest_run": entry.longest_run,
                }
                for entry in verdict.commands
            ],
        }
        for model in judge_models
    ]
    return per_judge_answers, records


async def _judge_side(
    *,
    client: JudgeLLMClient,
    settings: JudgeSettings,
    side: str,
    response_text: str,
    questions: list[dict[str, str]],
    judge_models: list[str],
    repeats: int = 1,
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, Any]]]:
    """Each judge model answers `repeats` times; a question's answer is the majority."""
    question_ids = [q["id"] for q in questions]
    schema = answer_schema(question_ids)
    messages = build_judge_messages(response=response_text, questions=questions)
    repeats = max(1, repeats)
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
            for _ in range(repeats)
        ]
    )
    per_judge_answers: dict[str, dict[str, str | None]] = {}
    records: list[dict[str, Any]] = []
    for index, model in enumerate(judge_models):
        parsed = [
            parse_answers(raw.raw, question_ids)
            for raw in raws[index * repeats : (index + 1) * repeats]
        ]
        held = [
            (a, e, raw)
            for (a, e, ok), raw in zip(parsed, raws[index * repeats : (index + 1) * repeats])
            if ok and not raw.error
        ]
        answers = majority_answers([a for a, _, _ in held]) if held else parsed[0][0]
        explanations = held[0][1] if held else parsed[0][1]
        first = held[0][2] if held else raws[index * repeats]
        per_judge_answers[model] = answers
        records.append(
            {
                "side": side,
                "judge_model": model,
                "provider": first.provider,
                "answers": answers,
                "explanations": explanations,
                "yes_rate": judge_yes_rate(answers, questions),
                "parse_ok": bool(held),
                "error": None if held else first.error,
                "repeats": repeats,
                "repeats_held": len(held),
                "disputed": sum(
                    1 for qid in question_ids if len({a.get(qid) for a, _, _ in held}) > 1
                ),
            }
        )
    return per_judge_answers, records


async def _score_samples(
    *,
    client: JudgeLLMClient,
    request: ScoreBatchRequest,
    settings: JudgeSettings,
    prep_store: QuestionPrepStore,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    completed = 0
    progress_lock = asyncio.Lock()
    logger.info(
        "score_batch_started eval_run_id={} batch_id={} samples={} judges={} prep_id={}",
        request.eval_run_id,
        request.batch_id,
        len(request.samples),
        len(request.judge_models),
        request.category_prep_id or "",
    )

    async def _score_one(sample: JudgeSample) -> dict[str, Any]:
        nonlocal completed
        try:
            return await _score_one_inner(sample)
        except Exception as exc:
            async with progress_lock:
                completed += 1
            logger.warning(
                "score_batch_sample_failed eval_run_id={} batch_id={} completed={}/{} sample_id={} error={}",  # noqa: E501
                request.eval_run_id,
                request.batch_id,
                completed,
                len(request.samples),
                sample.sample_id,
                f"{type(exc).__name__}: {exc}",
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

        async def _side(side: str, response_text: str):
            if is_truncated(response_text):
                return _corrupted_side(
                    side=side,
                    questions=questions,
                    judge_models=request.judge_models,
                    reason="output truncated mid-generation",
                )
            if is_abandoned(response_text):
                return _corrupted_side(
                    side=side,
                    questions=questions,
                    judge_models=request.judge_models,
                    reason="turn unusable after repeated attempts; the benchmark abandons here",
                )
            leak = reserved_token_leak(response_text)
            if leak:
                logger.warning(
                    "score_batch_side_reserved_token eval_run_id={} sample_id={} side={} token={}",
                    request.eval_run_id,
                    sample.sample_id,
                    side,
                    leak,
                )
                return _corrupted_side(
                    side=side,
                    questions=questions,
                    judge_models=request.judge_models,
                    reason=f"reserved template token in output: {leak}",
                )
            looping = loop_verdict_for_document(response_text)
            if looping.looped:
                logger.warning(
                    "score_batch_side_looped eval_run_id={} batch_id={} sample_id={} side={} "
                    "reasons={} n_cmds={} dup_ratio={:.2f} max_run={}",
                    request.eval_run_id,
                    request.batch_id,
                    sample.sample_id,
                    side,
                    "; ".join(looping.reasons),
                    looping.n_cmds,
                    looping.dup_cmd_ratio,
                    looping.max_cmd_run,
                )
                return _looped_side(
                    side=side,
                    questions=questions,
                    judge_models=request.judge_models,
                    verdict=looping,
                )
            return await _judge_side(
                client=client,
                settings=settings,
                side=side,
                response_text=response_text,
                questions=questions,
                judge_models=request.judge_models,
                repeats=int(getattr(settings, "judge_repeats", 1) or 1),
            )

        (king_answers, king_recs), (chal_answers, chal_recs) = await asyncio.gather(
            _side("previous_king", sample.previous_king_output),
            _side("challenger", sample.challenger_output),
        )
        king_score = response_score(king_answers, questions)
        chal_score = response_score(chal_answers, questions)
        king_amputated = amputated_thinking(sample.previous_king_output)
        chal_amputated = amputated_thinking(sample.challenger_output)
        if king_amputated and king_score is not None:
            king_score = round(king_score * AMPUTATED_THINKING_MULTIPLIER, 6)
        if chal_amputated and chal_score is not None:
            chal_score = round(chal_score * AMPUTATED_THINKING_MULTIPLIER, 6)
        king_ok = all(r["parse_ok"] for r in king_recs) and king_score is not None
        chal_ok = all(r["parse_ok"] for r in chal_recs) and chal_score is not None
        scored = king_ok and chal_ok
        async with progress_lock:
            completed += 1
            logger.info(
                "score_batch_sample_done eval_run_id={} batch_id={} completed={}/{} sample_id={} "
                "scored={} king={} chal={} elapsed_s={:.1f}",
                request.eval_run_id,
                request.batch_id,
                completed,
                len(request.samples),
                sample.sample_id,
                scored,
                king_score,
                chal_score,
                time.monotonic() - started_at,
            )
        return {
            "sample_id": sample.sample_id,
            "questions": [{**q, "weight": question_weight(q)} for q in questions],
            "king_amputated_thinking": king_amputated,
            "chal_amputated_thinking": chal_amputated,
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
        request.eval_run_id,
        request.batch_id,
        sum(1 for r in records if r.get("scored")),
        len(records),
        time.monotonic() - started_at,
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
