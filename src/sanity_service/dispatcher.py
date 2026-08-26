from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from loguru import logger

from albedo_config import JudgeSettings, SanitySettings, get_judge_settings, get_sanity_settings
from albedo_eval_service.remote.dataset import format_messages
from albedo_eval_service.shared.edit_detection import REMOVAL_RE as _REMOVAL_RE
from albedo_eval_service.shared.edit_detection import named_in_removal
from albedo_eval_service.shared.observation_format import (
    MAX_CONSECUTIVE_BAD_TURNS,
    canonical_empty,
    claims_tracked_change,
    command_contract,
    degenerate_observation,
    deleted_files,
    detect_format,
    echoed_command,
    empty_output,
    has_content,
    leaked_turn,
    prints_nothing_on_success,
    renumbered_view,
    repair_output,
    repair_to_contract,
    requires_output,
    retry_feedback,
    silent_observation,
    stuttered_lines,
    unusable_turn,
    valid_output,
    without_tracked_changes,
    wrap,
)
from albedo_eval_service.shared.observation_memo import ObservationMemo
from albedo_eval_service.shared.sed_check import fabricated_sed_error, misdiagnosed_sed
from albedo_eval_service.shared.submit_protocol import (
    assign_submit,
    command_for,
    first_bash_command,
    rewrite_messages,
)
from albedo_eval_service.simulator.prompt_simulator import (
    COMPLETE_MARKER,
    DEGENERATE_RETRY,
    MUST_PRINT_RETRY,
    simulation_system_prompt,
)
from sanity_remote.models import SanityRunRequest
from sanity_service.chain import (
    _EDIT_RE as _CHAIN_EDIT_RE,
)
from sanity_service.chain import (
    META_LEAK_RE,
    SUBMIT_NUDGE,
    amputated_thinking,
    empty_submit_count,
    followup_instruction,
    generate_followup,
    generate_microtask,
    generate_nudge,
    generate_rejection,
    malformed_structure,
    micro_instruction,
    micro_target_touched,
    segment_has_edit,
    should_reject,
    unread_edited_files,
)
from sanity_service.dataset import sample_prompts
from sanity_service.db import ClaimedPreEval, PreEvalRepository
from sanity_service.judge_panel import make_client
from sanity_service.llm_check import SampleInput, run_gate
from sanity_service.remote_client import SanityRemoteClient
from sanity_service.tail_check import run_tail_check
from sanity_service.uploads import put_sanity_fault

_CANONICAL_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
)
_SED_ERROR_RE = re.compile(r"^\s*sed:\s", re.MULTILINE)
_BASH_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)\s*\n.*?```|<([a-z_]*bash[a-z_]*)>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

MAX_CONSECUTIVE_DEGENERATE_OBSERVATIONS = 3
_DEGENERATE_RETRY_TEMPERATURE = 0.3
MAX_CONSECUTIVE_SILENT_OBSERVATIONS = 6


@dataclass
class _TrajectoryState:
    sample_id: str
    prompt: str
    messages: list[dict[str, str]]
    turns: list[dict[str, Any]]
    stopped: bool = False
    error: str = ""
    heuristic_reason: str = ""
    segment: str = "context"
    segment_index: int = 0
    submit_clause: str = ""
    submit_marker: str = ""
    rewrite_mode: str = ""
    micro: dict[str, str] | None = None
    nudged_at: int = 0
    submits: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    retry_reason: str = ""
    consecutive_bad_turns: int = 0
    consecutive_silent_observations: int = 0
    last_followup: str = ""
    observation_memo: ObservationMemo = dataclasses.field(default_factory=ObservationMemo)


class SanityDispatcher:
    def __init__(self, *, settings: SanitySettings, repository: PreEvalRepository) -> None:
        self.settings = settings
        self.repository = repository

    def _build_request(
        self, submission: dict[str, Any], host: Any, attempt_id: UUID
    ) -> SanityRunRequest:
        samples = sample_prompts(
            seed=str(submission["block_hash"]),
            n=self.settings.sample_count,
            manifest_path=self.settings.dataset_manifest_path,
            manifest_hash=self.settings.dataset_manifest_hash,
            dataset_root=self.settings.dataset_root,
        )
        return SanityRunRequest(
            run_id=str(attempt_id),
            model_uri=submission["model_uri"],
            digest=submission.get("model_hash") or "",
            prompts=[s.prompt for s in samples],
            sample_ids=[s.sample_id or f"sanity-sample:{i}" for i, s in enumerate(samples)],
            prompt_messages=[
                s.messages or [{"role": "user", "content": s.prompt}] for s in samples
            ],
            assistant_turns=self.settings.trajectory_assistant_turns,
            gen_max_tokens=self.settings.gen_max_tokens,
        )

    def claim_once(self) -> ClaimedPreEval | None:
        return self.repository.claim_next_pre_eval(
            worker_id=self.settings.worker_id,
            lease_seconds=self.settings.lease_seconds,
            request_builder=self._build_request,
        )

    async def dispatch_once(self) -> bool:
        claimed = self.claim_once()
        if not claimed:
            logger.debug("[sanity-dispatch] no claimable pre-eval")
            return False
        logger.info(
            "[sanity-dispatch] claimed submission={} digest={:.16} host={}",
            claimed.submission_id,
            claimed.request.digest,
            claimed.remote_host.id,
        )
        client = SanityRemoteClient(
            base_url=claimed.remote_host.base_url,
            auth_token=self.settings.remote_auth_token,
            timeout_seconds=self.settings.remote_event_timeout_seconds,
        )
        try:
            await client.ready()
            if claimed.request.assistant_turns > 1:
                result = await self._run_multiturn(client, claimed)
            else:
                result = await self._run_remote_request(client, claimed.request, claimed)
            await self._complete(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                prompts=list(claimed.request.prompts),
                result=result,
            )
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                logger.warning(
                    "[sanity-dispatch] worker busy, releasing claim submission={} digest={:.16}: {}",  # noqa: E501
                    claimed.submission_id,
                    claimed.request.digest,
                    exc,
                )
                self.repository.release_pre_eval_attempt(
                    submission_id=claimed.submission_id,
                    attempt_id=claimed.attempt_id,
                    fault_message=str(exc),
                )
                return True
            logger.warning(
                "[sanity-dispatch] worker HTTP error submission={} digest={:.16}: {}",
                claimed.submission_id,
                claimed.request.digest,
                exc,
            )
            self.repository.mark_pre_eval_failed(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                fault_class="INFRA_FAULT",
                fault_code="worker_unreachable",
                fault_message=str(exc),
                retryable=True,
            )
            return True
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning(
                "[sanity-dispatch] worker unreachable submission={} digest={:.16}: {}",
                claimed.submission_id,
                claimed.request.digest,
                exc,
            )
            self.repository.mark_pre_eval_failed(
                submission_id=claimed.submission_id,
                attempt_id=claimed.attempt_id,
                repo=claimed.request.model_uri,
                digest=claimed.request.digest,
                fault_class="INFRA_FAULT",
                fault_code="worker_unreachable",
                fault_message=str(exc),
                retryable=True,
            )
            return True
        finally:
            await client.aclose()

    async def _run_remote_request(
        self,
        client: SanityRemoteClient,
        request: SanityRunRequest,
        claimed: ClaimedPreEval,
    ) -> dict[str, Any]:
        start = await client.start_run(request)
        run_id = str(start.get("run_id") or request.run_id)
        self.repository.heartbeat_attempt(
            attempt_id=claimed.attempt_id,
            lease_seconds=self.settings.lease_seconds,
        )
        return await self._follow_until_result(
            client,
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            run_id=run_id,
        )

    async def _run_multiturn(
        self, client: SanityRemoteClient, claimed: ClaimedPreEval
    ) -> dict[str, Any]:
        request = claimed.request
        turn_count = max(1, int(request.assistant_turns))
        states = _trajectory_states(request)
        await _inject_microtasks(states)
        kept_warm = False
        decided_early = False
        try:
            for turn_index in range(turn_count):
                active = [
                    state
                    for state in states
                    if not state.stopped and not state.error and not state.heuristic_reason
                ]

                if not self.settings.consensus and any(s.heuristic_reason for s in states):
                    decided_early = True
                    break
                if not active:
                    break

                turn_request = request.model_copy(
                    update={
                        "run_id": f"{claimed.attempt_id}:turn-{turn_index + 1}",
                        "prompts": [state.prompt for state in active],
                        "sample_ids": [state.sample_id for state in active],
                        "prompt_messages": [state.messages for state in active]
                        if turn_index == 0
                        else None,
                        "teardown_after_run": False,
                    }
                )
                kept_warm = True
                logger.info(
                    "[sanity-dispatch] trajectory turn {}/{} samples={}",
                    turn_index + 1,
                    turn_count,
                    len(active),
                )
                result = await self._run_remote_request(client, turn_request, claimed)
                if result.get("state") == "failed":
                    return result
                _apply_turn_result(active, result)
                for _ in range(MAX_CONSECUTIVE_BAD_TURNS - 1):
                    redo = [state for state in active if state.retry_reason]
                    if not redo:
                        break
                    redo_request = request.model_copy(
                        update={
                            "run_id": (
                                f"{claimed.attempt_id}:turn-{turn_index + 1}"
                                f"-retry{redo[0].consecutive_bad_turns}"
                            ),
                            "prompts": [state.prompt for state in redo],
                            "sample_ids": [state.sample_id for state in redo],
                            "prompt_messages": None,
                            "teardown_after_run": False,
                        }
                    )
                    kept_warm = True
                    redo_result = await self._run_remote_request(client, redo_request, claimed)
                    if redo_result.get("state") == "failed":
                        return redo_result
                    _apply_turn_result(redo, redo_result)
                if turn_index == turn_count - 1:
                    break
                await _append_observations(active, str(claimed.attempt_id), turn_index + 1)
                if turn_index >= turn_count - 8 and (turn_count - turn_index) % 4 == 0:
                    await _inject_submit_nudges(active)
            if not decided_early:
                _run_chain_checks(states, turn_count)
                await run_tail_check(states)
            return _trajectory_result(str(claimed.attempt_id), states, turn_count)
        finally:
            if kept_warm:
                try:
                    await client.teardown()
                except Exception as exc:
                    logger.warning("[sanity-dispatch] remote teardown failed: {}", exc)

    async def _follow_until_result(
        self, client: SanityRemoteClient, *, submission_id: UUID, attempt_id: UUID, run_id: str
    ) -> dict[str, Any]:
        seen = 0
        while True:
            events = [event async for event in client.iter_events(run_id)]
            for event in events[seen:]:
                ev_type = event.get("type", "?")
                logger.info(
                    "[sanity-dispatch] worker event={} run={} submission={:.8}",
                    ev_type,
                    run_id,
                    str(submission_id),
                )
                self.repository.record_remote_event(
                    submission_id=submission_id, attempt_id=attempt_id, event=event
                )
                if event.get("type") == "result":
                    logger.info(
                        "[sanity-dispatch] result received run={} state={} submission={:.8}",
                        run_id,
                        event.get("state"),
                        str(submission_id),
                    )
                    self.repository.heartbeat_attempt(
                        attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds
                    )
                    return event
            seen = max(seen, len(events))
            self.repository.heartbeat_attempt(
                attempt_id=attempt_id, lease_seconds=self.settings.lease_seconds
            )
            status = await client.get_run(run_id)
            if status.get("type") == "result" or status.get("state") in {"succeeded", "failed"}:
                if status.get("type") == "result":
                    self.repository.record_remote_event(
                        submission_id=submission_id, attempt_id=attempt_id, event=status
                    )
                return status
            await asyncio.sleep(self.settings.remote_event_poll_seconds)

    async def _complete(
        self,
        *,
        submission_id: UUID,
        attempt_id: UUID,
        repo: str,
        digest: str,
        prompts: list[str],
        result: dict[str, Any],
    ) -> None:
        logger.info(
            "[sanity-dispatch] completing submission={:.8} digest={:.16} state={}",
            str(submission_id),
            digest,
            result.get("state"),
        )
        if result.get("state") == "failed":
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code=result.get("fault_code", "worker_fault"),
                fault_message=result.get("fault_message", ""),
                retryable=bool(result.get("retryable", True)),
            )
            return

        responses = list(result.get("responses", []))
        heuristics = list(result.get("heuristics", []))
        submit_commands = list(result.get("submit_commands", []))
        samples = [
            SampleInput(
                prompt=prompts[i] if i < len(prompts) else "",
                response=responses[i],
                heuristic_passed=bool(heuristics[i].get("passed", True))
                if i < len(heuristics)
                else True,
                heuristic_reason=heuristics[i].get("reason", "") if i < len(heuristics) else "",
                heuristic_infra=bool(heuristics[i].get("infra")) if i < len(heuristics) else False,
                submit_command=submit_commands[i] if i < len(submit_commands) else "",
            )
            for i in range(len(responses))
        ]
        client = make_client()
        try:
            gate = await run_gate(
                samples,
                client,
                consensus=self.settings.consensus,
                skip_viability=self.settings.skip_viability,
            )
        except Exception as exc:
            logger.exception(
                f"[sanity-dispatch] judge gate failed submission={submission_id}: {exc}"
            )
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code="judges_failed",
                fault_message=str(exc),
                retryable=True,
            )
            return
        finally:
            await client.aclose()

        if gate.infra_fault:
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="INFRA_FAULT",
                fault_code="judges_unavailable",
                fault_message=gate.reason,
                retryable=True,
            )
        elif gate.passed:
            self.repository.mark_pre_eval_passed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                responses=responses,
                reason=gate.reason,
                timing={},
            )
        else:
            detail = {
                "submission_id": str(submission_id),
                "repo": repo,
                "digest": digest,
                "fault_code": str(gate.llm_gate),
                "reason": gate.reason,
                "decision_mode": gate.decision_mode,
                "gate": dataclasses.asdict(gate),
                "prompts": prompts,
                "responses": responses,
                "checked_at": datetime.now(UTC).isoformat(),
            }
            artifact_uri = put_sanity_fault(str(submission_id), digest, detail)
            self.repository.mark_pre_eval_failed(
                submission_id=submission_id,
                attempt_id=attempt_id,
                repo=repo,
                digest=digest,
                fault_class="MINER_FAULT",
                fault_code=str(gate.llm_gate),
                fault_message=gate.reason,
                retryable=False,
                responses=responses,
                artifact_uri=artifact_uri,
            )

    async def reconcile_once(self, *, limit: int = 10, follow_timeout: float = 50.0) -> int:
        in_flight = self.repository.list_reconcilable_pre_eval(limit=limit)
        logger.info("[sanity-dispatch] reconcile found={}", len(in_flight))
        if not in_flight:
            return 0
        reconciled = 0
        for active in in_flight:
            client = SanityRemoteClient(
                base_url=active.remote_host.base_url,
                auth_token=self.settings.remote_auth_token,
                timeout_seconds=self.settings.remote_event_timeout_seconds,
            )
            try:
                result = await asyncio.wait_for(
                    self._follow_until_result(
                        client,
                        submission_id=active.submission_id,
                        attempt_id=active.attempt_id,
                        run_id=active.run_id,
                    ),
                    timeout=follow_timeout,
                )
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "[sanity-dispatch] reconcile skipped submission={} run={}: {}",
                    active.submission_id,
                    active.run_id,
                    exc,
                )
                continue
            finally:
                await client.aclose()
            try:
                await self._complete(
                    submission_id=active.submission_id,
                    attempt_id=active.attempt_id,
                    repo=active.repo,
                    digest=active.digest,
                    prompts=active.prompts,
                    result=result,
                )
            except Exception as exc:
                logger.exception(
                    "[sanity-dispatch] reconcile _complete failed submission={}: {}",
                    active.submission_id,
                    exc,
                )
                continue
            reconciled += 1
        return reconciled

    async def run_forever(self) -> None:
        while True:
            try:
                did_work = await self.dispatch_once()
                if not did_work:
                    logger.debug(
                        "[sanity-dispatch] idle — sleeping {}s", self.settings.dispatch_poll_seconds
                    )
                    await asyncio.sleep(self.settings.dispatch_poll_seconds)
            except Exception as exc:
                logger.exception(
                    "[sanity-dispatch] unhandled error in dispatch loop, retrying in {}s: {}",
                    self.settings.dispatch_poll_seconds,
                    exc,
                )
                await asyncio.sleep(self.settings.dispatch_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Albedo sanity pre-eval dispatcher.")
    parser.add_argument(
        "--once", action="store_true", help="Claim and dispatch at most one pre-eval."
    )
    parser.add_argument(
        "--sweep-abandoned",
        action="store_true",
        help="Reclaim expired pre-eval attempts.",
    )
    parser.add_argument(
        "--reconcile-running",
        action="store_true",
        help="Replay in-flight pre-eval runs.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max active runs to reconcile.")
    args = parser.parse_args()

    settings = get_sanity_settings()
    dispatcher = SanityDispatcher(
        settings=settings,
        repository=PreEvalRepository(
            settings.database_url,
            min_free_gpus=settings.min_free_gpus,
            max_retry_count=settings.max_retry_count,
        ),
    )
    if args.sweep_abandoned:
        logger.info(
            "[sanity-dispatch] abandoned={}",
            dispatcher.repository.sweep_abandoned_pre_eval(worker_id=settings.worker_id),
        )
    elif args.reconcile_running:
        try:
            logger.info(
                "[sanity-dispatch] reconciled={}",
                asyncio.run(dispatcher.reconcile_once(limit=args.limit)),
            )
        except KeyboardInterrupt:
            logger.info("[sanity-dispatch] reconciler interrupted by signal, exiting cleanly")
    elif args.once:
        asyncio.run(dispatcher.dispatch_once())
    else:
        asyncio.run(dispatcher.run_forever())


def _trajectory_states(request: SanityRunRequest) -> list[_TrajectoryState]:
    sample_ids = request.sample_ids or [f"sanity-sample:{i}" for i in range(len(request.prompts))]
    prompt_messages = request.prompt_messages or []
    states: list[_TrajectoryState] = []
    for i, prompt in enumerate(request.prompts):
        messages = (
            prompt_messages[i]
            if i < len(prompt_messages)
            else [{"role": "user", "content": prompt}]
        )
        clean_messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
            for message in messages
        ]
        sample_id = sample_ids[i] if i < len(sample_ids) else f"sanity-sample:{i}"
        marker, command = assign_submit(sample_id, salt=str(request.run_id))
        rewritten, rewrite_mode = rewrite_messages(clean_messages, command)
        if rewrite_mode == "failed":
            logger.warning(
                "[sanity-dispatch] submit rewrite failed sample={} command={}", sample_id, command
            )
            marker = COMPLETE_MARKER
            command = command_for(COMPLETE_MARKER, "gitdiff")
        states.append(
            _TrajectoryState(
                sample_id=sample_id,
                prompt=prompt,
                messages=rewritten,
                turns=[
                    {"role": message["role"], "content": message["content"]}
                    for message in rewritten
                ],
                submit_clause=command,
                submit_marker=marker,
                rewrite_mode=rewrite_mode,
            )
        )
    return states


async def _inject_microtasks(states: list[_TrajectoryState]) -> None:
    settings = get_judge_settings()
    client = make_client(settings)
    try:
        for state in states:
            clause = state.submit_clause
            try:
                state.micro = await generate_microtask(client, settings, state, clause)
            except Exception as exc:
                state.error = f"microtask_generation_failed: {exc}"
                continue
            instruction = micro_instruction(state.micro, clause)
            state.segment = "micro"
            state.messages.append({"role": "user", "content": instruction})
            state.turns.append(
                {"role": "user", "content": instruction, "segment": "micro", "injected": True}
            )
            state.prompt = format_messages(
                state.messages,
                tokenizer_path=str(_CANONICAL_TOKENIZER_PATH),
                enable_thinking=True,
            )
            logger.info(
                "[sanity-dispatch] microtask sample={} marker={} rewrite={} target={}:{}",
                state.sample_id,
                state.submit_marker,
                state.rewrite_mode,
                state.micro.get("file"),
                state.micro.get("function"),
            )
    finally:
        await client.aclose()


async def _inject_submit_nudges(states: list[_TrajectoryState]) -> None:
    pending = [s for s in states if not (s.error or s.heuristic_reason or s.submits)]
    if not pending:
        return
    settings = get_judge_settings()
    client = make_client(settings)
    try:
        nudges = await asyncio.gather(
            *[generate_nudge(client, settings, s.submit_clause) for s in pending],
            return_exceptions=True,
        )
    finally:
        await client.aclose()
    for state, nudge in zip(pending, nudges, strict=False):
        if isinstance(nudge, Exception):
            nudge = SUBMIT_NUDGE.format(clause=state.submit_clause)
        state.messages.append({"role": "user", "content": nudge})
        state.turns.append(
            {
                "role": "user",
                "content": nudge,
                "segment": state.segment,
                "injected": True,
                "nudge": True,
            }
        )
        state.prompt = format_messages(
            state.messages, tokenizer_path=str(_CANONICAL_TOKENIZER_PATH), enable_thinking=True
        )
        state.nudged_at = state.nudged_at or len([t for t in state.turns if t.get("score_target")])
        logger.info("[sanity-dispatch] submit nudge injected sample={}", state.sample_id)


def _run_chain_checks(states: list[_TrajectoryState], turn_count: int) -> None:
    healthy = [s for s in states if not s.error]
    if healthy and not any(not sub.get("post_nudge") for s in healthy for sub in s.submits):
        for state in healthy:
            state.heuristic_reason = state.heuristic_reason or (
                "chain: no unprompted submission on any sample"
            )
        logger.warning("[sanity-dispatch] no unprompted submission across chain samples")
    for state in states:
        if state.error or state.heuristic_reason:
            continue
        if not state.submits and not _has_edited(state):
            state.heuristic_reason = f"chain: no submission and no edits in {turn_count} turns"
        elif state.micro and not micro_target_touched(state, state.micro):
            state.heuristic_reason = (
                f"chain: micro-task submitted without touching {state.micro.get('file')}"
            )
        elif empty_submit_count(state, state.submit_marker) > max(1, 2 * _edit_turns(state)):
            state.heuristic_reason = "chain: repeated submissions without doing any work"
        elif amputated_thinking(state):
            state.heuristic_reason = "chain: reasoning absent on majority of turns"
        elif unread := unread_edited_files(state):
            state.heuristic_reason = f"chain: files edited without being read first: {unread[:3]}"
        elif malformed_structure(state):
            state.heuristic_reason = f"chain: {malformed_structure(state)}"
        if state.heuristic_reason:
            logger.warning("[sanity-dispatch] {} {}", state.sample_id, state.heuristic_reason)


def _apply_turn_result(states: list[_TrajectoryState], result: dict[str, Any]) -> None:
    responses = list(result.get("responses", []))
    heuristics = list(result.get("heuristics", []))
    for i, state in enumerate(states):
        state.retry_reason = ""
        if i >= len(responses):
            state.error = "missing_generation_response"
            continue
        response = str(responses[i] or "")
        reason = (
            str(heuristics[i].get("reason") or "heuristic failed")
            if i < len(heuristics) and not bool(heuristics[i].get("passed", True))
            else unusable_turn(response)
        )
        failed = bool(reason)
        if failed and not _has_bash_command(response):
            state.consecutive_bad_turns += 1
            if state.consecutive_bad_turns >= MAX_CONSECUTIVE_BAD_TURNS:
                state.turns.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "score_target": True,
                        "segment": state.segment,
                    }
                )
                state.heuristic_reason = (
                    f"{reason} on {state.consecutive_bad_turns} consecutive turns"
                )
                continue
            state.retry_reason = reason
            feedback = retry_feedback(reason)
            state.messages.append({"role": "user", "content": feedback})
            state.turns.append(
                {"role": "user", "content": feedback, "injected": True, "retry_feedback": True}
            )
            state.prompt = format_messages(
                state.messages,
                tokenizer_path=str(_CANONICAL_TOKENIZER_PATH),
                enable_thinking=True,
            )
            logger.warning(
                "[sanity-dispatch] {} bad turn ({}/{}): {} — re-asking",
                state.sample_id,
                state.consecutive_bad_turns,
                MAX_CONSECUTIVE_BAD_TURNS,
                reason,
            )
            continue
        state.consecutive_bad_turns = 0
        state.turns.append(
            {
                "role": "assistant",
                "content": response,
                "score_target": True,
                "segment": state.segment,
            }
        )


async def _append_observations(
    states: list[_TrajectoryState], eval_run_id: str, turn_index: int
) -> None:
    active = []
    submitted = []
    for state in states:
        if state.error or state.stopped or state.heuristic_reason:
            continue
        assistant_output = str(state.turns[-1].get("content") or "")
        if state.submit_marker and state.submit_marker in first_bash_command(assistant_output):
            submitted.append((state, assistant_output))
        elif not _has_bash_command(assistant_output):
            _append_observation(
                state, _missing_command_observation(state.sample_id, state.messages)
            )
        else:
            active.append((state, assistant_output))
    if not active and not submitted:
        return

    settings = get_judge_settings()
    client = make_client(settings)
    try:
        results = await asyncio.gather(
            *[
                _simulate_observation(
                    client=client,
                    settings=settings,
                    eval_run_id=eval_run_id,
                    state=state,
                    assistant_output=assistant_output,
                )
                for state, assistant_output in active
            ],
            return_exceptions=True,
        )
        rejected = {id(s) for s, _ in submitted if should_reject(s)}
        followups = await asyncio.gather(
            *[
                (generate_rejection if id(state) in rejected else generate_followup)(
                    client, settings, state, assistant_output
                )
                for state, assistant_output in submitted
            ],
            return_exceptions=True,
        )
    finally:
        await client.aclose()

    for (state, assistant_output), result in zip(active, results, strict=False):
        if isinstance(result, Exception):
            state.error = f"{type(result).__name__}: {result}"
            continue
        _append_observation(state, result)
        quiet_by_design = prints_nothing_on_success(first_bash_command(assistant_output))
        state.consecutive_silent_observations = (
            state.consecutive_silent_observations + 1
            if silent_observation(result) and not quiet_by_design
            else 0
        )
        if state.consecutive_silent_observations >= MAX_CONSECUTIVE_SILENT_OBSERVATIONS:
            state.error = (
                f"simulator answered {state.consecutive_silent_observations} consecutive "
                f"commands with no output"
            )
            logger.warning(
                "[sanity-dispatch] {} dropped as infra at turn {}: {}",
                state.sample_id,
                turn_index,
                state.error,
            )
            continue
        state.prompt = format_messages(
            state.messages,
            tokenizer_path=str(_CANONICAL_TOKENIZER_PATH),
            enable_thinking=True,
        )
    for (state, assistant_output), followup in zip(submitted, followups, strict=False):
        text = "" if isinstance(followup, Exception) else str(followup)
        if id(state) in rejected and text:
            _reject_submission(state, assistant_output, text, turn_index=turn_index)
        else:
            _advance_segment(state, assistant_output, text, turn_index=turn_index)
    logger.info(
        "[sanity-dispatch] simulated observations turn={} samples={} submits={}",
        turn_index,
        len(active),
        len(submitted),
    )


def _reject_submission(
    state: _TrajectoryState, assistant_output: str, rejection: str, *, turn_index: int
) -> None:
    state.submits.append(
        {
            "turn": turn_index,
            "segment": state.segment,
            "rejected": True,
            "format_ok": state.submit_clause.split("&&")[0].strip() in assistant_output,
            "has_edit": segment_has_edit(state, state.segment),
        }
    )
    _append_observation(state, rejection)
    state.turns[-1].update(injected=True)
    state.turns[-1].pop("environment_observation", None)
    state.prompt = format_messages(
        state.messages, tokenizer_path=str(_CANONICAL_TOKENIZER_PATH), enable_thinking=True
    )


def _advance_segment(
    state: _TrajectoryState, assistant_output: str, followup: str, *, turn_index: int
) -> None:
    state.submits.append(
        {
            "turn": turn_index,
            "segment": state.segment,
            "format_ok": state.submit_clause.split("&&")[0].strip() in assistant_output,
            "post_nudge": bool(state.nudged_at) and turn_index > state.nudged_at,
            "has_edit": segment_has_edit(state, state.segment),
        }
    )
    first = state.segment == "micro"
    state.segment_index += not first
    repeated = followup.strip() and followup.strip() == state.last_followup
    state.last_followup = followup.strip()
    if not followup.strip() or repeated:
        state.stopped = True
        logger.info(
            "[sanity-dispatch] {} submission accepted ({})",
            state.sample_id,
            "repeated request" if repeated else "no grounded follow-up",
        )
        return
    state.segment = "real" if first else f"followup_{state.segment_index}"
    instruction = followup_instruction(followup, state.submit_clause, first=first)
    _append_observation(state, instruction)
    state.turns[-1].update(segment=state.segment, injected=True)
    state.turns[-1].pop("environment_observation", None)
    state.prompt = format_messages(
        state.messages,
        tokenizer_path=str(_CANONICAL_TOKENIZER_PATH),
        enable_thinking=True,
    )


def _append_observation(state: _TrajectoryState, observation: str) -> None:
    assistant_output = str(state.turns[-1].get("content") or "")
    state.messages.extend(
        [
            {"role": "assistant", "content": assistant_output},
            {"role": "user", "content": observation},
        ]
    )
    state.turns.append(
        {
            "role": "user",
            "content": observation,
            "environment_observation": True,
        }
    )


def _has_edited(state: _TrajectoryState) -> bool:
    return bool(_edit_turns(state))


def _edit_turns(state: _TrajectoryState) -> int:
    """How many turns actually changed a file — the work the empty-submit check is named for."""
    return sum(
        1
        for t in state.turns
        if t.get("role") == "assistant" and _CHAIN_EDIT_RE.search(str(t.get("content") or ""))
    )


_EXECUTES_RE = re.compile(
    r"\b(?:python\d?|pytest|go|cargo|npm|npx|yarn|make|bash|sh|mocha|tox|mvn|gradle|java|node)\b"
)


def _state_fingerprint(state: _TrajectoryState) -> str:
    mutating = [
        first_bash_command(content)
        for t in state.turns
        if t.get("role") == "assistant"
        for content in [str(t.get("content") or "")]
        if _CHAIN_EDIT_RE.search(content)
        or _REMOVAL_RE.search(first_bash_command(content))
        or _EXECUTES_RE.search(first_bash_command(content))
    ]
    return hashlib.sha1("\0".join(mutating).encode("utf-8", "replace")).hexdigest()


def _named_in_removal(state: _TrajectoryState, path: str) -> bool:
    assistant = [str(t.get("content") or "") for t in state.turns if t.get("role") == "assistant"]
    return named_in_removal(assistant, [first_bash_command(a) for a in assistant], path)


async def _simulate_observation(
    *,
    client: Any,
    settings: JudgeSettings,
    eval_run_id: str,
    state: _TrajectoryState,
    assistant_output: str,
) -> str:
    key = hashlib.sha1(
        "\0".join(
            (
                state.sample_id,
                _state_fingerprint(state),
                first_bash_command(assistant_output),
            )
        ).encode("utf-8", "replace")
    ).hexdigest()
    return await state.observation_memo.observe(
        key,
        lambda: _simulate_observation_uncached(
            client=client,
            settings=settings,
            eval_run_id=eval_run_id,
            state=state,
            assistant_output=assistant_output,
        ),
    )


async def _simulate_observation_uncached(
    *,
    client: Any,
    settings: JudgeSettings,
    eval_run_id: str,
    state: _TrajectoryState,
    assistant_output: str,
) -> str:
    sample_id, prompt, messages = state.sample_id, state.prompt, state.messages
    fmt = detect_format(sample_id, messages)
    transcript = _simulation_transcript(
        messages=messages, prompt=prompt, assistant_output=assistant_output
    )
    command = first_bash_command(assistant_output)
    observation = ""
    for attempt in range(MAX_CONSECUTIVE_DEGENERATE_OBSERVATIONS):
        ask = transcript if attempt == 0 else f"{transcript}\n\n{DEGENERATE_RETRY}"
        response = await client.complete(
            model=settings.evaluator_model,
            messages=[
                {"role": "system", "content": simulation_system_prompt(fmt)},
                {"role": "user", "content": ask},
            ],
            temperature=0.0 if attempt == 0 else _DEGENERATE_RETRY_TEMPERATURE,
            max_tokens=settings.simulation_max_tokens,
            provider=_evaluator_provider(settings),
            accept=lambda raw: (
                valid_output(raw, fmt)
                and not degenerate_observation(raw)
                and not stuttered_lines(raw)
                and not fabricated_sed_error(command, raw)
            ),
        )
        if response.error:
            raise RuntimeError(response.error)
        observation = repair_output(response.raw.strip(), fmt)
        if leaked_turn(observation):
            logger.warning(
                "[sanity-dispatch] observation came back as a transcript turn sample_id={} "
                "attempt={}/{}: {!r}",
                sample_id,
                attempt + 1,
                MAX_CONSECUTIVE_DEGENERATE_OBSERVATIONS,
                observation[:160],
            )
            observation = ""
            continue
        if fabricated_sed_error(command, observation):
            logger.warning(
                "[sanity-dispatch] invented a sed error real sed does not give sample_id={} "
                "command={!r}: {!r}",
                sample_id,
                command[:120],
                observation[:120],
            )
            return empty_output(fmt)
        if diagnostic := misdiagnosed_sed(command, observation):
            logger.info(
                "[sanity-dispatch] replaced an invented sed diagnostic sample_id={}: {!r} -> {!r}",
                sample_id,
                observation[:80],
                diagnostic,
            )
            return wrap(diagnostic, fmt, returncode=1)
        if not degenerate_observation(observation) and not stuttered_lines(observation):
            break
        logger.warning(
            "[sanity-dispatch] observation collapsed into repeated lines sample_id={} "
            "attempt={}/{}: {!r}",
            sample_id,
            attempt + 1,
            MAX_CONSECUTIVE_DEGENERATE_OBSERVATIONS,
            observation[:160],
        )
    else:
        if not observation:
            return empty_output(fmt)
        if degenerate_observation(observation):
            raise RuntimeError(
                f"simulator collapsed into repeated lines on "
                f"{MAX_CONSECUTIVE_DEGENERATE_OBSERVATIONS} consecutive attempts at one step"
            )
        logger.warning(
            "[sanity-dispatch] kept a stuttered observation after retries sample_id={}: {}",
            sample_id,
            stuttered_lines(observation),
        )
    if META_LEAK_RE.search(observation):
        logger.warning(
            "[sanity-dispatch] observation broke character sample_id={}: {!r}",
            sample_id,
            observation[:160],
        )
        return empty_output(fmt)
    if not valid_output(observation, fmt):
        fallback = empty_output(fmt)
        logger.warning(
            "[sanity-dispatch] observation_simulation_invalid_format eval_run_id={} "
            "sample_id={} fmt={} fallback={!r}",
            eval_run_id,
            sample_id,
            fmt,
            fallback,
        )
        return fallback
    if echoed_command(command, observation):
        logger.warning(
            "[sanity-dispatch] observation echoed the command back sample_id={} command={!r}",
            sample_id,
            command[:80],
        )
        observation = empty_output(fmt)
    phantom = (not _has_edited(state) and claims_tracked_change(command, observation)) or any(
        not _named_in_removal(state, path) for path in deleted_files(command, observation)
    )
    if phantom:
        logger.warning(
            "[sanity-dispatch] claimed a change no command made sample_id={} command={!r}",
            sample_id,
            command[:80],
        )
        observation = without_tracked_changes(observation, fmt)
    observation = repair_to_contract(observation, fmt, command_contract(command))
    observation = renumbered_view(command, canonical_empty(observation, fmt))
    if requires_output(command) and not has_content(observation, fmt):
        return await _retry_for_output(
            client=client,
            settings=settings,
            sample_id=sample_id,
            fmt=fmt,
            transcript=transcript,
            command=command,
            observation=observation,
        )
    return observation


async def _retry_for_output(
    *,
    client: Any,
    settings: JudgeSettings,
    sample_id: str,
    fmt: str,
    transcript: str,
    command: str,
    observation: str,
) -> str:
    response = await client.complete(
        model=settings.evaluator_model,
        messages=[
            {"role": "system", "content": simulation_system_prompt(fmt)},
            {"role": "user", "content": f"{transcript}\n\n{MUST_PRINT_RETRY}"},
        ],
        temperature=0.0,
        max_tokens=settings.simulation_max_tokens,
        provider=_evaluator_provider(settings),
        accept=lambda raw: (
            valid_output(raw, fmt) and has_content(raw, fmt) and not degenerate_observation(raw)
        ),
    )
    candidate = "" if response.error else response.raw.strip()
    recovered = (
        valid_output(candidate, fmt)
        and has_content(candidate, fmt)
        and not degenerate_observation(candidate)
    )
    logger.info(
        "[sanity-dispatch] must_print_retry sample_id={} command={!r} recovered={}",
        sample_id,
        command[:80],
        recovered,
    )
    return candidate if recovered else observation


def _trajectory_result(
    run_id: str, states: list[_TrajectoryState], turn_count: int
) -> dict[str, Any]:
    responses = ["" if state.error else _format_scored_trajectory(state.turns) for state in states]
    heuristics = [
        {
            "passed": not state.error and not state.heuristic_reason,
            "reason": state.error or state.heuristic_reason,
            # state.error is a chain-infra failure (evaluator/simulator), never model behavior
            "infra": bool(state.error),
        }
        for state in states
    ]
    return {
        "type": "result",
        "run_id": run_id,
        "state": "succeeded",
        "responses": responses,
        "heuristics": heuristics,
        "submit_commands": [state.submit_clause for state in states],
        "assistant_turns": turn_count,
    }


def _format_scored_trajectory(turns: list[dict[str, Any]]) -> str:
    target_count = sum(
        1 for turn in turns if turn.get("role") == "assistant" and turn.get("score_target")
    )
    target_label = (
        "CANDIDATE OUTPUT"
        if target_count == 1
        else f"CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT {target_count}"
    )
    assistant_index = 0
    parts = [
        "FULL CANDIDATE TRAJECTORY",
        f"Score ONLY {target_label}. The ENVIRONMENT OBSERVATION is context only.",
    ]
    for turn in turns:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").rstrip()
        if role == "assistant" and turn.get("score_target"):
            assistant_index += 1
            label = f"CANDIDATE OUTPUT {assistant_index}"
        elif role == "user" and turn.get("environment_observation"):
            label = "ENVIRONMENT OBSERVATION (context only, do not score)"
        else:
            label = (
                f"CONTEXT {role.upper()} (do not score)" if role else "CONTEXT TURN (do not score)"
            )
        parts.append(f"\n{label}:\n------\n{content}\n------")
    return "\n".join(parts).strip()


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
        sections.append(f"### {role}\n{str(message.get('content') or '').rstrip()}")
    return "\n\n".join(sections).rstrip()


def _evaluator_provider(settings: JudgeSettings) -> dict[str, Any]:
    block: dict[str, Any] = {"allow_fallbacks": True, "quantizations": ["fp8"]}
    order = [p.strip() for p in settings.evaluator_providers.split(",") if p.strip()]
    if order:
        block["order"] = order
        block["allow_fallbacks"] = False
    return block


def _has_bash_command(output: str) -> bool:
    return bool(_BASH_BLOCK_RE.search(output))


def _completion_observation(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    return wrap(COMPLETE_MARKER, detect_format(sample_id, messages))


def _missing_command_observation(
    sample_id: str, messages: list[dict[str, str]] | None = None
) -> str:
    return wrap(
        "No bash command found in assistant message.",
        detect_format(sample_id, messages),
        returncode=2,
    )
