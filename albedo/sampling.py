"""Deterministic SWE-ZERO manifest sampling - shared by sanity, eval dispatch, and GPU workers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_SHARD_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)?data/train-[A-Za-z0-9_.-]*\.parquet$")


@dataclass(frozen=True, order=True)
class SweZeroSampleId:
    # A (shard, row, turn) coordinate into the pinned SWE-ZERO dataset.
    shard_name: str
    row_idx: int
    turn_idx: int

    def as_string(self) -> str:
        # Canonical "shard:row:turn" form stored in eval_runs.dataset_sample_ids.
        return f"{self.shard_name}:{self.row_idx}:{self.turn_idx}"


def multi_source_manifest_sample_ids(
    manifest: dict[str, Any],
    *,
    block_hash: str,
    sample_count: int = 128,
    max_turns_per_sample: int = 10,
) -> list[str]:
    # Samples coordinates across all manifest sources by weight (largest remainder); deterministic
    # in block_hash so the CPU dispatcher and the GPU worker derive the identical sample list.
    if "sources" not in manifest:
        raise ValueError(
            "dataset manifest must define a 'sources' array; single-source manifests are "
            "not supported (the eval requires the combined multi-dataset manifest)"
        )

    _validate_sampling_args(block_hash, sample_count, max_turns_per_sample)
    sources = _normalized_sources(manifest)
    if not sources or sample_count == 0:
        return []

    allocations = _allocate_by_weight(sources, sample_count)

    selected: list[str] = []
    for source in sources:
        rng = random.Random(str(block_hash))
        selected.extend(
            _select_from_shards(
                source["shards"],
                rng=rng,
                sample_count=allocations[source["name"]],
                max_turns_per_sample=max_turns_per_sample,
            )
        )

    if len(selected) < sample_count:
        already = set(selected)
        for source in sources:
            if len(selected) >= sample_count:
                break
            rng = random.Random(str(block_hash))
            for sid in _select_from_shards(
                source["shards"],
                rng=rng,
                sample_count=sample_count,
                max_turns_per_sample=max_turns_per_sample,
            ):
                if sid in already:
                    continue
                selected.append(sid)
                already.add(sid)
                if len(selected) >= sample_count:
                    break

    return selected


def _validate_sampling_args(block_hash: str, sample_count: int, max_turns_per_sample: int) -> None:
    # Rejects arguments that would make the deterministic sample undefined.
    if not block_hash:
        raise ValueError("block_hash is required for eval dataset sampling")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if max_turns_per_sample <= 0:
        raise ValueError("max_turns_per_sample must be positive")


def _select_from_shards(
    shards: list[dict[str, Any]],
    *,
    rng: random.Random,
    sample_count: int,
    max_turns_per_sample: int,
) -> list[str]:
    # Shuffles all rows with the seeded rng, then walks turns per row until sample_count is met.
    if not shards or sample_count <= 0:
        return []

    rows: list[tuple[int, str, int]] = []
    for shard_idx, shard in enumerate(shards):
        for row_idx in range(shard["rows"]):
            rows.append((shard_idx, shard["name"], row_idx))

    row_order = list(range(len(rows)))
    rng.shuffle(row_order)

    selected: list[str] = []
    seen: set[tuple[int, int, int]] = set()
    while len(selected) < sample_count:
        made_progress = False
        for row_position in row_order:
            shard_idx, shard_name, row_idx = rows[row_position]
            for turn_idx in range(max_turns_per_sample):
                key = (shard_idx, row_idx, turn_idx)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(SweZeroSampleId(shard_name, row_idx, turn_idx).as_string())
                made_progress = True
                if len(selected) >= sample_count:
                    break
            if len(selected) >= sample_count:
                break
        if not made_progress:
            break

    return selected


def _allocate_by_weight(sources: list[dict[str, Any]], sample_count: int) -> dict[str, int]:
    # Apportions sample_count across sources by weight using the largest-remainder method.
    total_weight = sum(source["weight"] for source in sources)
    if total_weight <= 0:
        raise ValueError("manifest source weights must sum to a positive value")

    exact = [(source["name"], sample_count * source["weight"] / total_weight) for source in sources]
    allocations = {name: math.floor(value) for name, value in exact}
    remainder = sample_count - sum(allocations.values())
    ranked = sorted(exact, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for name, _ in ranked[:remainder]:
        allocations[name] += 1
    return allocations


def _normalized_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    # Validates and name-sorts manifest sources so iteration order is deterministic.
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest.sources must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    grand_total = 0
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("manifest source entries must be objects")
        name = source.get("name")
        weight = source.get("weight")
        if not isinstance(name, str) or not name:
            raise ValueError("manifest source name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"duplicate manifest source name: {name}")
        seen_names.add(name)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError("manifest source weight must be a non-negative number")
        shards = _normalized_shards(source)
        source_total = sum(shard["rows"] for shard in shards)
        declared_source_total = source.get("total_rows")
        if declared_source_total is not None and declared_source_total != source_total:
            raise ValueError(f"manifest source {name} total_rows does not match shard rows")
        normalized.append({"name": name, "weight": float(weight), "shards": shards})
        grand_total += source_total

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != grand_total:
        raise ValueError("manifest total_rows does not match source rows")

    normalized.sort(key=lambda source: source["name"])
    return normalized


def _normalized_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    # Validates shard names/rows and cross-checks the declared total.
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("manifest.shards must be a list")

    normalized = []
    total_rows = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("manifest shard entries must be objects")
        name = shard.get("name") or shard.get("path")
        rows = shard.get("rows")
        if not isinstance(name, str) or not _SHARD_RE.match(name):
            raise ValueError("manifest shards must be (<source>/)data/train-*.parquet files")
        if not isinstance(rows, int) or rows < 0:
            raise ValueError("manifest shard rows must be non-negative integers")
        normalized.append({"name": name, "rows": rows})
        total_rows += rows

    declared_total = manifest.get("total_rows")
    if declared_total is not None and declared_total != total_rows:
        raise ValueError("manifest total_rows does not match shard rows")

    return normalized


# ── Manifest + prompt loading (merged from dataset.py) ───────────────────────

_PROMPTS_FILE = Path(__file__).parent / "prompts.json"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


@dataclass(frozen=True)
class SanitySample:
    # One sampled prompt to send to the GPU worker.
    prompt: str
    messages: list[dict[str, str]] | None = None


def load_manifest_file(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    # Loads a manifest JSON from disk and verifies its sha256 against the pinned hash.
    manifest_path = Path(path)
    payload = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    normalized_expected = expected_sha256.removeprefix("sha256:")
    if normalized_expected and actual_sha256 != normalized_expected:
        raise ValueError(
            f"dataset manifest hash mismatch: expected {normalized_expected}, got {actual_sha256}"
        )
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return loaded


def sample_prompts(
    *,
    seed: str,
    n: int = 3,
    max_turns: int = 10,
    manifest_path: str = "",
    manifest_hash: str = "",
    dataset_root: str = "",
) -> list[SanitySample]:
    # Deterministically samples n SWE-ZERO prompts for a challenger; falls back to prompts.json.
    if manifest_path and dataset_root:
        manifest = load_manifest_file(manifest_path, expected_sha256=manifest_hash)
        sample_ids = multi_source_manifest_sample_ids(
            manifest, block_hash=str(seed), sample_count=n, max_turns_per_sample=max_turns
        )
        root = Path(dataset_root)
        return [_load_sanity_sample(root, sample_id) for sample_id in sample_ids]
    return _fallback_prompts(n)


def _fallback_prompts(n: int) -> list[SanitySample]:
    # Static prompts.json fallback for local/dev when no SWE-ZERO manifest is configured.
    prompts: list[str] = json.loads(_PROMPTS_FILE.read_text())[:n]
    return [
        SanitySample(prompt, messages=[{"role": "user", "content": prompt}]) for prompt in prompts
    ]


# ── Parquet row -> prompt (shared with the GPU workers) ──────────────────────


def _load_sanity_sample(root: Path, sample_id: str) -> SanitySample:
    # CPU-side wrapper: manual chat template (no tokenizer), target turn ignored.
    shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
    row = _read_parquet_row(root / shard_name, row_idx)
    prompt, _target, messages = _prompt_from_row(
        row, turn_idx=turn_idx, tokenizer_path=None, enable_thinking=False
    )
    return SanitySample(prompt=prompt, messages=messages or [])


@dataclass(frozen=True)
class EvalSample:
    # One duel turn: formatted prompt, optional gold assistant target, and the raw messages.
    sample_id: str
    prompt: str
    target: str | None = None
    messages: list[dict[str, str]] | None = None


def load_swe_zero_samples(
    *,
    dataset_root: str | Path,
    sample_ids: list[str],
    tokenizer_path: str | Path | None = None,
    enable_thinking: bool = True,
) -> list[EvalSample]:
    # Loads each (shard, row, turn) coordinate from the pinned parquet dataset.
    root = Path(dataset_root)
    return [
        _load_sample(
            root, sample_id, tokenizer_path=tokenizer_path, enable_thinking=enable_thinking
        )
        for sample_id in sample_ids
    ]


def _load_sample(
    root: Path,
    sample_id: str,
    *,
    tokenizer_path: str | Path | None,
    enable_thinking: bool,
) -> EvalSample:
    # Reads one parquet row and turns it into a formatted prompt + target.
    shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
    row = _read_parquet_row(root / shard_name, row_idx)
    prompt, target, messages = _prompt_from_row(
        row, turn_idx=turn_idx, tokenizer_path=tokenizer_path, enable_thinking=enable_thinking
    )
    return EvalSample(sample_id=sample_id, prompt=prompt, target=target, messages=messages)


def _parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    # Splits "<shard>:<row>:<turn>" and validates the shard name against the manifest pattern.
    shard_name, row_idx_raw, turn_idx_raw = sample_id.rsplit(":", 2)
    if not _SHARD_RE.match(shard_name):
        raise ValueError(f"unsupported SWE-ZERO shard in sample_id: {sample_id}")
    return shard_name, int(row_idx_raw), int(turn_idx_raw)


def _read_parquet_row(path: Path, row_idx: int) -> dict[str, Any]:
    # Streams the shard in batches and slices out exactly one row.
    if not path.exists():
        raise FileNotFoundError(f"SWE-ZERO shard not found: {path}")
    if row_idx < 0:
        raise ValueError("row_idx must be non-negative")

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=1024):
        if seen + batch.num_rows <= row_idx:
            seen += batch.num_rows
            continue
        local_idx = row_idx - seen
        return batch.slice(local_idx, 1).to_pydict() | {"__row_idx": [row_idx]}
    raise IndexError(f"row_idx {row_idx} out of range for shard {path}")


def _prompt_from_row(
    row: dict[str, Any],
    *,
    turn_idx: int,
    tokenizer_path: str | Path | None,
    enable_thinking: bool,
) -> tuple[str, str | None, list[dict[str, str]] | None]:
    # Builds the prompt up to the turn_idx-th assistant turn, with that turn as the target.
    normalized = {key: _unwrap_column(value) for key, value in row.items()}
    turns = _extract_turns(normalized)
    if turns:
        assistant_turns = [index for index, turn in enumerate(turns) if _role(turn) == "assistant"]
        source_index = (
            assistant_turns[turn_idx]
            if turn_idx < len(assistant_turns)
            else min(turn_idx, len(turns) - 1)
        )
        prompt_turns = turns[:source_index]
        target = (
            _content(turns[source_index]) if _role(turns[source_index]) == "assistant" else None
        )
        messages = _messages_from_turns(prompt_turns)
        prompt = format_messages(
            messages, tokenizer_path=tokenizer_path, enable_thinking=enable_thinking
        )
        return prompt, target, messages

    for key in ("prompt", "instruction", "question", "input", "text"):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            return value, None, [{"role": "user", "content": value}]
    fallback = {key: value for key, value in normalized.items() if not key.startswith("__")}
    prompt = json.dumps(fallback, sort_keys=True)
    return prompt, None, [{"role": "user", "content": prompt}]


def _extract_turns(row: dict[str, Any]) -> list[Any]:
    # Finds the conversation list under any of the common column names.
    for key in ("messages", "turns", "conversation", "trajectory"):
        value = row.get(key)
        parsed = _maybe_json(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for nested_key in ("messages", "turns", "conversation"):
                nested = parsed.get(nested_key)
                if isinstance(nested, list):
                    return nested
    return []


def format_messages(
    messages: list[dict[str, str]],
    *,
    tokenizer_path: str | Path | None = None,
    enable_thinking: bool = True,
) -> str:
    # Applies the model tokenizer's chat template when given, else the manual Qwen template.
    if tokenizer_path is not None:
        tokenizer = _load_tokenizer(str(tokenizer_path))
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if getattr(tokenizer, "chat_template", None) is None:
            kwargs["chat_template"] = _QWEN3_CHAT_TEMPLATE
        return tokenizer.apply_chat_template(messages, **kwargs)
    return _manual_chat_template(messages)


def _manual_chat_template(messages: list[dict[str, str]]) -> str:
    # Tokenizer-free Qwen chat-format fallback.
    parts = []
    for message in messages:
        role = _chat_role(message.get("role"))
        content = message.get("content", "")
        if content:
            parts.append(f"{_IM_START}{role}\n{content}{_IM_END}")
    parts.append(f"{_IM_START}assistant\n")
    return "\n".join(parts)


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_path: str):
    # Caches AutoTokenizer instances per path (transformers stays a GPU-host-only import).
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _messages_from_turns(turns: list[Any]) -> list[dict[str, str]]:
    # Normalizes raw dataset turns into chat messages, dropping empty ones.
    messages: list[dict[str, str]] = []
    for turn in turns:
        content = _content(turn)
        if content:
            messages.append({"role": _chat_role(_role(turn)), "content": content})
    return messages


def _chat_role(role: str | None) -> str:
    # Maps dataset role aliases onto the chat roles the template accepts.
    if role in {"assistant", "system", "user"}:
        return role
    if role in {"human", "prompter"}:
        return "user"
    if role in {"gpt", "bot", "model"}:
        return "assistant"
    return "user"


def _unwrap_column(value: Any) -> Any:
    # Unwraps single-element pydict columns produced by the one-row slice.
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _maybe_json(value: Any) -> Any:
    # Parses JSON-encoded string columns, passing everything else through.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _role(turn: Any) -> str | None:
    # Extracts the speaker role from a raw turn.
    if isinstance(turn, dict):
        role = turn.get("role") or turn.get("speaker") or turn.get("from")
        return str(role).lower() if role is not None else None
    return None


def _content(turn: Any) -> str:
    # Extracts the text content from a raw turn.
    if isinstance(turn, dict):
        for key in ("content", "text", "value", "message"):
            value = turn.get(key)
            if value is not None:
                return str(value)
    return str(turn) if turn is not None else ""


_QWEN3_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}
"""
