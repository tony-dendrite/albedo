from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .sampling import _SHARD_RE

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

# Sources whose samples are tool-call probes: shard prefix -> pinned tools file. The OpenHands
# toolset is fixed (5 tools); pinning the schemas keeps prompt rendering byte-identical across
# validators regardless of what any dataset row carries.
_TOOL_SOURCE_PREFIX = "swe-zero-tools/"
_OPENHANDS_TOOLS_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "tools" / "openhands_tools.json"
)
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


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    prompt: str
    target: str | None = None
    messages: list[dict[str, Any]] | None = None


@lru_cache(maxsize=1)
def _openhands_tools_json() -> str:
    return _OPENHANDS_TOOLS_PATH.read_text(encoding="utf-8")


def _tools_for_shard(shard_name: str) -> list[dict[str, Any]] | None:
    if shard_name.startswith(_TOOL_SOURCE_PREFIX):
        return json.loads(_openhands_tools_json())
    return None


def load_swe_zero_samples(
    *,
    dataset_root: str | Path,
    sample_ids: list[str],
    tokenizer_path: str | Path | None = None,
    enable_thinking: bool = True,
) -> list[EvalSample]:
    root = Path(dataset_root)
    return [
        _load_sample(
            root,
            sample_id,
            tokenizer_path=tokenizer_path,
            enable_thinking=enable_thinking,
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
    shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
    row = _read_parquet_row(root / shard_name, row_idx)
    prompt, target, messages = _prompt_from_row(
        row,
        turn_idx=turn_idx,
        tokenizer_path=tokenizer_path,
        enable_thinking=enable_thinking,
        tools=_tools_for_shard(shard_name),
    )
    return EvalSample(sample_id=sample_id, prompt=prompt, target=target, messages=messages)


def _parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    shard_name, row_idx_raw, turn_idx_raw = sample_id.rsplit(":", 2)
    if not _SHARD_RE.match(shard_name):
        raise ValueError(f"unsupported SWE-ZERO shard in sample_id: {sample_id}")
    return shard_name, int(row_idx_raw), int(turn_idx_raw)


def _read_parquet_row(path: Path, row_idx: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"SWE-ZERO shard not found: {path}")
    if row_idx < 0:
        raise ValueError("row_idx must be non-negative")

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
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None, list[dict[str, Any]] | None]:
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
            _target_text(turns[source_index]) if _role(turns[source_index]) == "assistant" else None
        )
        messages = _messages_from_turns(prompt_turns)
        prompt = format_messages(
            messages, tokenizer_path=tokenizer_path, enable_thinking=enable_thinking, tools=tools
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


def format_user_prompt(
    prompt: str, *, tokenizer_path: str | Path | None = None, enable_thinking: bool = True
) -> str:
    return format_messages(
        [{"role": "user", "content": prompt}],
        tokenizer_path=tokenizer_path,
        enable_thinking=enable_thinking,
    )


def format_messages(
    messages: list[dict[str, Any]],
    *,
    tokenizer_path: str | Path | None = None,
    enable_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    if tokenizer_path is not None:
        tokenizer = _load_tokenizer(str(tokenizer_path))
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if getattr(tokenizer, "chat_template", None) is None:
            if tools is not None:
                raise ValueError(
                    "tool samples require the canonical chat template; the fallback "
                    "template cannot render tools"
                )
            kwargs["chat_template"] = _QWEN3_CHAT_TEMPLATE
        return tokenizer.apply_chat_template(messages, **kwargs)
    if tools is not None:
        raise ValueError("tool samples require a tokenizer_path (manual template lacks tools)")
    return _manual_chat_template(messages)


def _manual_chat_template(messages: list[dict[str, str]]) -> str:
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
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _messages_from_turns(turns: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        tool_calls = _normalized_tool_calls(turn)
        if tool_calls:
            # Don't route through _content: a struct turn whose content is None would be
            # stringified wholesale there. Tool-call turns keep blank content instead.
            raw = turn.get("content")
            content = str(raw) if raw is not None else ""
        else:
            content = _content(turn)
            if not content:
                continue
        message: dict[str, Any] = {"role": _chat_role(_role(turn)), "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
    return messages


def _normalized_tool_calls(turn: Any) -> list[dict[str, Any]]:
    """OpenAI-shape tool calls with `arguments` as a dict — the canonical template iterates
    `tool_call.function.arguments|items`, so JSON-string arguments (as stored in the
    swe-zero-tools parquet) must be decoded here."""
    if not isinstance(turn, dict):
        return []
    calls = turn.get("tool_calls")
    if not isinstance(calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        entry: dict[str, Any] = {
            "type": call.get("type") or "function",
            "function": {"name": str(function["name"]), "arguments": arguments},
        }
        if call.get("id"):
            entry["id"] = call["id"]
        normalized.append(entry)
    return normalized


def _target_text(turn: Any) -> str:
    """Reference next-turn text for artifacts: content plus any tool calls it carried.
    Turns without tool calls keep the exact _content behavior."""
    tool_calls = _normalized_tool_calls(turn)
    if not tool_calls:
        return _content(turn)
    raw = turn.get("content")
    content = str(raw).strip() if raw is not None else ""
    calls = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
    return f"{content}\n{calls}" if content else calls


def _chat_role(role: str | None) -> str:
    if role in {"assistant", "system", "user", "tool"}:
        return role
    if role in {"human", "prompter"}:
        return "user"
    if role in {"gpt", "bot", "model"}:
        return "assistant"
    return "user"


def _unwrap_column(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _role(turn: Any) -> str | None:
    if isinstance(turn, dict):
        role = turn.get("role") or turn.get("speaker") or turn.get("from")
        return str(role).lower() if role is not None else None
    return None


def _content(turn: Any) -> str:
    if isinstance(turn, dict):
        for key in ("content", "text", "value", "message"):
            value = turn.get(key)
            if value is not None:
                return str(value)
    return str(turn) if turn is not None else ""
