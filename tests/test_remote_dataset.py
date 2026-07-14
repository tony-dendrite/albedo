import json

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service import remote_dataset
from albedo_eval_service.remote_dataset import load_swe_zero_samples


def test_load_swe_zero_sample_from_messages_json(tmp_path):
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "Use the right assertion."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path, sample_ids=["data/train-00000.parquet:0:0"]
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "data/train-00000.parquet:0:0"
    assert samples[0].prompt == (
        "<|im_start|>system\n"
        "Be concise.<|im_end|>\n"
        "<|im_start|>user\n"
        "Fix the failing test.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert samples[0].target == "Use the right assertion."


def test_load_swe_zero_sample_uses_tokenizer_chat_template(tmp_path, monkeypatch):
    captured = {}

    class _Tokenizer:
        chat_template = "native-template"

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "templated prompt"

    monkeypatch.setattr(remote_dataset, "_load_tokenizer", lambda path: _Tokenizer())

    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "Use the right assertion."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=["data/train-00000.parquet:0:0"],
        tokenizer_path="/models/qwen",
        enable_thinking=True,
    )

    assert samples[0].prompt == "templated prompt"
    assert samples[0].messages == messages[:2]
    assert captured["messages"] == messages[:2]
    assert captured["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_load_swe_zero_sample_supplies_canonical_template_when_missing(tmp_path, monkeypatch):
    captured = {}

    class _Tokenizer:
        chat_template = None

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "canonical templated prompt"

    monkeypatch.setattr(remote_dataset, "_load_tokenizer", lambda path: _Tokenizer())

    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": "Done."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=["data/train-00000.parquet:0:0"],
        tokenizer_path="/models/qwen",
        enable_thinking=False,
    )

    assert samples[0].prompt == "canonical templated prompt"
    assert captured["messages"] == messages[:1]
    assert captured["kwargs"]["tokenize"] is False
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert captured["kwargs"]["enable_thinking"] is False
    assert "chat_template" in captured["kwargs"]
    assert "<|im_start|>assistant" in captured["kwargs"]["chat_template"]
    assert "enable_thinking" in captured["kwargs"]["chat_template"]


def test_load_swe_zero_sample_from_prompt_column(tmp_path):
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    table = pa.table({"prompt": ["Explain pytest fixtures."]})
    pq.write_table(table, shard_dir / "train-00001.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path, sample_ids=["data/train-00001.parquet:0:0"]
    )

    assert samples[0].prompt == "Explain pytest fixtures."
    assert samples[0].target is None


def test_load_sample_from_namespaced_source(tmp_path):
    # mini-coder shards live under a namespaced subdir; the sample_id carries the prefix.
    shard_dir = tmp_path / "mini-coder" / "data"
    shard_dir.mkdir(parents=True)
    messages = [
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": "Patched it."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000-of-00060.parquet")

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=["mini-coder/data/train-00000-of-00060.parquet:0:0"],
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "mini-coder/data/train-00000-of-00060.parquet:0:0"
    assert samples[0].target == "Patched it."


def _tool_turn(role, content, tool_calls=None):
    # Mimic parquet struct decoding: every turn carries every key, absent values are None.
    return {"role": role, "content": content, "tool_calls": tool_calls}


def _tool_trajectory():
    return [
        _tool_turn("system", "You are OpenHands agent."),
        _tool_turn("user", "Fix the bug in /workspace/repo."),
        _tool_turn(
            "assistant",
            "I'll list the repo first.",
            [
                {
                    "function": {"name": "execute_bash", "arguments": '{"command": "ls /workspace"}'},
                    "id": "call-1",
                    "type": "function",
                }
            ],
        ),
        _tool_turn("tool", "file_a.py\nfile_b.py"),
        _tool_turn(
            "assistant",
            None,
            [
                {
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/workspace/repo/file_a.py"}',
                    },
                    "id": "call-2",
                    "type": "function",
                }
            ],
        ),
    ]


def _write_tool_shard(tmp_path):
    shard_dir = tmp_path / "swe-zero-tools" / "data"
    shard_dir.mkdir(parents=True)
    table = pa.table({"trajectory": [_tool_trajectory()]})
    pq.write_table(table, shard_dir / "train-00000-of-00064.parquet")
    return "swe-zero-tools/data/train-00000-of-00064.parquet"


def test_tool_source_passes_pinned_tools_and_structured_tool_calls(tmp_path, monkeypatch):
    captured = {}

    class _Tokenizer:
        chat_template = "native-template"

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "templated prompt"

    monkeypatch.setattr(remote_dataset, "_load_tokenizer", lambda path: _Tokenizer())
    shard = _write_tool_shard(tmp_path)

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=[f"{shard}:0:1"],
        tokenizer_path="/models/qwen",
    )

    tools = captured["kwargs"]["tools"]
    assert [t["function"]["name"] for t in tools] == [
        "execute_bash", "think", "finish", "task_tracker", "str_replace_editor",
    ]

    messages = captured["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    call = messages[2]["tool_calls"][0]
    # arguments must be a decoded dict: the canonical template iterates arguments|items.
    assert call["function"]["arguments"] == {"command": "ls /workspace"}
    assert call["type"] == "function" and call["id"] == "call-1"

    # target = the anchor assistant turn (blank content, one tool call), serialized.
    assert "str_replace_editor" in samples[0].target
    assert '"command": "view"' in samples[0].target


def test_non_tool_source_gets_no_tools_kwarg(tmp_path, monkeypatch):
    captured = {}

    class _Tokenizer:
        chat_template = "native-template"

        def apply_chat_template(self, messages, **kwargs):
            captured["kwargs"] = kwargs
            return "templated prompt"

    monkeypatch.setattr(remote_dataset, "_load_tokenizer", lambda path: _Tokenizer())

    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    messages = [
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": "Done."},
    ]
    table = pa.table({"messages": [json.dumps(messages)]})
    pq.write_table(table, shard_dir / "train-00000.parquet")

    load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=["data/train-00000.parquet:0:0"],
        tokenizer_path="/models/qwen",
    )

    assert "tools" not in captured["kwargs"]


def test_tool_source_renders_with_canonical_template(tmp_path):
    import pytest

    pytest.importorskip("transformers")
    shard = _write_tool_shard(tmp_path)
    tokenizer_path = (
        remote_dataset.Path(remote_dataset.__file__).resolve().parents[2]
        / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
    )

    samples = load_swe_zero_samples(
        dataset_root=tmp_path,
        sample_ids=[f"{shard}:0:1"],
        tokenizer_path=tokenizer_path,
    )
    prompt = samples[0].prompt

    # Tool schemas rendered into the system block.
    assert "<tools>" in prompt and '"execute_bash"' in prompt
    # Prior assistant tool call rendered in the canonical markup with decoded arguments.
    assert "<tool_call>" in prompt and "<function=execute_bash>" in prompt
    assert "<parameter=command>" in prompt and "ls /workspace" in prompt
    # Tool result rendered as a tool response block.
    assert "<tool_response>" in prompt and "file_a.py" in prompt
    # Ends ready for the model's next assistant turn (thinking opener included by default).
    assert "<|im_start|>assistant" in prompt.rstrip().rsplit("<|im_end|>", 1)[-1]
