from __future__ import annotations

from albedo_eval_service.shared.observation_format import OPENHANDS_TRUNCATION_NOTICE
from repo_context_service.overlay import build_overlay

PATH = "pkg/big.py"
TRUE = "\n".join(f"line_{i:04d} = {i}  # real source content here" for i in range(900)) + "\n"
SMALL = "import os\nimport sys\n"


def _overlay(command: str, observation: str, base: str = TRUE):
    messages = [
        {"role": "assistant", "content": f"```bash\n{command}\n```"},
        {"role": "user", "content": observation},
    ]
    return build_overlay(messages, [PATH], {"big.py": PATH}, lambda rel: base)


def _overlay_from(assistant_texts: list[str]):
    messages = [{"role": "assistant", "content": text} for text in assistant_texts]
    return build_overlay(messages, [PATH], {"big.py": PATH}, lambda rel: TRUE)


def _search_replace(path: str, old: str, new: str) -> str:
    return f"Editing `{path}`:\n\n```\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```"


def _openhands_clip(text: str) -> str:
    return f"{text[:15000]}\n{OPENHANDS_TRUNCATION_NOTICE}\n{text[-15000:]}"


def _returncode_clip(text: str) -> str:
    return (
        "<warning>\nThe output of your last command was too long.\n</warning><output_head>\n"
        f"{text[:5000]}\n</output_head>\n<elided_chars>\n{len(text) - 10000} characters elided\n"
        f"</elided_chars>\n<output_tail>\n{text[-5000:]}\n</output_tail>"
    )


def test_an_intact_read_is_adopted():
    assert _overlay("cat pkg/big.py", SMALL, base=SMALL).read(PATH) == SMALL


def test_a_clipped_read_is_never_adopted():
    for observation in (
        _openhands_clip(TRUE),
        _returncode_clip(TRUE),
        f"{TRUE[:2000]}\n<response clipped>",
    ):
        assert _overlay("cat pkg/big.py", observation).read(PATH) is None


def test_a_file_the_agent_wrote_enters_the_listing_with_its_content():
    body = "#!/usr/bin/env python3\nprint('hi')"
    overlay = _overlay_from([f"```bash\ncat <<'EOF' > test_x.py\n{body}\nEOF\n```"])
    assert overlay.created == {"test_x.py"}
    assert overlay.read("test_x.py") == body + "\n"
    assert "test_x.py" in overlay.listing([PATH])


def test_a_full_rewrite_clears_dirt():
    overlay = _overlay_from(
        [
            f"```bash\nsed -i 's/a/b/' {PATH}\n```",
            f"```bash\ncat <<'EOF' > {PATH}\nfresh\nEOF\n```",
        ]
    )
    assert not overlay.is_dirty(PATH)
    assert overlay.read(PATH) == "fresh\n"


def test_a_heredoc_does_not_mask_a_later_edit_in_the_same_turn():
    overlay = _overlay_from(
        [
            f"```bash\ncat <<'EOF' > {PATH}\nSEEDED\nEOF\n```",
            f"```bash\ncat <<'EOF' > fresh.py\nX\nEOF\nsed -i 's/q/r/' {PATH}\n```",
        ]
    )
    assert overlay.read("fresh.py") == "X\n"
    assert overlay.is_dirty(PATH)


SOURCE = "def go():\n    return tuple(x for x in o)\nSIZE = 1\nTAIL = 2\n"


def _sed(command: str, base: str = SOURCE):
    messages = [{"role": "assistant", "content": f"```bash\n{command}\n```"}]
    return build_overlay(messages, [PATH], {"big.py": PATH}, lambda rel: base)


def test_an_in_place_sed_is_applied_rather_than_forgotten():
    plain = _sed(f"sed -i 's/SIZE = 1/SIZE = 99/' {PATH}")
    assert not plain.is_dirty(PATH)
    assert plain.read(PATH) == SOURCE.replace("SIZE = 1", "SIZE = 99")

    addressed = _sed(f"cd /testbed && sed -i '2s|tuple(x for x in o)|[x for x in o]|' {PATH}")
    assert addressed.read(PATH) == SOURCE.replace("tuple(x for x in o)", "[x for x in o]")

    deleted = _sed(f"sed -i '3d' {PATH}")
    assert deleted.read(PATH) == "def go():\n    return tuple(x for x in o)\nTAIL = 2\n"


def test_an_in_place_sed_we_cannot_model_still_marks_the_file_dirty():
    for command in (
        f"sed -i -e 's/SIZE = 1/SIZE = 2/' -e 's/TAIL = 2/TAIL = 3/' {PATH}",  # two expressions
        f"sed -i 's/SIZE = 1/SIZE = 2/;s/TAIL = 2/TAIL = 3/' {PATH}",  # two expressions, one arg
        f"sed -i '/SIZE/d' {PATH}",  # pattern address
        f"sed -i 's/SIZE = 1/&& extra/' {PATH}",  # backreference in the replacement
        f"sed -i 's/absent from the file/x/' {PATH}",  # our copy disagrees with the agent's
        f"sed -i 's/SIZE = 1/SIZE = 2/' {PATH} && sed -i '3d' {PATH}",  # two edits, one path
    ):
        overlay = _sed(command)
        assert overlay.is_dirty(PATH), command
        assert overlay.read(PATH) is None, command
