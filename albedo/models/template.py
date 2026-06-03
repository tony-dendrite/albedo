"""albedo.models.template — chat-template injection and tokenizer hygiene."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Resolved relative to repo root so it works regardless of cwd.
_REPO_ROOT = Path(__file__).parent.parent.parent
_CANONICAL_TEMPLATE = _REPO_ROOT / "archs" / "qwen3" / "chat_template.jinja"

_TOKENIZER_CONFIG = "tokenizer_config.json"
_CHAT_TEMPLATE_KEY = "chat_template"


def _load_canonical_template() -> str:
    """Read the canonical Qwen3 chat template from archs/qwen3/."""
    if not _CANONICAL_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Canonical chat template not found at {_CANONICAL_TEMPLATE}. "
            "Ensure archs/qwen3/chat_template.jinja is present in the repo."
        )
    return _CANONICAL_TEMPLATE.read_text(encoding="utf-8")


def _tokenizer_config_path(model_dir: str) -> Path:
    return Path(model_dir) / _TOKENIZER_CONFIG


def ensure_chat_template(model_dir: str) -> bool:
    """Inject the canonical Qwen3 chat template if absent or non-standard.

    Returns True if written, False if already canonical.
    """
    cfg_path = _tokenizer_config_path(model_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"ensure_chat_template: {cfg_path} not found in model directory"
        )

    canonical = _load_canonical_template()

    cfg: dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    existing = cfg.get(_CHAT_TEMPLATE_KEY, "")

    if existing == canonical:
        log.debug("ensure_chat_template: template already canonical in %s", model_dir)
        return False

    cfg[_CHAT_TEMPLATE_KEY] = canonical
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "ensure_chat_template: wrote canonical template to %s (%s)",
        cfg_path,
        "injected" if not existing else "replaced non-standard",
    )
    return True


def scrub_tokenizer_config(model_dir: str) -> None:
    """Strip auto_map and force trust_remote_code=False in tokenizer_config.json.

    Safe to call when keys are absent. Security: prevents arbitrary code exec via HF hooks.
    """
    cfg_path = _tokenizer_config_path(model_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"scrub_tokenizer_config: {cfg_path} not found in model directory"
        )

    cfg: dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    changed = False

    if "auto_map" in cfg:
        del cfg["auto_map"]
        changed = True
        log.info("scrub_tokenizer_config: removed auto_map from %s", cfg_path)

    if cfg.get("trust_remote_code") is not False:
        cfg["trust_remote_code"] = False
        changed = True
        log.info("scrub_tokenizer_config: set trust_remote_code=False in %s", cfg_path)

    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
