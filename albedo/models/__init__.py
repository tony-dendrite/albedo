"""albedo.models — model lifecycle: ref, reveal, download, upload, template."""

from albedo.models.ref import ModelRef
from albedo.models.reveal import build_reveal_v4, parse_reveal_v4
from albedo.models.download import materialize_model, list_remote_files, prune_model_cache
from albedo.models.upload import upload_model_folder
from albedo.models.template import ensure_chat_template
from albedo.models.compat import config_lock_violation

__all__ = [
    "ModelRef",
    "build_reveal_v4",
    "parse_reveal_v4",
    "materialize_model",
    "upload_model_folder",
    "list_remote_files",
    "prune_model_cache",
    "ensure_chat_template",
    "config_lock_violation",
]
