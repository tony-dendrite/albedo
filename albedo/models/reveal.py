"""albedo.models.reveal — v4 reveal string: ``v4|{repo}|{digest}|{hotkey}``."""
from __future__ import annotations

from albedo.models.ref import ModelRef

_VERSION = "v4"
_SEP = "|"
_EXPECTED_PARTS = 4


def build_reveal_v4(
    ref_or_repo: "ModelRef | str",
    digest_or_hotkey: str,
    hotkey: str | None = None,
) -> str:
    """Build a v4 reveal string.

    Accepts ``(ModelRef, hotkey)`` or legacy ``(repo, digest, hotkey)``.
    """
    if isinstance(ref_or_repo, ModelRef):
        ref = ref_or_repo
        hk = digest_or_hotkey
    else:
        # positional: (repo, digest, hotkey)
        if hotkey is None:
            raise TypeError(
                "build_reveal_v4(repo, digest, hotkey) requires three arguments"
            )
        ref = ModelRef(repo=ref_or_repo, digest=digest_or_hotkey)
        hk = hotkey

    if not hk:
        raise ValueError("hotkey must be a non-empty SS58 address")

    return _SEP.join([_VERSION, ref.repo, ref.digest, hk])


def parse_reveal_v4(data: str) -> tuple[ModelRef, str]:
    """Parse a v4 reveal string into ``(ModelRef, author_hotkey_ss58)``.

    Raises ValueError if malformed, wrong version, or invalid ModelRef.
    """
    parts = data.split(_SEP)
    if len(parts) != _EXPECTED_PARTS:
        raise ValueError(
            f"Expected {_EXPECTED_PARTS} pipe-separated fields, got {len(parts)}: {data!r}"
        )

    version, repo, digest, hk = parts

    if version != _VERSION:
        raise ValueError(
            f"Unsupported reveal version {version!r}; expected {_VERSION!r}"
        )
    if not hk:
        raise ValueError("Hotkey field is empty in reveal string")

    ref = ModelRef(repo=repo, digest=digest)  # validates repo/digest
    return ref, hk
