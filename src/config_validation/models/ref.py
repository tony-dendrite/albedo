from __future__ import annotations

import os
import re
from dataclasses import dataclass

_REPO_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$")

BACKEND_HF = "hf"
BACKEND_HIPPIUS = "hippius"
BACKEND_S3 = "s3"

_HIPPIUS_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_S3_REPO_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.-]*/models/registrations/[0-9a-f]{64}$")


def cache_repo(repo: str) -> str:
    """Two-level cache path key: s3 prefixes flatten to '<bucket>/<registration>'."""
    if repo.startswith("s3://"):
        bucket, _, rest = repo.removeprefix("s3://").partition("/")
        return f"{bucket}/{rest.rstrip('/').rpartition('/')[2]}"
    return repo


def detect_backend(digest: str) -> str | None:
    if _HIPPIUS_DIGEST_RE.match(digest):
        return BACKEND_HIPPIUS
    if _GIT_SHA1_RE.match(digest) or _GIT_SHA256_RE.match(digest):
        return BACKEND_HF
    return None


def _default_backend() -> str:
    b = os.environ.get("ALBEDO_MODEL_BACKEND", BACKEND_HF).strip().lower()
    return b if b in (BACKEND_HF, BACKEND_HIPPIUS) else BACKEND_HF


def _allow_mutable() -> bool:
    return os.environ.get("ALBEDO_ALLOW_MUTABLE_REF", "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ModelRef:
    repo: str
    digest: str
    backend: str = ""

    def __post_init__(self) -> None:
        if self.repo.startswith("s3://"):
            if not _S3_REPO_RE.match(self.repo):
                raise ValueError(
                    f"ModelRef.repo {self.repo!r} is not a canonical private-store prefix"
                )
            if not _HIPPIUS_DIGEST_RE.match(self.digest):
                raise ValueError("private-store refs require a 'sha256:<hex64>' model digest")
            if self.backend and self.backend != BACKEND_S3:
                raise ValueError(
                    f"ModelRef.backend {self.backend!r} contradicts the s3:// repo scheme"
                )
            object.__setattr__(self, "backend", BACKEND_S3)
            return
        if not _REPO_RE.match(self.repo):
            raise ValueError(
                f"ModelRef.repo {self.repo!r} is not a valid lowercase '<namespace>/<name>' id"
            )
        detected = detect_backend(self.digest)
        if detected is None:
            if not _allow_mutable():
                raise ValueError(
                    "ModelRef.digest must be an immutable pin — a Hippius 'sha256:<hex64>' "
                    f"or an HF git revision (40/64 hex); got {self.digest!r}"
                )
            detected = _default_backend()
        chosen = self.backend or detected
        if chosen not in (BACKEND_HF, BACKEND_HIPPIUS):
            raise ValueError(f"ModelRef.backend must be 'hf' or 'hippius'; got {self.backend!r}")
        if self.backend and detect_backend(self.digest) not in (None, self.backend):
            raise ValueError(
                f"ModelRef.backend {self.backend!r} contradicts the {self.digest!r} digest format"
            )
        object.__setattr__(self, "backend", chosen)

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.digest}"

    def __str__(self) -> str:
        return self.immutable_ref
