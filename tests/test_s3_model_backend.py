"""s3 (private R2 store) model backend: refs, downloads, byte verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_private_store import model_files

from albedo_config import RemoteSettings
from albedo_eval_service.modelstore.resolver import ModelArtifactResolver
from config_validation.models import BACKEND_S3, ModelRef, cache_repo
from config_validation.storage import _paths, _s3
from model_validation.storage import download
from model_validation.storage.preflight import safetensors_dtypes
from private_store.digests import ArtifactIntegrityError, model_digest_from_inventory

RID = "a" * 64
BUCKET = "albedo-private-models-enam"
REPO = f"s3://{BUCKET}/models/registrations/{RID}"


def _files_with_real_safetensors() -> dict[str, bytes]:
    files = model_files()
    header = json.dumps({"linear.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}})
    blob = len(header).to_bytes(8, "little") + header.encode() + b"\x00\x00"
    files["model-00001-of-00002.safetensors"] = blob
    files["model-00002-of-00002.safetensors"] = blob
    return files


def _digest(files: dict[str, bytes]) -> str:
    return model_digest_from_inventory(
        [(path, len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()]
    )


class FakeS3Client:
    def __init__(self, files: dict[str, bytes]):
        prefix = f"models/registrations/{RID}/"
        self.objects = {prefix + name: data for name, data in files.items()}
        self.objects[prefix + "manifest.json"] = b"{}"
        self.downloads = 0

    def list_objects_v2(self, Bucket, Prefix, **_):
        contents = [
            {"Key": key, "Size": len(data)}
            for key, data in sorted(self.objects.items())
            if key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket, key, dest, **_):
        self.downloads += 1
        Path(dest).write_bytes(self.objects[key])

    def get_object(self, Bucket, Key, Range):
        start, end = map(int, Range.removeprefix("bytes=").split("-"))
        data = self.objects[Key][start : end + 1]
        return {"Body": type("B", (), {"read": lambda self_: data})()}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                yield client.list_objects_v2(Bucket, Prefix)

        return _Paginator()


@pytest.fixture
def s3_env(tmp_path, monkeypatch):
    files = _files_with_real_safetensors()
    client = FakeS3Client(files)
    monkeypatch.setattr(_s3, "_client", lambda: client)
    monkeypatch.setattr(_paths, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(download, "MODEL_CACHE_DIR", str(tmp_path))
    ref = ModelRef(repo=REPO, digest=f"sha256:{_digest(files)}")
    return files, client, ref, tmp_path


def test_model_ref_accepts_private_store_uris_and_rejects_malformed():
    ref = ModelRef(repo=REPO, digest="sha256:" + "b" * 64)
    assert ref.backend == BACKEND_S3
    assert cache_repo(REPO) == f"{BUCKET}/{RID}"
    assert cache_repo("alice/model") == "alice/model"
    with pytest.raises(ValueError, match="canonical private-store prefix"):
        ModelRef(repo="s3://bucket/other/prefix", digest="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ModelRef(repo=REPO, digest="b" * 40)


def test_list_files_hides_the_manifest_and_resolves(s3_env):
    files, _, ref, _ = s3_env
    assert sorted(_s3.list_files(ref)) == sorted(files)
    ok, msg = _s3.revision_resolves(ref)
    assert ok and "4 files" in msg


def test_download_full_verifies_bytes_and_caches(s3_env):
    files, client, ref, tmp_path = s3_env
    dest = Path(_s3.download_full(ref))
    assert dest == tmp_path / "s3" / BUCKET / RID / ref.digest.replace(":", "_")
    assert sorted(p.name for p in dest.iterdir()) == sorted(files)
    first_pass = client.downloads
    assert Path(_s3.download_full(ref)) == dest  # verified marker short-circuits
    assert client.downloads == first_pass


def test_download_full_rejects_tampered_bytes(s3_env):
    _, client, ref, _ = s3_env
    client.objects[f"models/registrations/{RID}/config.json"] = b"tampered"
    with pytest.raises(ArtifactIntegrityError, match="digest mismatch"):
        _s3.download_full(ref)
    assert not _s3._marker(_s3._cache_dir(ref)).exists()


def test_download_config_fetches_only_config_files(s3_env):
    _, _, ref, _ = s3_env
    dest = Path(_s3.download_config(ref))
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["config.json", "model.safetensors.index.json"]


def test_safetensors_dtypes_reads_headers_with_ranged_gets(s3_env):
    _, _, ref, _ = s3_env
    assert safetensors_dtypes(ref) == {
        "model-00001-of-00002.safetensors": {"BF16"},
        "model-00002-of-00002.safetensors": {"BF16"},
    }


def test_make_room_protects_private_store_models(s3_env, monkeypatch):
    _, _, ref, tmp_path = s3_env
    protected_dir = tmp_path / "s3" / BUCKET / RID / ("sha256_" + "c" * 64)
    victim_dir = tmp_path / "hf" / "alice" / "model" / ("d" * 40)
    for directory in (protected_dir, victim_dir):
        directory.mkdir(parents=True)
        (directory / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(download, "MAX_CACHED_MODELS", 1)
    download.make_room(ref, protected_repos=[REPO])
    assert protected_dir.exists()
    assert not victim_dir.exists()


def test_download_rejects_path_traversal_object_keys(s3_env):
    # a miner-controlled object whose relative name escapes the model dir
    _, client, ref, _ = s3_env
    client.objects[f"models/registrations/{RID}/../../../etc/evil"] = b"pwned"
    with pytest.raises(ValueError, match="escapes the model directory"):
        _s3.download_full(ref)


def test_check_repo_rejects_forbidden_files_before_download(s3_env, monkeypatch):
    # "push malicious scripts": the file allowlist runs on our list_files output
    from model_validation.validate.repo import check as check_repo

    _, client, ref, _ = s3_env
    client.objects[f"models/registrations/{RID}/backdoor.py"] = b"import os"
    ok, msg = check_repo(_s3.list_files(ref))
    assert not ok and "backdoor.py" in msg


def test_eval_resolver_downloads_and_verifies_private_store_refs(tmp_path, monkeypatch):
    files = _files_with_real_safetensors()
    client = FakeS3Client(files)

    class FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def client(self, service, **kwargs):
            client.client_kwargs = {**self.kwargs, **kwargs}
            return client

    monkeypatch.setattr("boto3.session.Session", FakeSession)
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ro-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "ro-secret")

    resolver = ModelArtifactResolver(
        RemoteSettings(model_cache_dir=str(tmp_path), use_canonical_model_config=False)
    )
    ref = f"{REPO}@sha256:{_digest(files)}"
    resolved = resolver.resolve(ref)
    assert resolved.source == "s3"
    resolved_dir = Path(resolved.local_path)
    names = [p.name for p in resolved_dir.iterdir() if p.name != ".albedo-model-cache.json"]
    assert sorted(names) == sorted(files)
    assert client.client_kwargs["endpoint_url"] == "https://account.r2.cloudflarestorage.com"
    assert client.client_kwargs["aws_access_key_id"] == "ro-key"
    # second resolve is a cache hit
    assert resolver.resolve(ref).cache_hit

    # a tampered store is refused and the cache is not poisoned
    client.objects[f"models/registrations/{RID}/config.json"] = b"tampered"
    tampered = ModelArtifactResolver(
        RemoteSettings(model_cache_dir=str(tmp_path / "fresh"), use_canonical_model_config=False)
    )
    with pytest.raises(ArtifactIntegrityError):
        tampered.resolve(ref)
