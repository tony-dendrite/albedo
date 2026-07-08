#!/usr/bin/env python3
"""Dataset tooling: download the pinned sources, or author a new manifest (prepare/manifest)."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from loguru import logger

from albedo.sampling import _SHARD_RE  # noqa: E402
from albedo.settings import get_settings

# ── download: fetch the PINNED manifest + sources (deploy-time) ──────────────


def _fetch_manifest(uri: str) -> bytes:
    # Pulls the manifest object from S3 (s3://bucket/key) using the ALBEDO_S3_* creds.
    if not uri.startswith("s3://"):
        sys.exit(f"manifest uri must be s3://bucket/key, got: {uri}")
    bucket, key = uri[5:].split("/", 1)
    s3 = get_settings().s3
    client = boto3.client(
        "s3",
        endpoint_url=s3.endpoint or "https://s3.hippius.com",
        aws_access_key_id=s3.access_key,
        aws_secret_access_key=s3.secret_key,
    )
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _download_source(repo: str, glob: str, dest: Path) -> None:
    # Downloads one source's parquet shards into dest, retrying through HF 429 rate limits.
    from huggingface_hub import snapshot_download

    for attempt in range(1, 21):
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(dest),
                allow_patterns=[glob],
                token=os.environ.get("HF_TOKEN") or None,
            )
            return
        except Exception as exc:  # noqa: BLE001 - retry on transient HF rate limits
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                logger.warning(f"[{repo}] rate-limited (attempt {attempt}); sleeping 60s")
                time.sleep(60)
                continue
            raise
    sys.exit(f"exhausted retries downloading {repo}")


def _resolve_manifest(root: Path, expected: str) -> bytes:
    # Prefers a hash-matching local copy (dataset root, then the repo-shipped one) over S3.
    candidates = [
        root / "manifest.json",
        Path(__file__).resolve().parent.parent / "assets" / "dataset-manifest.json",
    ]
    for path in candidates:
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == expected:
                logger.info(f"using local manifest {path} (hash verified)")
                return data
    uri = get_settings().eval.dataset_manifest_uri
    if not uri:
        sys.exit("no hash-matching local manifest and ALBEDO_EVAL_DATASET_MANIFEST_URI unset")
    data = _fetch_manifest(uri)
    got = hashlib.sha256(data).hexdigest()
    if got != expected:
        sys.exit(f"manifest hash mismatch: got {got}, expected {expected}")
    return data


def cmd_download(args) -> None:
    # Resolves the pinned manifest (local copy or S3), then downloads each source into the layout.
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    e = get_settings().eval
    if not e.dataset_manifest_hash:
        sys.exit("ALBEDO_EVAL_DATASET_MANIFEST_HASH is required in .env")

    data = _resolve_manifest(root, e.dataset_manifest_hash.removeprefix("sha256:"))
    (root / "manifest.json").write_bytes(data)
    manifest = json.loads(data)
    logger.info(
        f"manifest ok ({manifest.get('version')}), "
        f"sources: {[s['name'] for s in manifest['sources']]}"
    )

    for src in manifest["sources"]:
        name, repo = src["name"], src["repo"]
        glob = src.get("shard_glob", "data/train-*.parquet")
        dest = root / name
        have = len(list(dest.glob(glob)))
        want = len(src.get("shards", []))
        if want and have >= want:
            logger.info(f"[{name}] already complete ({have}/{want})")
            continue
        logger.info(f"[{name}] downloading {repo} -> {dest} (glob {glob})")
        _download_source(repo, glob, dest)
        logger.info(f"[{name}] done ({len(list(dest.glob(glob)))} shards)")

    logger.info(f"all sources present under {root}")


# ── prepare/manifest: author a NEW dataset pin (rare) ────────────────────────

log = logging.getLogger("prepare_datasets")


SOURCES: dict[str, dict[str, str]] = {
    "swe-zero": {
        "repo": "AlienKevin/SWE-ZERO-12M-trajectories",
        "shard_glob": "data/train-*.parquet",
    },
    "mini-coder": {"repo": "ricdomolm/mini-coder-trajs-400k", "shard_glob": "data/train-*.parquet"},
}


def _enable_fast_transfer() -> None:

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _expected_parquet_shards(repo_id: str, shard_glob: str) -> set[str]:
    """Repo-relative paths of every shard matching shard_glob (one metadata call, no
    file content) so we can fetch only what is missing."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return {f for f in files if fnmatch.fnmatch(f, shard_glob)}


def _local_parquet_shards(dest: Path, shard_glob: str) -> set[str]:
    """Repo-relative paths of shards already present under dest/ (matched by shard_glob)."""
    subdir, _, name_pat = shard_glob.rpartition("/")
    data_dir = dest / subdir if subdir else dest
    if not data_dir.is_dir():
        return set()
    prefix = f"{subdir}/" if subdir else ""
    return {f"{prefix}{p.name}" for p in data_dir.glob(name_pat)}


def download_source(
    name: str, repo_id: str, shard_glob: str, root: Path, *, force: bool, max_workers: int
) -> Path:
    from huggingface_hub import hf_hub_download

    dest = root / name

    expected = _expected_parquet_shards(repo_id, shard_glob)
    if not expected:
        raise RuntimeError(f"{name}: no shards in repo {repo_id} matching {shard_glob!r}")
    present = _local_parquet_shards(dest, shard_glob)
    to_fetch = sorted(expected) if force else sorted(expected - present)

    if not to_fetch:
        log.info(
            "%s: complete (%d/%d shards present) — skipping", name, len(present), len(expected)
        )
        return dest
    log.info(
        "%s: %d/%d present, downloading %d missing -> %s (%d parallel workers)",
        name,
        len(present),
        len(expected),
        len(to_fetch),
        dest,
        max_workers,
    )

    def _one(rel: str) -> None:
        # Resumes partial files and skips already-complete ones; writes to dest/<rel>.
        # HF_TOKEN (if set) is picked up from the env automatically.
        hf_hub_download(
            repo_id,
            rel,
            repo_type="dataset",
            local_dir=str(dest),
            force_download=force,
            token=os.environ.get("HF_TOKEN"),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, rel) for rel in to_fetch]
        for fut in as_completed(futures):
            fut.result()  # re-raise the first download failure
            done += 1
            if done % 50 == 0 or done == len(to_fetch):
                log.info("%s: %d/%d downloaded", name, done, len(to_fetch))

    still_missing = expected - _local_parquet_shards(dest, shard_glob)
    if still_missing:
        raise RuntimeError(
            f"{name}: {len(still_missing)} shard(s) still missing after download, "
            f"e.g. {sorted(still_missing)[:3]}"
        )
    log.info("%s: done (%d shards)", name, len(expected))
    return dest


def _upload_manifest_to_hippius(manifest_path: Path, key: str) -> str:
    """Upload manifest.json to Hippius S3 (public-read) and return the s3:// URI.

    Uploads the exact on-disk bytes so the object's sha256 equals the pinned manifest hash.
    Reuses the validators' ALBEDO_S3_* Hippius credentials (auto-loaded from albedo/.env).
    """
    import boto3
    from botocore.config import Config

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hippius_validation import config as hv

    if not (hv.S3_BUCKET and hv.S3_ACCESS_KEY and hv.S3_SECRET_KEY):
        raise SystemExit(
            "--upload needs Hippius S3 credentials: set ALBEDO_S3_BUCKET, ALBEDO_S3_ACCESS_KEY "
            "and ALBEDO_S3_SECRET_KEY (in albedo/.env)."
        )

    body = manifest_path.read_bytes()
    client = boto3.client(
        "s3",
        endpoint_url=hv.S3_ENDPOINT,
        aws_access_key_id=hv.S3_ACCESS_KEY,
        aws_secret_access_key=hv.S3_SECRET_KEY,
        region_name="decentralized",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )
    client.put_object(
        Bucket=hv.S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
        ACL="public-read",
    )
    log.info("manifest sha256: %s", hashlib.sha256(body).hexdigest())
    return f"s3://{hv.S3_BUCKET}/{key}"


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


sys.path.insert(0, str(Path(__file__).resolve().parent))


DEFAULT_VERSION = "swe-zero+mini-coder-v1"


def _parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        weights[name.strip()] = float(value)
    if not weights:
        raise SystemExit("--weights must be like 'swe-zero=0.7,mini-coder=0.3'")
    return weights


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_source(name: str, weight: float, root: Path, *, max_workers: int = 8) -> dict:
    if name not in SOURCES:
        raise SystemExit(f"{name}: unknown source (not in prepare_datasets.SOURCES)")
    repo = SOURCES[name]["repo"]
    shard_glob = SOURCES[name]["shard_glob"]
    data_dir = root / name / "data"
    name_pattern = shard_glob.rsplit("/", 1)[-1]
    files = sorted(data_dir.glob(name_pattern), key=lambda p: p.name)
    if not files:
        raise SystemExit(
            f"{name}: no parquet shards under {data_dir} (run prepare_datasets.py first)"
        )

    def _shard(path: Path) -> dict:
        shard_path = f"{name}/data/{path.name}"
        if not _SHARD_RE.match(shard_path):
            raise SystemExit(
                f"{name}: shard name {shard_path!r} is not a valid (<source>/)data/train-*.parquet"
            )
        rows = pq.ParquetFile(path).metadata.num_rows
        return {"path": shard_path, "rows": rows, "sha256": _sha256(path)}

    # Hashing reads every shard's bytes; overlap that I/O across shards.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shards = list(pool.map(_shard, files))  # map preserves input (sorted) order
    total_rows = sum(s["rows"] for s in shards)

    return {
        "name": name,
        "repo": repo,
        "shard_glob": shard_glob,
        "weight": weight,
        "shards": shards,
        "total_rows": total_rows,
    }


def build_manifest_dict(
    root: Path, weights: dict[str, float], *, version: str = DEFAULT_VERSION, max_workers: int = 8
) -> dict:
    sources = [
        _build_source(name, weight, root, max_workers=max_workers)
        for name, weight in weights.items()
    ]
    sources.sort(key=lambda s: s["name"])
    return {
        "version": version,
        "sources": sources,
        "total_rows": sum(s["total_rows"] for s in sources),
    }


def write_manifest(
    root: Path,
    weights: dict[str, float],
    *,
    out_path: Path | None = None,
    version: str = DEFAULT_VERSION,
    max_workers: int = 8,
) -> tuple[Path, dict, str]:
    """Build the combined manifest, write it locally, and return (path, manifest, sha256)."""
    out_path = Path(out_path) if out_path else Path(root) / "manifest.json"
    manifest = build_manifest_dict(Path(root), weights, version=version, max_workers=max_workers)
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    out_path.write_bytes(payload)
    return out_path, manifest, hashlib.sha256(payload).hexdigest()


def print_manifest_summary(out_path: Path, manifest: dict, digest: str) -> None:
    sources = manifest["sources"]
    print(f"wrote {out_path} ({manifest['total_rows']} rows across {len(sources)} sources)")
    for source in sources:
        print(
            f"  {source['name']}: weight={source['weight']} rows={source['total_rows']} "
            f"shards={len(source['shards'])}"
        )
    print()
    print(f"sha256: {digest}")
    print()
    print("Update these to pin the new manifest:")
    print(f"  ALBEDO_EVAL_DATASET_MANIFEST_HASH={digest}")
    print(f"  SANITY_DISPATCH_DATASET_MANIFEST_HASH={digest}")
    print("  ALBEDO_EVAL_SAMPLING_ALGO=swe-zero-multi-source-sample-v1")
    print("  src/albedo_eval_service/config.py  -> dataset_manifest_hash default")
    print("  src/sanity_service/settings.py     -> dataset_manifest_hash default")


def cmd_prepare(argv) -> None:
    parser = argparse.ArgumentParser(
        description="Download eval datasets from HF and build the combined manifest locally."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dir to download <source>/data/*.parquet into and write manifest.json (local only).",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help=f"Comma-separated source names to fetch (default: all of {','.join(SOURCES)}).",
    )
    parser.add_argument(
        "--weights",
        default="swe-zero=0.7,mini-coder=0.3",
        help="Per-source manifest weights (default: swe-zero=0.7,mini-coder=0.3).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if shards already exist."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent files downloaded in parallel (default: 16). Raise for many small shards.",
    )
    parser.add_argument(
        "--skip-manifest", action="store_true", help="Only download; do not build manifest.json."
    )
    parser.add_argument(
        "--out", default=None, help="Manifest output path (default: <dataset-root>/manifest.json)."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload only manifest.json to Hippius S3 (ALBEDO_S3_* creds); not the datasets.",
    )
    parser.add_argument(
        "--upload-key",
        default="datasets/manifest.json",
        help="Destination key in the bucket for --upload (default: datasets/manifest.json).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _enable_fast_transfer()

    root = Path(args.dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; known: {list(SOURCES)}")

    for name in names:
        meta = SOURCES[name]
        download_source(
            name,
            meta["repo"],
            meta["shard_glob"],
            root,
            force=args.force,
            max_workers=args.max_workers,
        )

    out_path = Path(args.out) if args.out else root / "manifest.json"

    if args.skip_manifest:
        log.info("skipping manifest build (--skip-manifest)")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_manifest import _parse_weights, print_manifest_summary, write_manifest

        all_weights = _parse_weights(args.weights)
        missing = [n for n in names if n not in all_weights]
        if missing:
            raise SystemExit(f"no --weights entry for downloaded source(s): {missing}")
        weights = {n: all_weights[n] for n in names}

        out_path, manifest, digest = write_manifest(
            root, weights, out_path=out_path, max_workers=args.max_workers
        )
        print_manifest_summary(out_path, manifest, digest)

    if args.upload:
        if not out_path.exists():
            raise SystemExit(
                f"--upload: no manifest at {out_path} (build one first, or drop --skip-manifest)."
            )
        log.info("uploaded manifest -> %s", _upload_manifest_to_hippius(out_path, args.upload_key))


def cmd_manifest(argv) -> None:
    parser = argparse.ArgumentParser(description="Build the combined eval dataset manifest.")
    parser.add_argument(
        "--dataset-root", required=True, help="Root dir holding <source>/data/*.parquet."
    )
    parser.add_argument("--weights", required=True, help="e.g. 'swe-zero=0.7,mini-coder=0.3'")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Manifest version label.")
    parser.add_argument("--out", default=None, help="Output path (default: <root>/manifest.json).")
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Parallel shard-hashing workers (default: 8)."
    )
    args = parser.parse_args(argv)

    out_path, manifest, digest = write_manifest(
        Path(args.dataset_root),
        _parse_weights(args.weights),
        out_path=Path(args.out) if args.out else None,
        version=args.version,
        max_workers=args.max_workers,
    )
    print_manifest_summary(out_path, manifest, digest)


def main() -> None:
    # One entrypoint: datasets.py {download <root> | prepare ... | manifest ...}.
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="fetch pinned manifest + all sources")
    dl.add_argument("root")
    sub.add_parser("prepare", add_help=False)
    sub.add_parser("manifest", add_help=False)
    args, rest = parser.parse_known_args()
    if args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "prepare":
        cmd_prepare(rest)
    else:
        cmd_manifest(rest)


if __name__ == "__main__":
    main()
