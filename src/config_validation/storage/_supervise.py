"""Supervised, killable model downloads.

``snapshot_download()`` runs in-process and blocks on the network with no way to
interrupt it — a wedged transfer (dead CDN socket, hung xet worker) hangs the
calling thread forever, and neither ``asyncio.wait_for`` nor a thread pool can
reclaim it. This module runs the download in a child process guarded by a stall
watchdog: the child is killed and the download resumed when its target directory
stops growing, and abandoned (raising, so the caller can retry) after a few
consecutive stalls. Only full-model downloads use this; config-only fetches are
small and stay in-process.
"""
from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 10.0

# Tunable per host via env, read at import so a redeploy picks up changes.
OUT_OF_PROCESS = os.environ.get("ALBEDO_DOWNLOAD_OUT_OF_PROCESS", "1") not in ("0", "false", "False", "")
STALL_SECONDS = float(os.environ.get("ALBEDO_DOWNLOAD_STALL_SECONDS", "180"))
STALL_RETRIES = int(os.environ.get("ALBEDO_DOWNLOAD_STALL_RETRIES", "3"))
# Hippius pulls from decentralized storage — slower, with longer *legitimate* gaps between
# chunks — so it tolerates a wider no-progress window before a kill.
# Worst case (HIPPIUS_STALL_SECONDS * HIPPIUS_STALL_RETRIES = 2400s) is kept under the sanity
# worker's download_timeout_s so this watchdog, not the blunt outer timeout, is what fires
# — bump that outer timeout too if you widen these.
HIPPIUS_STALL_SECONDS = float(os.environ.get("ALBEDO_HIPPIUS_DOWNLOAD_STALL_SECONDS", "1200"))
HIPPIUS_STALL_RETRIES = int(os.environ.get("ALBEDO_HIPPIUS_DOWNLOAD_STALL_RETRIES", "2"))


def _dir_bytes(path: Path) -> int:
    """Bytes *actually written* under ``path`` (allocated blocks, not apparent size).

    hippius_hub preallocates each file to its full size up front, so ``st_size`` jumps to
    the final total and sits flat while data is still streaming in — which the watchdog
    would misread as a stall. ``st_blocks`` reflects blocks actually allocated, so it tracks
    real download progress for both preallocated (hippius) and append-style (HF) writes.
    """
    total = 0
    for item in path.rglob("*"):
        try:
            st = item.stat()
        except OSError:
            continue
        if item.is_file():
            total += st.st_blocks * 512
    return total


def _tail_file(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return "(download log unavailable)"
    text = data.decode("utf-8", errors="replace").strip()
    return text or "(no output captured)"


_ERROR_LINE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception|Interrupt)\b")


def _summarize_error_log(text: str, max_chars: int = 300) -> str:
    """Compress a child's log tail to one human-readable failure reason.

    This string becomes ``fault_message`` and is shown verbatim on the public
    dashboard — keep the final exception line (plus its explanatory follow-up when
    present) instead of a raw traceback.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(no output captured)"
    for idx in range(len(lines) - 1, -1, -1):
        if _ERROR_LINE.match(lines[idx]):
            summary = lines[idx]
            follow = lines[idx + 1] if idx + 1 < len(lines) else ""
            if follow and not follow.startswith(("Please ", "If you ", "For more", "Traceback")):
                summary += " — " + follow
            return summary[:max_chars]
    return lines[-1][:max_chars]


def _spawn(child_call: str, args: list[str], log_path: Path) -> subprocess.Popen:
    handle = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [sys.executable, "-c", child_call, *args],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # Popen holds its own dup of the fd; the parent's copy is no longer needed.
        handle.close()


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def supervise_download(
    *,
    child_call: str,
    args: list[str],
    watch_dir: Path,
    label: str,
    stall_seconds: float | None = None,
    max_attempts: int | None = None,
) -> None:
    """Run ``child_call`` in a child process, killing + resuming it if it stalls.

    ``child_call`` is Python passed to ``python -c`` that re-imports the backend and
    runs its download; ``args`` become the child's ``sys.argv[1:]``. A watchdog samples
    ``watch_dir``'s byte total every ``_HEARTBEAT_INTERVAL_S``; if it stops growing for
    ``stall_seconds`` the child (its own process group) is terminated and the download
    retried, resuming from what already landed on disk. An attempt that changed the
    directory resets the retry budget, so a transfer that keeps making progress is
    resumed as many times as it needs; ``TimeoutError`` is raised only after
    ``max_attempts`` *consecutive* attempts with zero byte progress. ``RuntimeError``
    on a genuine child error — both are retryable infra faults one level up.
    ``stall_seconds`` / ``max_attempts`` default to the HF-tuned module globals; the
    Hippius backend passes its own wider ones.
    """
    stall = STALL_SECONDS if stall_seconds is None else stall_seconds
    log_path = watch_dir.parent / f"{watch_dir.name}.download.log"
    attempts = max(1, STALL_RETRIES if max_attempts is None else max_attempts)
    attempt = 0
    fruitless = 0
    while True:
        attempt += 1
        baseline = _dir_bytes(watch_dir)
        proc = _spawn(child_call, args, log_path)
        start = time.monotonic()
        last_bytes = baseline
        last_progress = start
        progressed = False
        stalled = False
        while proc.poll() is None:
            time.sleep(_HEARTBEAT_INTERVAL_S)
            current = _dir_bytes(watch_dir)
            now = time.monotonic()
            log.info(
                "download %s attempt=%d elapsed=%.0fs bytes=%d",
                label, attempt, now - start, current,
            )
            if current != last_bytes:
                last_bytes = current
                last_progress = now
                progressed = True
            elif now - last_progress >= stall:
                log.warning(
                    "download %s stalled attempt=%d bytes=%d no_progress=%.0fs — killing",
                    label, attempt, current, now - last_progress,
                )
                _terminate(proc)
                stalled = True
                break
        if stalled:
            fruitless = 0 if progressed else fruitless + 1
            if fruitless >= attempts:
                raise TimeoutError(
                    f"download of {label} made no progress for {stall:.0f}s "
                    f"across {attempts} consecutive attempts"
                )
            continue
        if proc.returncode == 0:
            log_path.unlink(missing_ok=True)
            return
        detail = _summarize_error_log(_tail_file(log_path, 4000))
        raise RuntimeError(f"download of {label} exited {proc.returncode}: {detail}")
