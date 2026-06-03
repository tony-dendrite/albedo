"""albedo.validator.faults — classify eval failures as miner-fault (terminal) vs infra (retry)."""
from __future__ import annotations

# Detail substrings that mark an admission/config failure as a transient fetch
# problem (hub down, 404 mid-propagation) rather than the miner's own fault.
_TRANSIENT_FETCH = ("could not download", "could not list", "materialize", "404")
# Detail substrings that are always the miner's fault regardless of code.
_MINER_DETAIL = ("chal_vllm_start_failed", "chal_injection_detected")
# Failure codes that are inherently infra/transient (retry is warranted).
_INFRA_CODES = frozenset({"infra_failure", "eval_error", "eval_infra", "no_verdict"})
# Failure codes that are inherently the miner's fault (no retry).
_MINER_CODES = frozenset({
    "duplicate", "spoof_rejected", "identity_mismatch",
    "not_registered", "challenger_rejected", "no_king",
})


def is_infra_failure(verdict: dict) -> bool:
    """True when the eval server aborted before a meaningful duel ran."""
    if verdict.get("error"):
        return True

    n_done = int(verdict.get("n_done") or 0)
    n_valid = int(verdict.get("n_valid") or 0)
    king_vllm_errors = int(verdict.get("king_vllm_errors") or 0)
    chal_vllm_errors = int(verdict.get("chal_vllm_errors") or 0)

    if king_vllm_errors > 0:
        return True

    if n_done == 0 and n_valid == 0:
        if chal_vllm_errors > 0:
            return False  # challenger's fault
        return True  # nothing ran at all — infra

    return False


def is_miner_fault(code: str, detail: str) -> bool:
    """True when the failure is the miner's own fault and no retry should be granted.

    Transient infrastructure / fetch errors return False so they can be retried.
    Unknown codes default to True (no infinite retry on something we don't understand).
    """
    d = (detail or "").lower()
    if any(s in d for s in _MINER_DETAIL):
        return True
    if code in _INFRA_CODES:
        return False
    if code == "admission_rejected":
        return not any(s in d for s in _TRANSIENT_FETCH)
    if code in _MINER_CODES:
        return True
    return True
