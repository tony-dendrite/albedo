"""Pure, side-effect-free statistics for pairwise per-metric duel scoring."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_lcb(
    deltas: list[float],
    *,
    resamples: int,
    alpha: float,
    rng_seed: bytes,
) -> tuple[float, float, float]:
    """Return (mean, lcb_at_alpha, standard_error) via bootstrap resampling."""
    arr = np.asarray(deltas, dtype=np.float64)
    n   = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean = float(arr.mean())
    se   = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    entropy = int.from_bytes(rng_seed[:8], "little")
    rng     = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy=entropy)))
    boot    = arr[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    lcb     = float(np.quantile(boot, alpha))

    return mean, lcb, se


def judge_mean(metric_scores: dict[str, float]) -> float:
    """Mean of the per-metric scores for one judge (0.0–1.0)."""
    if not metric_scores:
        return 0.0
    return sum(metric_scores.values()) / len(metric_scores)


def aggregate_duel(
    per_judge_metric_scores: dict[str, dict[str, list[float]]],
) -> tuple[float, float, dict[str, float], dict[str, float], str]:
    """Aggregate a duel metric-first (challenger perspective, 0–100).

    Order of means:
      1. per judge, per metric : metric_mean[j][k] = mean over the N scored tasks
      2. per judge             : judge_score[j]    = mean of that judge's 5 metric means
      3. across judges         : challenger_score  = mean of the judge scores

    king_score = 100 - challenger_score (the two always sum to 100).

    Args:
        per_judge_metric_scores: judge_model -> {metric -> [score per scored turn]},
            where each score is 1.0 (challenger) / 0.5 (draw) / 0.0 (king). A judge's
            parse is all-or-nothing per turn, so its 5 metric lists are equal length.

    Returns:
        (challenger_score, king_score, by_judge, by_metric, winner)
        by_judge[j]  = judge_score[j] * 100
        by_metric[k] = mean over judges of metric_mean[j][k] * 100
        winner is "challenger" | "king" | "tie".

    With this order challenger_score == mean(by_judge) == mean(by_metric) exactly.
    """
    # Preserve metric order as first seen across judges.
    metric_order: list[str] = []
    for md in per_judge_metric_scores.values():
        for k in md:
            if k not in metric_order:
                metric_order.append(k)

    metric_mean: dict[str, dict[str, float]] = {}   # judge -> {metric: mean over tasks}
    judge_score: dict[str, float] = {}              # judge -> mean of its 5 metric means
    for jm, md in per_judge_metric_scores.items():
        means = {k: sum(vs) / len(vs) for k, vs in md.items() if vs}
        if not means:
            continue
        metric_mean[jm] = means
        judge_score[jm] = sum(means.values()) / len(means)

    if not judge_score:
        return 0.0, 0.0, {}, {}, "tie"

    challenger = sum(judge_score.values()) / len(judge_score) * 100.0
    king       = 100.0 - challenger

    by_judge = {jm: round(s * 100.0, 4) for jm, s in judge_score.items()}
    by_metric: dict[str, float] = {}
    for k in metric_order:
        vals = [metric_mean[jm][k] for jm in metric_mean if k in metric_mean[jm]]
        if vals:
            by_metric[k] = round(sum(vals) / len(vals) * 100.0, 4)

    challenger = round(challenger, 4)
    king       = round(king, 4)
    winner = ("challenger" if challenger > king
              else "king" if challenger < king else "tie")

    return challenger, king, by_judge, by_metric, winner
