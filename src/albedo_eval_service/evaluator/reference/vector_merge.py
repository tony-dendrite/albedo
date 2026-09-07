from __future__ import annotations

from typing import Any

from .questions import normalise_span

MERGE_SKELETON = """You compare several independently written lists of "milestones" for the SAME \
coding task. Each list was written by the same extractor on a different attempt, from the same \
reference trajectories. Group the statements that assert the SAME underlying fact, change, or check.

Rules:
- Two statements belong together when they establish the same thing about the code or the task, \
even if worded differently, at different length, or with one naming a detail the other omits. A \
different ROUTE to the same fact is the same milestone.
- Two statements are different when they establish different facts (e.g. "where the defect is" \
versus "how the output buffer is built"), or when one is the fix and the other is the check of \
the fix.
- A statement that combines two facts another list splits goes in the cluster of its MAIN fact.
- Every statement appears in exactly one cluster. A cluster may have one member.
Output ONLY strict JSON, no prose: {"clusters":[{"label":"<5-12 word name of the fact>",\
"members":["A1","B3",...]}]}"""


def merge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "members": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["label", "members"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clusters"],
        "additionalProperties": False,
    }


def reading_label(reading: int) -> str:
    return chr(ord("A") + reading)


def member_id(reading: int, index: int) -> str:
    return f"{reading_label(reading)}{index + 1}"


def _parse_member(member: str, readings: list[list[dict[str, Any]]]) -> tuple[int, int] | None:
    member = str(member or "").strip()
    if len(member) < 2 or not member[0].isalpha() or not member[1:].isdigit():
        return None
    reading = ord(member[0].upper()) - ord("A")
    index = int(member[1:]) - 1
    if not (0 <= reading < len(readings) and 0 <= index < len(readings[reading])):
        return None
    return reading, index


def build_merge_messages(
    *, problem: str, readings: list[list[dict[str, Any]]]
) -> list[dict[str, str]]:
    blocks = []
    for reading, milestones in enumerate(readings):
        lines = [
            f"{member_id(reading, index)}. [{m.get('category', '?')}] "
            f"{str(m.get('statement', ''))[:400]}"
            for index, m in enumerate(milestones)
        ]
        blocks.append(
            f"LIST {reading_label(reading)} ({len(milestones)} statements):\n" + "\n".join(lines)
        )
    user = f"TASK (abridged):\n{problem[:1500]}\n\n" + "\n\n".join(blocks) + "\n\nGroup them."
    return [
        {"role": "system", "content": MERGE_SKELETON},
        {"role": "user", "content": user},
    ]


def exact_groups(readings: list[list[dict[str, Any]]]) -> list[list[tuple[int, int]]]:
    """Clusters by normalised statement text - the alignment no model is needed for."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for reading, milestones in enumerate(readings):
        for index, milestone in enumerate(milestones):
            key = normalise_span(
                str(milestone.get("statement") or milestone.get("text") or "")
            ).lower()
            groups.setdefault(key, []).append((reading, index))
    return list(groups.values())


def readings_reached(cluster: list[tuple[int, int]]) -> int:
    return len({reading for reading, _ in cluster})


def needs_alignment(
    clusters: list[list[tuple[int, int]]], readings: list[list[dict[str, Any]]]
) -> bool:
    """Whether exact grouping left a statement that might be another reading's fact reworded."""
    return len(readings) > 1 and any(readings_reached(c) < len(readings) for c in clusters)


def parse_clusters(
    payload: Any, readings: list[list[dict[str, Any]]]
) -> list[list[tuple[int, int]]] | None:
    """The aligner's clusters as (reading, index) pairs, made total and disjoint.

    Unknown ids are ignored, a member named twice stays in its first cluster, and a milestone the
    aligner forgot becomes its own cluster - a forgotten fact must not silently vanish. None only
    when the payload carries no clusters at all.
    """
    raw = payload.get("clusters") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return None
    seen: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []
    for cluster in raw:
        members = cluster.get("members") if isinstance(cluster, dict) else cluster
        group: list[tuple[int, int]] = []
        for member in members or []:
            parsed = _parse_member(str(member), readings)
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            group.append(parsed)
        if group:
            clusters.append(group)
    if not clusters:
        return None
    for reading, milestones in enumerate(readings):
        for index in range(len(milestones)):
            if (reading, index) not in seen:
                clusters.append([(reading, index)])
    return clusters


def merge_vectors(
    readings: list[list[dict[str, Any]]],
    clusters: list[list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """One vector from K validated readings: one milestone per cluster, best-evidenced member.

    The representative is the member whose evidence covers the most runs (earliest reading on a
    tie); its statement, category and necessity are the merged milestone's. Evidence is pooled
    across members, one entry per run, so consensus is every run that reached the fact in any
    reading. `depends_on` follows the representative's own dependencies into the merged ids.
    Order is the mean relative position of the members, which is the causal order the readings
    agree on.
    """

    def milestone(ref: tuple[int, int]) -> dict[str, Any]:
        return readings[ref[0]][ref[1]]

    kept = list(clusters)

    def position(cluster: list[tuple[int, int]]) -> float:
        return sum(index / max(1, len(readings[reading])) for reading, index in cluster) / len(
            cluster
        )

    kept.sort(key=position)
    cluster_of: dict[tuple[int, str], int] = {}
    for number, cluster in enumerate(kept):
        for ref in cluster:
            cluster_of[(ref[0], str(milestone(ref).get("id")))] = number

    merged: list[dict[str, Any]] = []
    for number, cluster in enumerate(kept):
        ordered = sorted(cluster)
        representative = max(
            ordered, key=lambda ref: (len(milestone(ref).get("consensus") or []), -ref[0], -ref[1])
        )
        rep = milestone(representative)
        evidence: list[dict[str, Any]] = []
        runs: set[int] = set()
        for ref in [representative] + [r for r in ordered if r != representative]:
            for entry in milestone(ref).get("evidence") or []:
                run = int(entry.get("run") or 0)
                if run in runs:
                    continue
                runs.add(run)
                evidence.append(entry)
        depends_on = []
        for dep in rep.get("depends_on") or []:
            target = cluster_of.get((representative[0], str(dep)))
            if target is not None and target != number:
                depends_on.append(f"m{target + 1}")
        merged.append(
            {
                **rep,
                "id": f"m{number + 1}",
                "consensus": sorted(runs),
                "evidence": evidence,
                "depends_on": depends_on,
                "readings": sorted({reading_label(r) for r, _ in cluster}),
                "merged_from": len(cluster),
            }
        )
    return merged


QUESTION_MERGE_SKELETON = """You compare several independently written lists of yes/no \
checklist QUESTIONS for the same coding task, all written from the same milestones. Group \
questions that test the SAME thing: a judge reading a candidate trajectory would give the same \
answer to both. Different wording, different length, one naming a detail the other omits: same \
group. Different depth (one asks whether the candidate touched an area, another whether it \
established a specific fact there) or different facts: different groups. Every question in \
exactly one group; a group may have one member.
Output ONLY strict JSON, no prose: \
{"clusters":[{"label":"<short name>","members":["A1","B3",...]}]}"""


def build_question_merge_messages(readings: list[list[dict[str, Any]]]) -> list[dict[str, str]]:
    blocks = []
    for reading, questions in enumerate(readings):
        lines = [
            f"{member_id(reading, index)}. ({q.get('milestone')}, rung {q.get('rung')}) "
            f"{q.get('text', '')}"
            for index, q in enumerate(questions)
        ]
        blocks.append(
            f"LIST {reading_label(reading)} ({len(questions)} questions):\n" + "\n".join(lines)
        )
    return [
        {"role": "system", "content": QUESTION_MERGE_SKELETON},
        {"role": "user", "content": "\n\n".join(blocks) + "\n\nGroup them."},
    ]


def select_questions(
    readings: list[list[dict[str, Any]]],
    clusters: list[list[tuple[int, int]]],
    ladder_min: int,
) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    spare: dict[str, list[dict[str, Any]]] = {}
    for cluster in sorted(clusters, key=lambda c: min(c)):
        representative = readings[min(cluster)[0]][min(cluster)[1]]
        target = by_id if readings_reached(cluster) >= min(2, len(readings)) else spare
        target.setdefault(str(representative.get("milestone")), []).append(representative)
    for mid, extra in spare.items():
        group = by_id.setdefault(mid, [])
        while len(group) < ladder_min and extra:
            group.append(extra.pop(0))
    for group in by_id.values():
        group.sort(key=lambda q: int(q.get("rung") or 0))
    return by_id
