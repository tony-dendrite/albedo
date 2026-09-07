"""Several readings of the same runs become one vector: majority by fact, evidence pooled."""

from __future__ import annotations

from albedo_eval_service.evaluator.reference.vector_merge import (
    build_merge_messages,
    build_question_merge_messages,
    exact_groups,
    merge_vectors,
    needs_alignment,
    parse_clusters,
    select_questions,
)


def _m(id, statement, runs=(1,), category="explore", depends_on=(), step=1):
    return {
        "id": id,
        "category": category,
        "statement": statement,
        "necessary": True,
        "consensus": list(runs),
        "depends_on": list(depends_on),
        "evidence": [{"run": r, "step": step, "source": "output", "span": f"s{r}"} for r in runs],
    }


def test_identical_readings_need_no_aligner_and_merge_to_themselves():
    reading = [_m("m1", "the defect is in compute_offset"), _m("m2", "the tests are exercised")]
    readings = [reading, reading, reading]
    clusters = exact_groups(readings)

    assert not needs_alignment(clusters, readings)
    merged = merge_vectors(readings, clusters)
    assert [m["statement"] for m in merged] == [m["statement"] for m in reading]
    assert all(m["readings"] == ["A", "B", "C"] and m["merged_from"] == 3 for m in merged)


def test_reworded_facts_merge_to_one_each_and_a_lone_fact_is_kept_in_causal_order():
    a = [_m("m1", "fact one"), _m("m2", "fact two")]
    b = [_m("m1", "fact one, worded otherwise"), _m("m2", "fact two"), _m("m3", "a detour")]
    c = [_m("m1", "fact two"), _m("m2", "fact one again")]
    readings = [a, b, c]
    assert needs_alignment(exact_groups(readings), readings)

    clusters = parse_clusters(
        {
            "clusters": [
                {"label": "one", "members": ["A1", "B1", "C2"]},
                {"label": "two", "members": ["A2", "B2", "C1"]},
            ]
        },
        readings,
    )
    # the detour B3 was forgotten by the aligner: it becomes its own cluster and is still kept
    assert [(1, 2)] in clusters
    merged = merge_vectors(readings, clusters)

    assert [m["merged_from"] for m in merged] == [3, 3, 1]
    assert [m["statement"] for m in merged] == ["fact one", "fact two", "a detour"]
    assert [m["id"] for m in merged] == ["m1", "m2", "m3"]


def test_the_best_evidenced_member_represents_the_fact_and_evidence_is_pooled():
    a = [_m("m1", "the defect site", runs=(1,))]
    b = [_m("m1", "the defect site is compute_offset", runs=(2, 3))]
    c = [_m("m1", "the defect site", runs=(1,))]
    readings = [a, b, c]
    merged = merge_vectors(readings, [[(0, 0), (1, 0), (2, 0)]])

    assert len(merged) == 1
    assert merged[0]["statement"] == "the defect site is compute_offset"
    assert merged[0]["consensus"] == [1, 2, 3]
    assert sorted(e["run"] for e in merged[0]["evidence"]) == [1, 2, 3]


def test_dependencies_follow_the_representative_into_the_merged_ids():
    a = [
        _m("m1", "the site"),
        _m("m2", "the fix", category="action", depends_on=("m1", "m9")),
    ]
    b = [_m("x", "the site"), _m("y", "the fix", category="action", depends_on=("x",))]
    readings = [a, b]
    merged = merge_vectors(readings, [[(0, 0), (1, 0)], [(0, 1), (1, 1)]])

    assert [m["id"] for m in merged] == ["m1", "m2"]
    assert merged[1]["depends_on"] == ["m1"]
    assert merged[0]["depends_on"] == []


def test_parse_clusters_ignores_unknown_ids_and_keeps_the_first_placement():
    readings = [[_m("m1", "a")], [_m("m1", "b")]]
    clusters = parse_clusters(
        {
            "clusters": [
                {"label": "", "members": ["A1", "Z9", "A1"]},
                {"label": "", "members": ["B1", "A1"]},
            ]
        },
        readings,
    )
    assert clusters == [[(0, 0)], [(1, 0)]]
    assert parse_clusters({"clusters": []}, readings) is None
    assert parse_clusters("nonsense", readings) is None


def test_merge_messages_show_every_reading_under_its_label():
    readings = [[_m("m1", "a fact")], [_m("m1", "another")]]
    messages = build_merge_messages(problem="fix the offset", readings=readings)
    user = messages[-1]["content"]
    assert "LIST A (1 statements):\nA1. [explore] a fact" in user
    assert "LIST B (1 statements):\nB1. [explore] another" in user


def _q(milestone, rung, text):
    return {"milestone": milestone, "rung": rung, "text": text, "tag": "reference:milestone"}


def test_questions_two_readings_asked_survive_and_a_thin_milestone_is_filled():
    a = [
        _q("m1", 1, "touched the file"),
        _q("m1", 2, "found the defect"),
        _q("m2", 1, "ran the check"),
    ]
    b = [
        _q("m1", 1, "worked with the file"),
        _q("m1", 2, "located the defect"),
        _q("m2", 1, "a stray one"),
    ]
    c = [
        _q("m1", 1, "opened the file"),
        _q("m1", 3, "fixed it"),
        _q("m2", 1, "exercised the check"),
    ]
    readings = [a, b, c]
    clusters = [[(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1)], [(2, 1)], [(0, 2), (2, 2)], [(1, 2)]]

    by_id = select_questions(readings, clusters, ladder_min=2)

    assert [q["text"] for q in by_id["m1"]] == ["touched the file", "found the defect"]
    assert [q["text"] for q in by_id["m2"]] == ["ran the check", "a stray one"]


def test_question_merge_messages_carry_milestone_and_rung():
    messages = build_question_merge_messages([[_q("m1", 2, "did it")], [_q("m1", 1, "did it too")]])
    assert "A1. (m1, rung 2) did it" in messages[-1]["content"]
    assert "B1. (m1, rung 1) did it too" in messages[-1]["content"]
