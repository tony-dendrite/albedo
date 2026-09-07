"""The three things between a model's answer and a checklist that no prompt can guarantee.

`validate_vector` decides which milestones are real, `approach_windows` decides what ground each one
is allowed to draw questions from, and `parse_ladder` decides what a rung number means. Each is
mechanical, each silently changes the checklist when it is wrong, and none of them is observable
from a passing eval run.
"""

from __future__ import annotations

import json

from albedo_eval_service.evaluator.reference.prompt_ladder import approach_windows, parse_ladder
from albedo_eval_service.evaluator.reference.prompt_milestones import validate_vector

_FINDING_STEP = {
    "assistant": (
        "THOUGHT: the offset is computed in float_values.py\n\n"
        "```bash\ngrep -n compute_offset float_values.py\n```"
    ),
    "observation": "def compute_offset(row):\n    return row.start - 1",
}
_CHECK_STEP = {
    "assistant": (
        "THOUGHT: run the scope tests\n\n```bash\npython -m pytest tests/test_scope.py -q\n```"
    ),
    "observation": "PASS test_globals\nPassed: 6/6",
}
_RUNS = [{"run": 1, "steps": [_FINDING_STEP, _CHECK_STEP]}]


def _milestone(**overrides):
    base = {
        "id": "m1",
        "category": "explore",
        "statement": "the offset computation is off by one",
        "necessary": True,
        "necessity_reason": "the fix cannot be aimed without it",
        "in_prefix": False,
        "consensus": [1],
        "depends_on": [],
        "evidence": [],
    }
    return {**base, **overrides}


def test_an_explore_the_agent_only_typed_into_its_own_block_is_unreached():
    """An `explore` may not rest on a block the agent authored: content it wrote into its own
    command was assumed, not discovered, so citing it would score the agent's confidence."""
    kept, dropped = validate_vector(
        [
            _milestone(
                evidence=[
                    {
                        "run": 1,
                        "step": 1,
                        "source": "edit",
                        "span": "grep -n compute_offset float_values.py",
                    }
                ]
            )
        ],
        _RUNS,
        task="fix the off-by-one",
    )

    assert kept == []
    reasons = [d["reason"] for d in dropped]
    assert "span_inadmissible_for_explore" in reasons
    assert "unreached" in reasons


def test_an_action_cannot_rest_on_the_agent_saying_it_will_edit():
    """`edit` and `command` mean "inside a code block the agent wrote", so prose may not satisfy
    them. Sharing one space with `stated` let an action be grounded on an intention - the evidence
    then read `source: "edit"` for content the agent only described, which is what saying a thing
    looks like rather than doing it."""
    step = {
        "assistant": (
            "THOUGHT: I will clamp the offset in compute_offset\n\n"
            "```bash\nsed -i 's/row.start - 1/max(0, row.start - 1)/' float_values.py\n```"
        ),
        "observation": "",
    }
    runs = [{"run": 1, "steps": [step]}]
    described = _milestone(
        category="action",
        statement="the offset is clamped at zero",
        evidence=[
            {"run": 1, "step": 1, "source": "edit", "span": "I will clamp the offset"},
        ],
    )
    installed = _milestone(
        category="action",
        statement="the offset is clamped at zero",
        evidence=[
            {"run": 1, "step": 1, "source": "edit", "span": "max(0, row.start - 1)"},
        ],
    )

    rejected, dropped = validate_vector([described], runs, task="fix the off-by-one")
    accepted, _ = validate_vector([installed], runs, task="fix the off-by-one")

    assert rejected == []
    assert "span_inadmissible_for_action" in [d["reason"] for d in dropped]
    assert len(accepted) == 1


def test_a_verification_is_cited_by_its_invocation_and_never_by_what_came_back():
    """The rule the extractor prompt says is broken most often. What an observation reported is the
    environment's answer; the agent's act is the command it aimed."""
    from_output = _milestone(
        category="verification",
        statement="the scope tests are exercised after the change",
        evidence=[{"run": 1, "step": 2, "source": "output", "span": "PASS test_globals"}],
    )
    from_command = _milestone(
        category="verification",
        statement="the scope tests are exercised after the change",
        evidence=[
            {
                "run": 1,
                "step": 2,
                "source": "command",
                "span": "python -m pytest tests/test_scope.py -q",
            }
        ],
    )

    rejected, dropped = validate_vector([from_output], _RUNS, task="fix it")
    accepted, _ = validate_vector([from_command], _RUNS, task="fix it")

    assert rejected == []
    assert "span_inadmissible_for_verification" in [d["reason"] for d in dropped]
    assert len(accepted) == 1


def test_a_span_the_task_already_carries_is_rejected_even_though_the_run_also_shows_it():
    """The given-material floor. The prefix is free to the candidate, so a milestone evidenced by
    something the task handed over measures reading the prompt, not doing the work."""
    kept, dropped = validate_vector(
        [
            _milestone(
                evidence=[
                    {
                        "run": 1,
                        "step": 1,
                        "source": "stated",
                        "span": "the offset is computed in float_values.py",
                    }
                ]
            )
        ],
        _RUNS,
        task="Bug report: the offset is computed in float_values.py and is one too low.",
    )

    assert kept == []
    assert "span_from_task" in [d["reason"] for d in dropped]


def test_a_survivor_cannot_ground_itself_on_a_dropped_milestone_or_on_a_run_it_lost():
    """Two prunings that keep the vector self-consistent: a `depends_on` naming a milestone that did
    not survive is removed, and `consensus` is recomputed to the runs whose evidence actually held,
    so `project_vector` never shows the writer a dependency or a route that is not there."""
    runs = [
        {"run": 1, "steps": [_FINDING_STEP]},
        {"run": 2, "steps": [{"assistant": "THOUGHT: nothing here", "observation": ""}]},
    ]
    optional = _milestone(id="m1", necessary=False)
    dependent = _milestone(
        id="m2",
        category="verification",
        statement="the tests covering the offset are exercised",
        depends_on=["m1"],
        consensus=[1, 2],
        evidence=[
            {
                "run": 1,
                "step": 1,
                "source": "command",
                "span": "grep -n compute_offset float_values.py",
            },
            {"run": 2, "step": 1, "source": "command", "span": "pytest tests/test_offset.py"},
        ],
    )

    kept, dropped = validate_vector([optional, dependent], runs, task="fix it")

    assert [m["id"] for m in kept] == ["m2"]
    assert kept[0]["depends_on"] == []
    assert kept[0]["consensus"] == [1]
    assert "optional" in [d["reason"] for d in dropped]
    assert "span_not_in_corpus" in [d["reason"] for d in dropped]


def test_consecutive_milestones_partition_a_run_rather_than_overlapping():
    """`landed` carries the previous milestone's landing step per run, so each milestone's approach
    is exactly the ground since the last thing that run is known to have reached. Overlapping
    windows would hand the same en-route finding to two milestones and score it twice."""
    steps = [f"step {i}" for i in range(1, 7)]
    landed: dict[int, int] = {}

    first = approach_windows(
        {"evidence": [{"run": 1, "step": 3, "source": "output", "span": "x"}]}, {1: steps}, landed
    )
    second = approach_windows(
        {"evidence": [{"run": 1, "step": 5, "source": "output", "span": "x"}]}, {1: steps}, landed
    )

    assert first[1] == ["step 1", "step 2", "step 3"]
    assert second[1] == ["step 4", "step 5"]


def test_a_milestone_landing_before_the_previous_one_yields_nothing_for_that_run():
    """A vector whose order does not follow step order in some run must yield an empty window, not a
    backwards slice: the reversed slice would silently be the whole rest of the trajectory."""
    steps = [f"step {i}" for i in range(1, 7)]
    landed: dict[int, int] = {}

    approach_windows(
        {"evidence": [{"run": 1, "step": 5, "source": "output", "span": "x"}]}, {1: steps}, landed
    )
    inverted = approach_windows(
        {"evidence": [{"run": 1, "step": 3, "source": "output", "span": "x"}]}, {1: steps}, landed
    )

    assert inverted[1] == []


def test_rungs_are_renumbered_per_milestone_and_unknown_milestones_are_dropped():
    """A rung is only meaningful as a position in its own ladder, so gaps and ties in what the model
    emitted are closed here — the emitted ORDER is kept, its arithmetic is not. A question citing a
    milestone that is not in the vector has nothing behind it and cannot be scored."""
    raw = json.dumps(
        {
            "questions": [
                {"milestone": "m1", "rung": 5, "text": "deepest?", "unearned": "x"},
                {"milestone": "m1", "rung": 2, "text": "shallowest?", "unearned": "x"},
                {"milestone": "m1", "rung": 5, "text": "also deep?", "unearned": "x"},
                {"milestone": "ghost", "rung": 1, "text": "from nowhere?", "unearned": "x"},
            ]
        }
    )

    out = parse_ladder(raw, known_ids={"m1"})

    assert [q["rung"] for q in out] == [1, 2, 3]
    assert [q["text"] for q in out] == ["shallowest?", "deepest?", "also deep?"]
    assert all(q["milestone"] == "m1" for q in out)


def test_a_prefix_claim_the_task_does_not_carry_costs_the_claim_and_not_the_milestone():
    """`in_prefix` decides whether a milestone produces any question at all, and the drop it causes
    is not recoverable — so an unquoted claim used to remove a real milestone silently and for
    good. The claim is now checked against the TASK exactly as evidence is: supported, the
    milestone is given and drops; unsupported, the claim is discarded and the milestone stands."""
    task = 'the blur function panics: assert!(sigma.is_normal(), "Sigma cannot be zero");'
    evidence = [{"run": 1, "step": 1, "source": "output", "span": "def compute_offset(row):"}]

    given = _milestone(in_prefix=True, prefix_quote="assert!(sigma.is_normal()", evidence=evidence)
    invented = _milestone(
        in_prefix=True, prefix_quote="fn nowhere_in_the_task()", evidence=evidence
    )
    unquoted = _milestone(in_prefix=True, prefix_quote="", evidence=evidence)

    kept_given, dropped_given = validate_vector([given], _RUNS, task)
    kept_invented, dropped_invented = validate_vector([invented], _RUNS, task)
    kept_unquoted, _ = validate_vector([unquoted], _RUNS, task)

    assert kept_given == []
    assert [d["reason"] for d in dropped_given] == ["given_by_task"]
    assert len(kept_invented) == 1
    assert kept_invented[0]["prefix_claim_unsupported"] is True
    assert kept_invented[0]["in_prefix"] is False
    assert [d["reason"] for d in dropped_invented] == []
    assert len(kept_unquoted) == 1


def test_a_claims_milestone_may_rest_on_the_reproduction_it_ran():
    """The extractor cites the invocation that put the task's claim to the repository about as
    often as what came back; both are the agent testing the claim, so `command` is admissible."""
    kept, dropped = validate_vector(
        [
            _milestone(
                category="claims",
                statement="the reported scope failure reproduces",
                evidence=[
                    {
                        "run": 1,
                        "step": 2,
                        "source": "command",
                        "span": "python -m pytest tests/test_scope.py -q",
                    }
                ],
            )
        ],
        _RUNS,
        task="fix it",
    )

    assert [m["id"] for m in kept] == ["m1"]
    assert dropped == []


def test_an_action_span_the_task_also_carries_is_still_the_agents_act():
    """The given-material floor is about discovered facts. The task may name the line an agent
    must install - the new message string, the expected behaviour - without installing it, so an
    action or verification span echoing task text is admissible; an explore span is not."""
    runs = [
        {
            "run": 1,
            "steps": [
                {
                    "assistant": (
                        "```python\nmessages['C412'] = 'Unnecessary list comprehension'\n```"
                    ),
                    "observation": "",
                }
            ],
        }
    ]
    task = "Add C412 with the message 'Unnecessary list comprehension' to the messages dict."
    action = _milestone(
        id="m1",
        category="action",
        statement="the C412 message is added to the messages dict",
        evidence=[
            {
                "run": 1,
                "step": 1,
                "source": "edit",
                "span": "messages['C412'] = 'Unnecessary list comprehension'",
            }
        ],
    )
    explore = _milestone(
        id="m2",
        category="explore",
        statement="the messages dict holds the rule texts",
        evidence=[
            {"run": 1, "step": 1, "source": "stated", "span": "Unnecessary list comprehension"}
        ],
    )

    kept, dropped = validate_vector([action, explore], runs, task=task)

    assert [m["id"] for m in kept] == ["m1"]
    assert [d["reason"] for d in dropped if d["id"] == "m2"][0] == "span_from_task"


def test_a_verification_flagged_optional_is_kept():
    """Whether a competent agent "had to" run the check is the counterfactual the extractor answers
    differently on identical input; the check of a fix is never one agent's route."""
    kept, dropped = validate_vector(
        [
            _milestone(
                category="verification",
                necessary=False,
                statement="the scope tests are exercised",
                evidence=[
                    {
                        "run": 1,
                        "step": 2,
                        "source": "command",
                        "span": "python -m pytest tests/test_scope.py -q",
                    }
                ],
            ),
            _milestone(id="m2", necessary=False),
        ],
        _RUNS,
        task="fix it",
    )

    assert [m["id"] for m in kept] == ["m1"]
    assert [d["reason"] for d in dropped] == ["optional"]


def test_a_span_cited_from_the_wrong_run_or_block_is_relocated_not_dropped():
    """Miscounting the run or mislabelling the block kind is the extractor's error, not the
    milestone's. The span is accepted where it really is - under an admissible block kind only -
    and consensus follows the evidence."""
    runs = [
        {"run": 1, "steps": [{"assistant": "THOUGHT: nothing here", "observation": ""}]},
        {"run": 2, "steps": [_FINDING_STEP]},
    ]
    kept, dropped = validate_vector(
        [
            _milestone(
                consensus=[1],
                evidence=[
                    # cited from run 1 as `stated`; it is run 2's observation
                    {"run": 1, "step": 1, "source": "stated", "span": "return row.start - 1"},
                ],
            ),
            _milestone(
                id="m2",
                category="explore",
                statement="the fact lives in a block the category does not admit",
                consensus=[2],
                # inside the agent's own fenced command: never admissible for explore
                evidence=[
                    {"run": 2, "step": 1, "source": "output", "span": "grep -n compute_offset"}
                ],
            ),
        ],
        runs,
        task="fix it",
    )

    assert [m["id"] for m in kept] == ["m1"]
    entry = kept[0]["evidence"][0]
    assert (entry["run"], entry["source"]) == (2, "output")
    assert kept[0]["consensus"] == [2]
    assert [d["reason"] for d in dropped] == ["span_inadmissible_for_explore", "unreached"]


def test_a_malformed_entry_in_the_payload_costs_only_itself():
    kept, dropped = validate_vector(
        [
            7,
            "text",
            _milestone(
                evidence=[{"run": 1, "step": 1, "source": "output", "span": "return row.start - 1"}]
            ),
        ],
        _RUNS,
        task="fix it",
    )
    assert [m["id"] for m in kept] == ["m1"] and dropped == []
