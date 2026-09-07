"""Turn each milestone into a SET of questions, written by one call over the whole vector.

One question per milestone gives a binary reading - reached it or did not - and cannot separate a
candidate that did most of the work from one that did none. A set at a range of difficulties can:
the number a candidate answers YES moves with how far it got rather than saturating.

That range is asked for in the prompt and recorded per question as `rung`. It is NOT enforced here:
no question is skipped because a shallower one failed, and no question is worth more than another.
The rung is carried for analysis and nothing in scoring reads it.

Two properties this module holds deliberately:

  * ONE FLAT TAG. A tag does two jobs - splitting the writer's work and telling the judge what
    counts as evidence - and only the first is worth dropping. The flat tag must be registered in
    TAG_WEIGHTS, or question_weight falls back to a default rather than the weight meant for it.
    Which milestone a question came from is carried separately, and carries no weight.
  * EQUAL QUESTIONS. A milestone decides what gets asked, not what an answer is worth: the
    question count per milestone is whatever the milestone supports, and every question a sample
    holds then counts the same towards that sample's score.
"""

from __future__ import annotations

import re
from typing import Any

from ...shared.json_extract import extract_json
from .questions import normalise_span

# What the writer is told to emit and what the schema pins it to. It is a placeholder, not the tag
# a question ships with: the real tag is the category of the milestone the question belongs to,
# known in code and does not depend on the writer getting it right. `_write_ladders` substitutes it.
LADDER_TAG = "reference:milestone"
RUNGS_MAX = 6
# Fewest questions a surviving milestone is allowed to carry before it is asked for again. A
# milestone with one or two is not a thin milestone - HOW MANY already says two is right only for
# something indivisible with no ground shown - it is nearly always a first pass that stopped early.
LADDER_MIN = 2
QUESTIONS_MAX = 60

# Two spans "agree" when they are the same code seen twice. Token overlap rather than equality,
# because one run may read a function through `sed -n` and another through `grep -n`, leaving
# fragments that differ at the edges. The threshold is deliberately loose: what it has to catch is
# the agents genuinely opening DIFFERENT code, which is what decides whether a question may name a
# site at all.
_AGREE_JACCARD = 0.6
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# The work leading to a milestone is bounded naturally by the previous milestone that run reached.
# The vector names one landing step per run per milestone and nothing in between, so most of what a
# run actually did is unaccounted for - which is why a question set with nothing to say in its
# middle drifts down into "did the candidate open the file".
#
# That natural bound is missing for the FIRST milestone in a run, where an unbounded window would
# swallow the whole trajectory and call every file in it shared. This cap exists for that case
# alone. Its value is not tuned.
APPROACH_STEPS_FIRST_MAX = 4

# How many of a run's own sentences from the approach window to show. A window can hold dozens, and
# passing all of them for every milestone would make the prose the bulk of the prompt. Capped per
# run rather than in total, so a run that worked quietly is not crowded out by one that narrated.
APPROACH_SENTENCES_MAX = 6

# Sentences a step ends on, kept per step rather than per window. A step opens with its plan and
# closes with what it found, so taking a window's first N selects plans and discards findings.
# Per step rather than from the window's tail, so a multi-step window keeps a finding from each of
# its steps instead of only the last.
APPROACH_SENTENCES_PER_STEP = 2

# What an agent writes when announcing its next move rather than reporting a result. The prompt
# tells the writer to discard exactly these, so passing them spends prompt budget on material that
# is unusable by construction.
_INTENT_RE = re.compile(
    r"^(let me|now let me|now i|i need to|i'll|i will|let's|first,? let me|next,? let me|"
    r"now,? let me|let me first)\b",
    re.I,
)
_PROSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_THOUGHT_RE = re.compile(r"^\s*THOUGHT:\s*", re.M)
_CMD_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.S)
_TARGET_RE = re.compile(
    r"[\w./-]+\.(?:py|rs|go|js|ts|tsx|c|h|cc|cpp|java|rb|toml|json|csv|rdb|md|txt|yaml|yml)\b"
    r"|\b[a-z_][a-z0-9_]{4,}\b(?=\()"
)
_TARGET_STOP = frozenset(
    "print range append format import return string struct output printf sprintf println".split()
)


def _tokens(span: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalise_span(span or "").lower()))


def spans_agree(spans: list[str]) -> bool:
    """Whether every span points at the same code, so a rung may name it.

    Fewer than two spans is not agreement, it is the absence of a comparison, and it answers NO.
    """
    sets = [_tokens(s) for s in spans if s and s.strip()]
    if len(sets) < 2:
        return False
    first = sets[0]
    return all(
        len(first & other) / max(1, len(first | other)) >= _AGREE_JACCARD for other in sets[1:]
    )


def project_vector(
    milestones: list[dict[str, Any]],
    trajectories: dict[int, list[str]] | None = None,
) -> str:
    """Render the whole validated vector for one question-writing call.

    What it keeps, and why each is needed. The statements a milestone rests on, because the
    increment between those and this milestone is where its middle questions come from, and naming
    them is what stops the writer restating an earlier milestone's conclusion. Whether the spans
    agree, because that decides whether a question may name a site. Every distinct span, because a
    question that names code has to name code the writer has actually been shown.

    Run numbers and consensus lists are withheld: the writer needs to know that the routes differed,
    never which agent took which.
    """
    by_id = {str(m.get("id")): m for m in milestones}
    landed: dict[int, int] = {}
    blocks: list[str] = []
    for index, milestone in enumerate(milestones, start=1):
        spans = [str(e.get("span", "")) for e in (milestone.get("evidence") or [])]
        distinct: list[str] = []
        for span in spans:
            if span.strip() and not any(spans_agree([span, kept]) for kept in distinct):
                distinct.append(span)
        agreed = spans_agree(spans)
        earlier = [
            by_id[d]["statement"]
            for d in (milestone.get("depends_on") or [])
            if d in by_id and by_id[d].get("statement")
        ]
        lines = [
            f"MILESTONE {index} (id {milestone.get('id')})",
            f"  CATEGORY: {milestone.get('category', '')}",
            f"  ACHIEVED: {milestone.get('statement', '')}",
        ]
        if milestone.get("necessity_reason"):
            lines.append(f"  WHY IT IS REQUIRED: {milestone['necessity_reason']}")
        for statement in earlier:
            lines.append(f"  ALREADY REACHED BEFORE THIS ONE: {statement}")
        lines.append(
            "  ROUTES: the agents reached this through the same code"
            if agreed
            else "  ROUTES: the agents reached this through DIFFERENT code — name none of it"
        )
        windows = approach_windows(milestone, trajectories, landed) if trajectories else {}
        approach = shared_approach(windows)
        if approach:
            lines.append(
                "  EVERY AGENT WORKED ON THIS ON THE WAY HERE: " + ", ".join(sorted(approach)[:8])
            )
        for _, sentences in sorted(approach_prose(windows).items()):
            lines.append("  ONE AGENT, IN ITS OWN WORDS, CROSSING THAT GROUND:")
            for sentence in sentences:
                lines.append(f"    - {sentence}")
        for span in distinct[:3]:
            lines.append(f"  WHAT THEY HAD IN FRONT OF THEM:\n{_indent(span)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def command_targets(step_text: str) -> set[str]:
    """The files and functions a step's own commands name.

    Read from the agent's blocks rather than from what came back, because what a run went looking
    for is its approach; what the environment returned is the environment's answer.
    """
    joined = " ".join(
        line for block in _CMD_RE.findall(step_text or "") for line in block.splitlines()
    )
    return {t for t in _TARGET_RE.findall(joined) if t.lower() not in _TARGET_STOP}


def approach_windows(
    milestone: dict[str, Any],
    trajectories: dict[int, list[str]],
    landed: dict[int, int],
) -> dict[int, list[str]]:
    """{run: the steps it took between the previous milestone and this one}.

    `landed` carries the previous milestone's landing step per run and is updated in place, so the
    windows partition the run rather than overlapping, and each milestone's approach is exactly the
    steps since the last thing that run is known to have reached. A window that inverts - two
    milestones landing on the same step, or a vector whose order does not follow step order in some
    run - yields nothing for that run rather than a backwards slice.
    """
    windows: dict[int, list[str]] = {}
    for entry in milestone.get("evidence") or []:
        run = int(entry.get("run") or 0)
        step = int(entry.get("step") or 0)
        steps = trajectories.get(run) or []
        if not steps or step < 1:
            continue
        # since the previous milestone this run reached, or - for its first - a bounded look-back
        low = landed[run] + 1 if run in landed else max(1, step - APPROACH_STEPS_FIRST_MAX)
        landed[run] = max(landed.get(run, 0), step)
        windows[run] = steps[low - 1 : min(step, len(steps))]
    return windows


def shared_approach(windows: dict[int, list[str]]) -> set[str]:
    """What every run worked on, on the way to reaching this milestone.

    Intersected across runs, which is what makes it safe to name: a target present in every run's
    approach cannot be one run's detour.
    """
    per_run = [
        set().union(*[command_targets(t) for t in window]) if window else set()
        for window in windows.values()
    ]
    return set.intersection(*per_run) if per_run else set()


def approach_prose(windows: dict[int, list[str]]) -> dict[int, list[str]]:
    """{run: what it said in its own words while crossing that window}.

    Addresses alone support no proposition: a filename can only become "did the candidate open X",
    which scores a motion. Sentences do carry propositions - "the search did not find the function
    in that file" is a finding - and an intermediate finding is what a middle question should ask
    about.

    Unlike the targets these CANNOT be intersected across runs, so every run's are shown separately
    and the prompt carries the guards: the shared intent is what the writer reads them for, the
    ROUTES line still governs whether code may be named, and a sentence that only announces an
    intention establishes nothing.
    """
    out: dict[int, list[str]] = {}
    for run, window in windows.items():
        sentences: list[str] = []
        for step in window:
            body = _THOUGHT_RE.sub("", _CMD_RE.sub(" ", step or "")).replace("\n", " ")
            found = [
                s.strip()
                for s in _PROSE_SPLIT_RE.split(body)
                if len(s.strip()) > 25 and not _INTENT_RE.match(s.strip())
            ]
            sentences.extend(found[-APPROACH_SENTENCES_PER_STEP:])
        if sentences:
            out[run] = sentences[-APPROACH_SENTENCES_MAX:]
    return out


def _indent(span: str) -> str:
    body = "\n".join(f"    {line}" for line in str(span).splitlines()[:12])
    return body or "    (nothing)"


LADDER_SKELETON = """You write the evaluation checklist that measures HOW FAR an agent got on one \
task. A judge answers your yes/no questions about a candidate TRAJECTORY: the original \
conversation, then CANDIDATE OUTPUT blocks with ENVIRONMENT OBSERVATION blocks between them. Only \
the CANDIDATE OUTPUT blocks are the candidate's own work, and only they can earn anything.

You are shown the milestones this task required, in the order they had to be reached. For each one \
you write a SET of questions at different depths: some only a candidate that fully reached it \
could earn, some a candidate that got partway could, and some about the ground it had to cross to \
get there. They are not a chain, and you are not required to make each one follow from the one \
before.

Each milestone carries a CATEGORY, and it decides what the questions are about:

claims - the TASK's own claim, put to the repository. Ask what was settled about the claim - \
that it held, that it did not, that it held only sometimes - never the claim restated. explore - \
a mechanism established in the code. Ask what became known. action - \
what the code now does that it did not before. Ask what is now true, and where. verification - a \
check run, named by the behaviour it exercises. Ask what it exercised, never what it returned.

===== THE SPREAD IS THE MEASUREMENT =====

Every question you write is answered independently, and the milestone's score is the fraction of \
its questions answered YES. So what measures how far a candidate got is not the ORDER of your \
questions but their SPREAD in difficulty.

A milestone whose questions are all equally hard measures almost nothing: nearly every candidate \
answers all of them or none, and a reader cannot tell the one that did most of the work from the \
one that did none. Give each milestone questions at a range of depths - at least one that a \
candidate which engaged with this part of the problem at all would earn, at least one that only a \
candidate which reached the whole of it would, and as many in between as the material supports.

Two checks before you emit a milestone's questions:

A QUESTION NOTHING FAILS MEASURES NOTHING. The easiest question is for a candidate that did real \
but incomplete work, not for any candidate at all. If a trajectory that never touched this part of \
the problem would still answer YES, raise it or drop it.

TWO QUESTIONS THAT ALWAYS MOVE TOGETHER ARE ONE QUESTION. If you cannot describe a candidate that \
would answer YES to one and NO to the other, you have written the same question twice at two \
wordings. Merge them and spend the slot elsewhere. This is the only thing that makes a question \
redundant - two questions about different things are never redundant merely because they sit at \
similar depths.

===== WHERE QUESTIONS COME FROM =====

Five places, best first.

WHAT THEY WORKED OUT ON THE WAY. A milestone may carry the agents' own sentences from the ground \
between the milestone before it and this one. Each thing they worked out crossing that ground is a \
candidate question, and these are usually the best middle questions you will get: they are the \
intermediate conclusions a candidate had to reach to get from there to here.

Read them for what became KNOWN, never for what was done. A sentence that only announces an \
intention - "let me read the whole file", "now I will check the tests" - establishes nothing \
and is not a question. A sentence that reports something learned is: "the search did not find the \
function in that file" means the candidate ruled a location out, and "the relevant file is the \
profiler rather \
than the decoder" means it settled where the work belongs. Ask about the thing settled.

Never quote these sentences, and never make one agent's WORDING the subject of a question. What \
became \
known through them is exactly what you should be asking about; the words it happens to be written \
in are not. They are one agent's words about shared ground, so where several agents' sentences are \
shown, what they have in common is the signal. The ROUTES line still decides whether a question \
may \
name code.

WHAT EVERY AGENT WORKED ON GETTING HERE. Where a milestone carries a line naming that, those are \
files and functions every one of them touched on the way to it - never one agent's detour, \
because only what they all worked on is listed. Every route to this milestone went through them, \
which is what makes them safe to name.

Two kinds of question come out of that list, and you want both.

The deeper one is the understanding a candidate must have picked up there, stated as what it now \
knows.

The shallower one is whether the candidate got to that ground at all. "Did the candidate work \
with the stream size calculation in memprofiler/stream.go", or "did the candidate establish that \
the defect is in the profiler rather than the decoder". Where the list names several files, one \
question may name the set: whether the candidate worked with any of them. This is the shallowest \
question a milestone can carry and it is a real one - a candidate that never went near this part \
of the problem answers NO, so it clears the floor; a candidate that oriented correctly and \
understood nothing further answers YES and nothing above it. Without such a question a milestone \
cannot tell those two apart, and scores them both zero.

WHAT HAD TO BE TRUE FIRST. A milestone rests on things that had to be known or done before it \
could be reached. Where those are themselves milestones you were shown, they are listed under \
ALREADY REACHED BEFORE THIS ONE, and they have their own questions. So do not write a question \
whose answer is one of those statements restated - that is the whole of the restriction, and it \
covers the conclusion only. The ground between that milestone and this one is NOT out of bounds: \
it is yours, it is where this milestone's middle questions come from, and something worked out \
while crossing it belongs here even when it touches the same file, function or mechanism the \
earlier statement named. What you want is the increment: the work between what was already \
reached and this milestone, which is usually where the interesting questions are.

THE PARTS OF THE MILESTONE ITSELF. Read the ACHIEVED line for the separate things it asserts. A \
milestone that says a calculation omits one field AND miscounts another carries two questions, \
one per \
omission, because a candidate can find one without the other.

THE WHOLE OF IT. One question is the milestone achieved completely, stated so that only a \
candidate \
that reached all of it answers YES.

===== NEVER SCORE THE MECHANICS =====

A question names what became known, what became true, what was put to the test, or WHERE the \
candidate worked. Never the command, utility, editor or flag that produced it.

"Did the candidate determine which code computes the size" is a good question; so is "did the \
candidate determine that the size omits a field". "Did the candidate grep for the function name" \
is not a question at all: the same work is done with any command, and a checklist that scores the \
command scores the habit of one agent rather than the capability of any.

Naming WHERE is not naming HOW. "Did the candidate work with the size calculation in \
memprofiler/stream.go" is a fair question: every route to that milestone goes through that file, \
so any candidate that did the work answers YES however it got there. What is banned is the \
instrument, never the location.

Treat this vocabulary as unusable inside a question: grep, rg, find, sed, awk, cat, head, tail, \
ls, nl, vim, str_replace, apply_patch, and every option string.

===== WHERE THE AGENTS DIFFERED, NAME NOTHING =====

Each milestone tells you whether the agents reached it through the same code or through different \
code.

SAME CODE: a question may name the file, function, value or expression, and naming it is BETTER, \
because a question tied to something the candidate had to obtain cannot be passed by guessing.

DIFFERENT CODE: they reached the same thing by different routes, and a question naming either \
route \
punishes every candidate that took the other. State what became known and name no file, function \
or \
expression at all. Do not resolve it by picking the route you understood best — that is the single \
most common way these questions go wrong.

===== FAKING MUST COST AS MUCH AS DOING =====

For every question, work out what an agent that only wanted to look competent would do. If it \
could \
earn the YES by restating the task, by emitting a command merely shaped like the right one, or by \
writing confident prose about work it never did, the question rewards imitation over capability. \
Rewrite it or drop it.

The strongest shape ties a conclusion to the observable it implies: "did the candidate establish \
that X, by showing Y". An agent that never looked cannot produce Y, and any route that produces Y \
earns the YES — which is exactly the combination you want.

===== THE ENVIRONMENT IS NOT THE CANDIDATE =====

Three things in the trajectory were not written by the candidate: the task, the conversation \
before \
its first block, and every ENVIRONMENT OBSERVATION. None of them is creditable.

So no question may turn on whether a test passed, whether a command succeeded, what a return code \
was, \
or whether an observation contains a given string. Apply this test: suppose the observation had \
come back empty or wrong while the candidate did exactly the same things. If your question's \
answer \
would change, its subject is the world and not the agent — rewrite it so the subject is something \
the candidate did, established or changed.

Observations matter in exactly one way: as proof the candidate obtained something. That an \
observation displayed a value is what makes the candidate's later use of that value grounded \
rather \
than guessed. Keep that direction.

This applies to the ground crossed on the way, and it is where that material is most often thrown \
away by mistake. Most of what an agent works out en route is worked out FROM an observation, and \
asking about the observation is the wrong end of it: "did the search come back empty" is the \
world, \
but "did the candidate settle that the calculation is not in the file the report named" is the \
candidate. Every en-route finding has that second form. Put it in that form rather than dropping \
it.

===== FORM =====

One short interrogative sentence per question. One clause, one condition, the subject inside the \
first \
three words. No "and" joining two things a candidate could do separately — that is two questions. \
No \
justifying clause, no opening conditional, and never address a turn by its number.

Verbs of intention earn nothing and must not appear: attempting, mentioning, planning, \
considering, \
recognising. A candidate either established it, changed it, or exercised it.

Much of what is settled on the way is settled NEGATIVELY - a location ruled out, a suspect \
cleared, \
a cause placed somewhere other than where the report pointed. That is a real thing established and \
you must not skip it for want of a phrasing. Write it as what the candidate settled, in the form \
"X \
rather than Y": "did the candidate establish that the size calculation lives in the profiler \
rather \
than in the decoder". Never as the absence itself.

Never disclose that milestones, a vector, or any other agent's work exists. The judge sees one \
candidate and your question, nothing else.

===== HOW MANY =====

As many questions as the milestone supports, up to {rungs}. Work through every source above \
before \
you decide you are finished: the ground crossed to get here, the milestones it rests on, each \
separate thing the statement asserts, and the whole. A milestone that asserts three things, or \
that carries several sentences of ground crossed to reach it, has five or six questions in it, and \
a set of two means you stopped early rather than that the milestone was thin.

Two is the right answer only for a milestone that asserts one indivisible thing AND has nothing \
recorded before it AND no ground shown. Padding is writing the same question twice at two \
wordings, or splitting one condition across two - never asking about a real thing that was settled \
on the way.

===== OUTPUT =====

Strict JSON, no prose, no code fences. Every question for every milestone, milestones in the order \
shown, and within each milestone ordered shallowest first:

{{"questions":[{{"milestone":"m1","rung":1,"tag":"{tag}","text":"...",\
"unearned":"..."}}]}}

`milestone` is the id shown in brackets on the MILESTONE line. `rung` is how deep the question \
sits, 1 being the shallowest - it records the depth you chose, and does NOT mean the question \
below \
it must be answered first. `unearned` is one concrete near-miss that must NOT earn a yes on THIS \
question: the sharpest one is usually a candidate that got close to this and stopped just short."""


def build_ladder_messages(*, vector: str) -> list[dict[str, str]]:
    """One call for the whole vector: the milestones, then what to do with them."""
    user = (
        "THE MILESTONES THIS TASK REQUIRED, in the order they had to be reached. Each is followed "
        "by why it is required, what had already been reached before it, whether the agents got "
        "there through the same code, and the content they had in front of them when they did:\n"
        f"------\n{vector.rstrip()}\n------\n\n"
        "Write the questions for every milestone above."
    )
    return [
        {"role": "system", "content": LADDER_SKELETON.format(rungs=RUNGS_MAX, tag=LADDER_TAG)},
        {"role": "user", "content": user},
    ]


def ladder_schema() -> dict[str, Any]:
    """Field names are chosen so ALPHABETICAL order is thinking order.

    This evaluator emits object properties alphabetically, so a key's name decides WHEN the model
    writes it. Renaming a field can therefore change what the model produces, up to and including an
    empty list. The order here is milestone < rung < tag < text < unearned, which has the model say
    which milestone and which depth it is writing, then the question, and only then the near-miss
    that must not earn it.

    `unearned` rather than `example_bad` for exactly that reason: `example_bad` sorts before
    `milestone` and would have the model write the counterexample before the question it is a
    counterexample to.
    """
    question = {
        "type": "object",
        "properties": {
            "milestone": {"type": "string"},
            "rung": {"type": "integer"},
            "tag": {"type": "string", "enum": [LADDER_TAG]},
            "text": {"type": "string"},
            "unearned": {"type": "string"},
        },
        "required": ["milestone", "rung", "tag", "text", "unearned"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 0,
                "maxItems": QUESTIONS_MAX,
                "items": question,
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def parse_ladder(raw: str, known_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Questions from a model reply, renumbered so rungs are contiguous per milestone.

    A rung index is only meaningful as a position in its own ladder, so gaps and duplicates in what
    the model emitted are closed here rather than trusted. Order within a milestone is preserved:
    the model was asked to emit ascending, and its ordering carries more information than its
    arithmetic.
    """
    payload = extract_json(raw or "", prefer_keys=("questions",))
    if isinstance(payload, list):
        payload = {"questions": payload}
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload.get("questions") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        milestone = str(item.get("milestone") or "").strip()
        if not text or not milestone:
            continue
        if known_ids is not None and milestone not in known_ids:
            continue
        rows.append(
            {
                "milestone": milestone,
                "rung": item.get("rung") if isinstance(item.get("rung"), int) else 0,
                "tag": LADDER_TAG,
                "text": text,
                "unearned": str(item.get("unearned") or "").strip(),
            }
        )
    out: list[dict[str, Any]] = []
    for milestone in dict.fromkeys(r["milestone"] for r in rows):
        ladder = [r for r in rows if r["milestone"] == milestone][:RUNGS_MAX]
        for height, row in enumerate(sorted(ladder, key=lambda r: r["rung"]), start=1):
            out.append({**row, "rung": height})
    return out[:QUESTIONS_MAX]
