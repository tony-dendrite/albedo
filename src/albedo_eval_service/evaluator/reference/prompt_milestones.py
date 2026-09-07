"""Reduce several reference runs to a "vector of change", then validate it.

    N reference runs  ->  1 extractor (vector of change)  ->  validate  ->  question ladder

The extractor reads all runs at once and reduces them to one ordered list of task-level
milestones, each justified for necessity by counterfactual (never by majority vote) and each
backed by a verbatim span per run, tagged with where that span came from.

`validate_vector` then enforces mechanically what the extractor prompt asks for: a milestone
survives only if it is necessary, not already given by the task, and evidenced by a span that
really occurs in the kind of block its `source` claims. The survivors go to `prompt_ladder`,
which turns each into several questions - so a milestone's value here is not only that it is true
but that it has enough internal structure to be split.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..shared.questions import _FENCE_RE
from .questions import normalise_span

# ---------------------------------------------------------------------------------- extractor

EXTRACTOR_SKELETON = """You read several independent accounts of the SAME task, written \
by strong agents working from the SAME starting point. Your job is to reduce them to ONE "vector \
of change": the ordered, task-level milestones a competent agent must reach to solve this task — \
and nothing that belongs to any single agent's route.

===== WHAT THE ACCOUNTS ARE, AND ARE NOT =====
Each account alternates the agent's own blocks with the ENVIRONMENT OBSERVATIONS that came back. \
The observations are RECONSTRUCTED, not recorded: the same command in two runs can return \
different commit messages, hashes, line numbers, test names or wording, and no run is \
authoritative. Read an observation for the FACT it carries, never for its surface detail.

That has one hard consequence. A milestone's statement may never depend on a detail only one \
run's observation shows. If one run's output names one set of tests and another names a different \
set, the milestone is "the tests covering <behaviour>", never a list of names. Where the runs \
agree on a fact, the fact is signal; where they disagree on a detail, the detail is noise. Never \
settle a disagreement by picking one run, and never build a union of two runs' details that no \
single run displayed.

This is not majority voting either. You are not asking "did two of three do it". You are asking \
"does this task, as its own logic dictates, require this milestone to progress". The accounts are \
evidence you read to learn what the milestones are, not the definition of them.

===== FAKING IT MUST COST AS MUCH AS DOING IT =====
This outranks everything below. Each milestone is then split into three to six independent \
questions and scored on the FRACTION of them answered yes, so ask what an agent that only wanted \
to look competent would do. If it could reach the milestone by restating the task, by emitting a \
command merely shaped like the right one, or by writing confident prose about work it never did, \
the milestone rewards imitation over capability and must not be emitted.

That split is also a floor on substance. A milestone must carry enough distinct ground that three \
separate questions can each land on a different part of it — the fact and its mechanism, the site \
and what it now does, the behaviour and what exercising it required. One that holds a single \
indivisible assertion cannot be split, so its fraction stops measuring how far the candidate got \
and starts measuring nothing. Prefer a milestone with internal structure over two that each have \
none; this is the counterweight to ONE FACT, ONE MILESTONE, not a licence to merge unrelated facts.

The strongest milestones tie an act to something the agent itself had to obtain: the only way to \
name the value is to have read it, and having read it is the capability being measured. Prefer \
these everywhere they exist.

===== NECESSITY BY COUNTERFACTUAL =====
For every milestone, run this test: remove the milestone entirely. Can a competent agent still \
complete the task? If the task cannot progress past some point without it — a defect that must \
be located, a line that must be changed, a fact that must be established before the change can \
be correct — the milestone is NECESSARY (necessary=true). If a competent agent could take a \
different route and still finish, it is OPTIONAL (necessary=false) and belongs to one agent's \
route, not the task.

necessary=false carries exactly one meaning: the counterfactual test failed. An optional \
milestone is recorded and then dropped, producing no question, so it is never worth emitting one \
to pad the vector — and never worth reaching for the flag to report anything else, least of all \
that a milestone was hard to evidence.

Write necessity_reason as the counterfactual in one sentence: what breaks, or cannot be reached, \
without the milestone.

Consensus is evidence, not reason. All runs doing something is a strong hint it is necessary, \
but a milestone done by only one run can still be necessary if the others failed, wandered, or \
stalled. A milestone done by all runs is still OPTIONAL if the task does not actually require it \
(all three wasted the same step). Judge by task logic; use consensus only as a check on your own \
judgment.

===== THE GIVEN-MATERIAL FLOOR =====
The TASK block is not just a problem statement. It is a system prompt, a problem description, and \
a conversation that has already run — including its commands, its ENVIRONMENT OBSERVATIONS, and \
every file content those observations displayed. All of it was generated in advance and belongs \
to nobody. The candidate starts after it and reads all of it for free.

So for every milestone, ask whether the TASK block already carries its content. If the problem \
description names the symptom, the file, the function and the shape of the defect, or if a \
command in the prefix already displayed the code, then reaching that milestone costs nothing. \
Mark in_prefix=true. Do this even when the counterfactual test would call the milestone \
necessary — "the agent cannot fix what it has not located" is true and irrelevant when the \
location was handed over.

A milestone with in_prefix=true is recorded and then dropped; it produces no question. Emit \
in_prefix=false only for what the agent had to discover for itself: the concrete mechanism, the \
exact lines, a value read from the repository, a fact the prefix does not carry. If a milestone \
mixes given content with discovered content, split it and keep the discovered part.

in_prefix=true must be QUOTED, exactly as evidence is. Set prefix_quote to the span of TASK text \
that carries the milestone's content, copied character-for-character from the TASK block. It is \
checked against the TASK, and a claim whose quote is not found there is discarded and the \
milestone kept — so a milestone you cannot point at in the TASK is not given, however strongly it \
feels implied. Set in_prefix=false with prefix_quote="" for everything else.

ACTION and VERIFICATION milestones are immune to this floor. The task may predict the outcome \
— "the tests should pass", "the ordering should be correct" — but it does not perform the \
work. Editing the code and running the verification are irreducible acts even when the task \
says what the result must be. Set in_prefix=false on them unless the prefix literally already \
made the edit or ran the verification.

===== ONE FACT, ONE MILESTONE =====
An explore milestone's identity is the FACT it establishes — never the source, route, or command \
that revealed it. Two runs reaching the same fact by different means (one reads git history, \
another reads a sibling file) are ONE explore milestone: one statement of the fact, consensus = \
the union of every run that established it, and one evidence entry per run from whatever that \
run showed.

Splitting one fact into "the git-history route" and "the sibling-file route" is the tool-shaped \
error in disguise: it demotes a single necessary fact into two optional routes and leaves the \
fact itself unscored. Before you emit two milestones, check whether they name the same fact. If \
they do, merge them and keep the fact, not the routes.

===== ONLY THE AGENT'S WORK COUNTS =====
Describe every milestone by what it establishes, changes, or exercises — never by the command, \
utility, flag, or editor used. "the defect is the column-offset computation in float_values.py" \
is a milestone; "runs grep -n" is not. Treat this vocabulary as unusable inside a statement: \
grep, rg, find, sed, awk, cat, head, tail, ls, vim, str_replace, apply_patch, and every option \
string. The same work is done with any of them.

A milestone whose only distinguishing content is an environment result — a test passing, a \
command succeeding, an exit code — is unusable. The environment is reconstructed and may report \
success for failing work. Apply this test: suppose the observation had come back empty or wrong \
while the agent did exactly the same things. If the milestone would no longer hold, it is scoring \
the environment. Ask what was aimed at, never what came back.

===== THE CAUSAL CHAIN =====
Emit milestones in causal order, each of exactly one category:

- claims — the TASK's own assertion, tested. The subject of the statement is a claim the TASK \
made, and the statement says what the accounts found when they put that claim to the repository: \
that it holds, that it does not, or that it holds inconsistently.
- explore — a mechanism the agent must establish in the code: the defect site, how the defect \
arises, or an intermediate fact reached on the way. The subject is the code, not the task.
  The boundary between these two is the SUBJECT, and the task decides it, not the work that \
reached the fact. If the task asserts it — the defect reproduces, the panic reads this, the \
output is that — the milestone is `claims`, however much building and running an account did to \
confirm it. `explore` begins where the task stops asserting: a mechanism the task does not \
state.
- action — a semantic change the agent must make: what the code now does that it did not before, \
and where it lands.
- verification — a check the agent must run, named by the BEHAVIOUR it exercises. Never by its \
result.

One relation rides on top of the milestones; never make a separate milestone of it:

- DEPENDENCY — depends_on lists the ids of earlier milestones this one rests on. An action that \
uses a value the agent had to read first depends on the reading milestone. Record the dependency \
on the acting milestone, and only where the earlier milestone genuinely supplies something the \
later one consumes. It is read as the ground this milestone starts from, so what the questions \
ask about is the increment between the two — which makes an accurate depends_on worth more than a \
generous one.

===== WHEN THE TASK'S CLAIM DOES NOT HOLD =====
The TASK block asserts a defect. That assertion is not evidence — it is a claim the agent is \
expected to test. Where the accounts show the stated reproduction being run and returning the \
result the task says it should NOT return, the claim did not hold. That is itself a necessary \
milestone of category `claims`: emit it, with the reproduction's own output as its span, because \
establishing that the reported defect does not occur is exactly the work the task demanded. A \
claims milestone is equally required where the claim DID hold and an account showed it holding \
— confirming the defect against the repository is work, and it is the evidence every later \
action rests on.

A claim that did not hold constrains everything after it. Do not emit an action milestone that \
presupposes the unreproduced defect, and do not recover the action from the task statement \
alone: with the defect unobserved, no required action is established, and such a milestone would \
score the task statement rather than the agent. Where the runs disagree — some observing the \
failure, others not — the milestone is that the behaviour is inconsistent, which is again a \
claims milestone and never an action.

===== EVIDENCE =====
For each milestone, give one evidence entry per run in consensus — no more and no fewer. A run \
absent from consensus has no entry. Each entry is {run, step, source, span}:

- run — the run number the span comes from.
- step — the REFERENCE STEP number inside that run where it appears.
- source — where in that step the span sits, exactly one of:
    output  — inside the ENVIRONMENT OBSERVATION.
    stated  — in the agent's own prose, outside any code block.
    edit    — inside a code block the agent wrote, showing the new content it installs.
    command — inside a code block the agent wrote, being the invocation itself.
- span — copied character-for-character from that place. No ellipsis, no paraphrase, no \
reflowing, no corrected spacing. Long enough to be unmistakably from that run, no longer.
  Start and end it on a boundary you can see: never inside a word, a quoted string or an \
escape sequence. A span that stops one character short of a closing \\n is not the text it came \
from, so it is not found there, and the milestone loses that run over a copying slip rather than \
a missing fact.

Which sources are admissible depends on the category, and this is checked:

- explore — output or stated only. A fact the agent typed into its own patch script is NOT \
established: it was assumed, not discovered. If the only place an explore milestone's content \
appears is a code block the agent authored, it is UNREACHED.
- action — edit, command or output. The block installing the new lines, the invocation that \
installs them where the edit is applied by a command such as `sed -i`, a heredoc or a patch \
tool, or an observation displaying them afterwards. Never the pre-change state being targeted, \
and never an invocation that merely runs something: the span must carry the new content.
- verification — command or stated. The aimed invocation, or the agent's own sentence naming what \
it is about to exercise. Never the verification's output: what came back is the environment's \
answer, not the agent's act.

The verification rule is the one most often broken, so here it is concretely. Suppose a run's \
step reads:

  REFERENCE STEP 13:
  Let me run the scope-variable tests to confirm S, A and Let still behave.
  ```bash
  python -m pytest tests/test_scope_vars.py -q
  ```
  ENVIRONMENT OBSERVATION:
  PASS test_globals
  PASS test_let
  Passed: 6/6

  REJECTED  {"source":"output","span":"PASS test_globals PASS test_let"}
            what came back — the environment's answer, not the agent's act. The verification
            is recorded UNREACHED.
  CORRECT   {"source":"command","span":"python -m pytest tests/test_scope_vars.py -q"}
  ALSO OK   {"source":"stated","span":"run the scope-variable tests to confirm S, A and Let
            still behave"}

The same holds for every verification: a reproduction script, a linter, a single test. Cite the \
invocation or the sentence aiming it. If neither is present in a run, that run has no admissible \
span for the verification.

NEVER invent a span, and never copy one run's span into another run's entry. If a run in \
consensus has no admissible span, remove that run from consensus and drop its entry. If that \
empties consensus, the milestone is UNREACHED: emit it with consensus=[] and evidence=[], and \
leave necessary at whatever the counterfactual test decided. Never flip necessary to false to \
signal that you could not ground a milestone — an empty consensus already says that, and saying \
it twice discards a milestone that a second look might have grounded. \
The accounts are truncated near their step cap, so a step the task's logic would eventually \
demand may never materialise; an unreached milestone is not a defect in the accounts, it is \
simply outside what can be scored. Never reason "this would be necessary at some later step" and \
then invent it.

===== OUTPUT =====
Output ONLY strict JSON, no prose, no code fences:
{"milestones":[{"id":"m1","category":"claims|explore|action|verification","statement":"...",\
"necessary":true,\
"necessity_reason":"...","in_prefix":false,"prefix_quote":"","consensus":[1,2,3],\
"depends_on":[],\
"evidence":[{"run":1,"step":3,"source":"output","span":"..."},\
{"run":2,"step":2,"source":"output","span":"..."},\
{"run":3,"step":5,"source":"stated","span":"..."}]}]}"""

EVIDENCE_SOURCES = ("output", "stated", "edit", "command")

# Which evidence sources can establish which category of milestone. An `explore` may never rest
# on a block the agent authored: content it typed into its own patch script was assumed, not
# found. A `verification` may never rest on output, because what came back is the environment's
# answer.
#
# `action` admits `command` because an edit is often applied BY a command — `sed -i`, a heredoc,
# a patch tool — so the invocation and the installed content are the same text. `edit` and
# `command` also share one haystack, so the label distinguishes nothing a validator can check;
# excluding `command` only discarded sound evidence over wording.
SOURCES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    # a claims milestone rests on what came back when the task's claim was put to the repository,
    # or on the account's own reading of that result across runs, which is the only place an
    # inconsistency between runs can be stated at all
    # `command` too: the extractor cites the reproduction's invocation for what the claim came to
    # as often as its output, and both are the agent putting the claim to the repository
    "claims": ("output", "stated", "command"),
    "explore": ("output", "stated"),
    "action": ("edit", "output", "command"),
    "verification": ("command", "stated"),
}


def milestone_tag(category: str) -> str:
    """The tag every question written for a milestone of this category carries.

    The category vocabulary lives in SOURCES_BY_CATEGORY and nowhere else, so a category added
    there is a tag here without further work - but it must also be given a weight in TAG_WEIGHTS,
    or question_weight falls back to the conservative default for an unrecognised reference tag.
    """
    return f"reference:{category}"


MILESTONE_TAGS: tuple[str, ...] = tuple(milestone_tag(category) for category in SOURCES_BY_CATEGORY)


def extractor_schema() -> dict[str, Any]:
    """Force the extractor's output shape: one milestone list, no prose."""
    evidence = {
        "type": "object",
        "properties": {
            "run": {"type": "integer"},
            "step": {"type": "integer"},
            "source": {"type": "string", "enum": list(EVIDENCE_SOURCES)},
            "span": {"type": "string"},
        },
        "required": ["run", "step", "source", "span"],
        "additionalProperties": False,
    }
    milestone = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "category": {"type": "string", "enum": list(SOURCES_BY_CATEGORY)},
            "statement": {"type": "string"},
            "necessary": {"type": "boolean"},
            "necessity_reason": {"type": "string"},
            "in_prefix": {"type": "boolean"},
            "prefix_quote": {"type": "string"},
            "consensus": {"type": "array", "items": {"type": "integer"}},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": evidence},
        },
        "required": [
            "id",
            "category",
            "statement",
            "necessary",
            "necessity_reason",
            "in_prefix",
            "prefix_quote",
            "consensus",
            "depends_on",
            "evidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"milestones": {"type": "array", "items": milestone}},
        "required": ["milestones"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------------- validation


def _haystacks(steps: list[dict[str, str | None]]) -> dict[str, str]:
    """One normalised search space per evidence source, over every step of one run.

    `step` is recorded for audit but not enforced: a model that miscounts steps should not cost a
    sound milestone its evidence. What is enforced is the KIND of block the span sits in, which is
    the distinction the admissibility rules rest on.

    The three spaces partition one run: the environment's replies, the agent's prose with its
    fenced blocks removed, and the content of those blocks. `edit` and `command` are both defined
    as "inside a code block the agent wrote", so they share the fenced space and no check can tell
    them apart - but neither may be satisfied by prose, which is what saying a thing looks like
    rather than doing it.
    """
    assistant = " ".join((step.get("assistant") or "") for step in steps)
    observation = " ".join((step.get("observation") or "") for step in steps)
    fenced = " ".join(match.group(1) for match in _FENCE_RE.finditer(assistant))
    return {
        "output": normalise_span(observation),
        "stated": normalise_span(_FENCE_RE.sub(" ", assistant)),
        "edit": normalise_span(fenced),
        "command": normalise_span(fenced),
    }


_SPAN_FROM_TASK_CHECKED = ("explore",)
_NEVER_OPTIONAL = ("verification",)


def _locate(
    span: str, run: int, source: str, allowed: tuple[str, ...], hay: dict[int, dict[str, str]]
) -> tuple[int, str] | None:
    kinds = [source] + [s for s in allowed if s != source] if source in allowed else list(allowed)
    for other in [run] + [r for r in sorted(hay) if r != run]:
        for kind in kinds:
            if span in hay.get(other, {}).get(kind, ""):
                return other, kind
    return None


def validate_vector(
    milestones: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    task: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Enforce mechanically what the extractor prompt asks for, and report every drop.

    Two things the prompt cannot enforce on its own are checked here: that a milestone is
    licensed at all (necessary, not given by the task prefix), and that every evidence span really
    occurs in a block kind its category admits, in some run - relocated if the extractor cited the
    wrong run or block label, so consensus is the runs where the evidence holds. A milestone whose
    evidence all fails is UNREACHED and dropped. Dangling depends_on ids are pruned so a surviving
    milestone cannot ground itself on a dropped one.
    """
    dropped: list[dict[str, str]] = []
    hay = {
        int(trajectory.get("run") or index): _haystacks(trajectory.get("steps") or [])
        for index, trajectory in enumerate(trajectories, 1)
    }
    task_hay = normalise_span(task)

    def drop(milestone: dict[str, Any], reason: str, detail: str = "") -> None:
        dropped.append(
            {
                "stage": "validate_vector",
                "reason": reason,
                "id": str(milestone.get("id", "")),
                "text": str(milestone.get("statement", ""))[:200],
                "detail": detail,
            }
        )

    kept: list[dict[str, Any]] = []
    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue  # a bare value where an object was promised: nothing to validate
        category = str(milestone.get("category"))
        if not milestone.get("necessary") and category not in _NEVER_OPTIONAL:
            drop(milestone, "optional")
            continue
        if milestone.get("in_prefix"):
            # `in_prefix` decides whether a milestone produces any question at all, and unlike
            # every evidence span it used to be taken on trust. A false claim is silent and total:
            # the milestone is dropped, no question is written, and `given_by_task` is not
            # recoverable, so no retry follows. So the claim is now held to the same standard as
            # evidence - quote it or lose it - and an unsupported claim costs the claim, never the
            # milestone.
            quote = normalise_span(str(milestone.get("prefix_quote") or ""))
            if quote and quote in task_hay:
                drop(milestone, "given_by_task", f"{quote[:120]!r}")
                continue
            logger.info(
                "milestone_prefix_claim_unsupported id={} quote={} statement={}",
                milestone.get("id"),
                f"{quote[:80]!r}" if quote else "<empty>",
                str(milestone.get("statement", ""))[:120],
            )
            milestone = {**milestone, "in_prefix": False, "prefix_claim_unsupported": True}
        allowed = SOURCES_BY_CATEGORY.get(category, ())
        if not allowed:
            drop(milestone, "unknown_category", category)
            continue
        evidence: list[dict[str, Any]] = []
        runs: set[int] = set()
        for entry in milestone.get("evidence") or []:
            span = normalise_span(str(entry.get("span", "")))
            source = str(entry.get("source", ""))
            run = int(entry.get("run") or 0)
            where = f"run {run} step {entry.get('step')}"
            if not span:
                drop(milestone, "empty_span", where)
                continue
            if category in _SPAN_FROM_TASK_CHECKED and span in task_hay:
                drop(milestone, "span_from_task", f"{where}: {span[:120]!r}")
                continue
            placed = _locate(span, run, source, allowed, hay)
            if placed is None:
                reason = (
                    f"span_inadmissible_for_{category}"
                    if _locate(span, run, source, EVIDENCE_SOURCES, hay)
                    else "span_not_in_corpus"
                )
                drop(milestone, reason, f"{where}: {span[:120]!r}")
                continue
            found_run, found_source = placed
            if found_run in runs:
                continue
            runs.add(found_run)
            evidence.append({**entry, "run": found_run, "source": found_source})
        if not evidence:
            drop(milestone, "unreached")
            continue
        kept.append({**milestone, "evidence": evidence, "consensus": sorted(runs)})

    surviving = {str(milestone["id"]) for milestone in kept}
    return [
        {**milestone, "depends_on": [d for d in milestone["depends_on"] if d in surviving]}
        for milestone in kept
    ], dropped


# ------------------------------------------------------------------------------------ call


def build_extractor_messages(
    *,
    task: str,
    references: list[str],
    candidate_turns: int,
) -> list[dict[str, str]]:
    """One extractor call over all references. The task and every reference are laid out before \
    any per-run commentary so the whole corpus is a single shared prefix."""
    run_blocks = "\n\n".join(
        f"===== RUN {index} =====\n{reference.rstrip()}"
        for index, reference in enumerate(references, 1)
    )
    user = (
        "TASK — the system prompt the agent operates under, the problem description, and the "
        "conversation that has already run, observations included. Every milestone must be "
        "checked against it: content it already carries is given, not discovered:\n"
        f"------\n{task.rstrip()}\n------\n\n"
        f"{len(references)} STRONG-AGENT ACCOUNTS — independent continuations from the same point "
        "under the same turn limit. They evidence what is achievable; no single route is the "
        "standard:\n"
        f"------\n{run_blocks}\n------\n\n"
        f"THE CANDIDATE CONTINUES FROM THE SAME POINT WITH {candidate_turns} TURNS, as these runs "
        "did. Whatever they reached is therefore reachable, and whatever none of them reached is "
        "not yours to invent."
    )
    return [
        {"role": "system", "content": EXTRACTOR_SKELETON},
        {"role": "user", "content": user},
    ]
