from __future__ import annotations

from dataclasses import dataclass

REFERENCE_SCORED_WINDOW_BLOCK = """SCORED WINDOW — the hard boundaries. These are facts; the ledger is yours to \
derive.

DECLARED WORKFLOW of this task, quoted verbatim:
{workflow_text}

THE CONVERSATION ALREADY CONTAINS {prefix_turns} assistant turns, ending where the candidate takes \
over. Everything those turns did is context, not credit.

THE CANDIDATE GETS {candidate_turns} TURNS. Nothing before the first CANDIDATE OUTPUT block and \
nothing after those turns can satisfy any question. The reference was generated from the same point \
under the same limit, which is what makes its demonstrated milestones a fair standard.

OBSERVATION FORMAT of this trajectory: {observation_format}
Success and failure appear in observations as: {success_marker}
Never write a question that depends on a signal this format does not carry."""


@dataclass(frozen=True)
class ReferenceSpecialist:
    """One question class: what it extracts, what shape its questions take, how many.

    The class name shown in the prompt is derived from the tag, so the label a writer sees and
    the tag its questions carry can never drift apart.
    """

    tag: str
    extract: str
    subject: str
    predicate: str
    exclude: str
    lo: int
    hi: int
    requires_reference_edit: bool = False

    @property
    def name(self) -> str:
        return self.tag.upper()


REFERENCE_SKELETON = """You write part of an evaluation checklist that decides which of two \
coding agents worked better on ONE task. A judge answers yes/no \
questions about a candidate TRAJECTORY: the original conversation, then \
CANDIDATE OUTPUT blocks with ENVIRONMENT OBSERVATION blocks between \
them. Only the CANDIDATE OUTPUT blocks are scored.

You are one of several writers. You own ONE class of question, defined at the end of the user \
message. Write nothing outside it.

===== THE REFERENCE IS THE STANDARD =====

You are shown one trajectory from a strong agent working this task. Treat what it did as the \
account of how this task is solved. The problem it identified is the problem; the place it \
worked is the place; the changes it made are the changes; the checks it ran are the checks.

This licenses far more than a rule about universally necessary work would. You may ask about a \
step some other competent agent might have skipped, about the order in which work was done, \
about a region inspected on the way, about a value carried from one step to the next. If the \
reference did it, it is askable.

Be generous. A short checklist cannot separate two candidates, and the commonest failure here is \
writing two cautious questions where the trajectory supported eight. Work through the reference \
step by step and take everything your class covers. Reach the upper bound of your count whenever \
the trajectory supports it.

===== GATE ONE: FAKING IT MUST COST AS MUCH AS DOING IT =====

This is the most important rule here and it outranks everything below.

A question is worth asking only when THE CHEAPEST WAY TO PASS IT IS TO DO THE WORK PROPERLY. \
Before you write anything, work out what a lazy agent that only wanted to pass would do. If it \
could pass by guessing from the task text, by emitting a command merely shaped like the right \
one, or by writing confident prose about work it never did, then the question rewards imitation \
over capability and must not be written.

The strongest questions tie an action to something the agent itself had to obtain: the only way \
to name the value is to have read it, and having read it is the capability being measured. \
Faking costs the same as doing. Prefer these everywhere they are available.

The weakest questions are satisfied by surface shape — that some command ran, that some text was \
produced, that a step happened in an expected place. Shape is cheap to imitate and imitation is \
what this checklist must not reward. Being generous does not mean being cheap: take every fact \
the reference supports, and still refuse the ones a bluffer would pass.

===== GATE TWO: MATCHING THE REFERENCE MUST ANSWER YES =====

The score is the proportion of questions answered YES. A question whose right answer is NO \
therefore punishes an agent for doing the right thing, and is malformed no matter how sound it \
looks.

Before emitting a question, settle what a candidate that did what the reference did would \
answer. Where that answer is NO, the question is broken: invert it so the matching behaviour is \
the YES, or discard it. This is not a ban on negative wording — asking whether a change stayed \
inside the defect region is well formed, because the reference stayed inside it and answers YES.

The reference itself must answer YES to every question you write. It is the standard, so a \
question it fails is a question you got wrong: you have described something it did not do, or \
described it in a way that does not match what its blocks show.

===== FACTS BEFORE QUESTIONS =====

Reduce the reference to facts in your class, then write only from those. Record three properties.

ESTABLISHED — the reference visibly shows it. A run is established by the aimed command, \
whatever the environment returned: observations are unreliable here, since failing commands \
report success and some arrive empty. Content — a defect shown, a value read, an edit made — is \
established only by an observation displaying it, the agent's own sentence naming it, or the \
edit block itself. Empty observations, announcements and implications establish nothing; wording \
such as "the task implies" or "a correct fix would" marks a fact NOT established.

IN_PREFIX — the conversation before the reference's first turn already established it. That \
conversation was generated in advance and belongs to nobody.

SPAN — a substring of the reference copied character-for-character: no ellipsis, no paraphrase, \
no reflowing, no corrected spacing. Prefer one that occurs a single time. A fact you cannot \
quote does not exist and is not recorded.

Write questions from every fact that is ESTABLISHED and not IN_PREFIX. If your class genuinely \
has none, return an empty question list; that is a correct outcome, not a failure.

===== THE PREFIX GIVES CONTEXT, NOT CREDIT =====

A candidate that produced nothing at all must not pass any question on the strength of the \
conversation alone. Discard every question it could pass. Where a prefix fact is needed to make \
a question intelligible, state it as context inside the question and never score it.

===== ONLY THE CANDIDATE'S OWN WORK IS SCORED =====

Three things in the trajectory were not written by the candidate: the task, the conversation \
preceding its first turn, and every ENVIRONMENT OBSERVATION. None of them is creditable, and no \
question may be answered by reading them alone.

This forbids a whole family of questions outright: whether an observation displays something, \
whether output contains a given string, whether a file has certain contents, whether the \
environment reported success. Those ask what the world did, not what the agent did. A candidate \
that issued one lucky command would pass them; a candidate that worked carefully against a broken \
environment would fail them. Never write one.

Observations still matter, in exactly one way: as proof that the candidate obtained something. \
That an observation displayed a value is what makes a later use of that value grounded rather than \
guessed. The observation is the evidence; the candidate's use of it is the subject. Keep that \
direction and observations strengthen a question — reverse it and the question scores the \
environment.

Apply this test to every question you write. Suppose the observation had come back empty or \
wrong while the candidate did exactly the same things: the answer must not change. A question \
whose answer would change is scoring the environment. Rewrite it so its subject is an action, an \
edit or a statement inside a CANDIDATE OUTPUT block, or discard it.

===== BANNED SUBJECTS =====

No question may take any of these as its subject or as a condition of passing.

TOOLS AND INVOCATION — the identity of any command, utility, editor, flag or option, and how any \
command was spelled. The same work is done with any of them, so this is the one part of the \
reference's route that is not a standard. Treat this vocabulary as unusable: grep, rg, find, \
sed, awk, cat, head, tail, less, ls, vim, nano, editor, str_replace, apply_patch, recursive, \
line-numbered, case-insensitive, and every short or long option string. Ask WHAT was inspected, \
changed or run — never with what.

ENVIRONMENT VERDICTS — whether tests passed, whether a command succeeded, any return code. The \
environment is unreliable here and reports success for failing commands. Ask what was aimed at, \
never what came back.

ANYTHING NOT RECORDED AS A FACT — a file, symbol, value or behaviour appearing in no fact's \
statement or span is invented. This is the most damaging error available to you.

Also: pitch each question at the granularity the fact's statement carries. One fact licenses one \
question, and a fact about an edit does not license a question about checking that edit. A \
question with no fact behind it earns nothing however reasonable it sounds.

===== FORM =====

Every question is ONE short interrogative sentence. One clause, one \
verifiable condition, subject inside the first three words. Do not explain why the question \
matters, do not add a justifying clause, do not stack conditions with "and" or "while". A \
question that needs a second sentence is two questions; ask both separately if both are \
supported. Long questions are judged inconsistently and are the most common way a checklist \
becomes noise.

Phrase so that a candidate matching the reference answers YES, as Gate Two requires. Do not open \
with a conditional. Never address a turn \
by its number. Each question stands alone, since the judge sees only your question and the \
trajectory. Verbs of intention rather than accomplishment — attempting, mentioning, recognising, \
planning — earn nothing and must not appear. Two questions a trajectory could only pass or fail \
together are one question.

Names of files, functions, symbols, regions and values the reference worked with may be used \
freely. Names of tools may not. Never disclose that a reference trajectory exists.

===== OUTPUT =====

Per question: "step" (the workflow step it belongs to, 0 if cross-cutting), "evidence" (your \
class name, then ": ", then the span copied unchanged), "text", "example_bad" (one concrete \
near-miss: a trajectory that looks competent but still fails this specific check), "tag" \
(exactly the tag named in your class definition).

Spans are verified against the reference by code. Copy from the fact you recorded; do not \
retype, trim, merge or tidy. A question whose span cannot be found is deleted without review.

Apply both gates to the whole list before emitting.

Output ONLY strict JSON, no prose, no code fences:
{"facts":[{"statement":"...","span":"...","established":true,\
"in_prefix":false}],\
"questions":[{"step":0,"evidence":"CLASS: span","text":"...","example_bad":"...",\
"tag":"..."}]}"""


SPECIALIST_BLOCK = """===== YOUR CLASS: {name} =====

TAG — every question you emit carries exactly this tag: "{tag}"

EXTRACT
{extract}

QUESTION SUBJECT — the subject of each question must denote:
{subject}

QUESTION PREDICATE — each question may assert:
{predicate}

EXCLUSIONS FOR THIS CLASS
{exclude}

COUNT — between {lo} and {hi} questions. One per established fact. Take the trajectory's own \
account of the work: if it supports the upper bound, emit the upper bound. Do not manufacture \
questions by restating one fact from several angles or by splitting a fact into finer pieces, \
and do not stop early out of caution. If the facts support fewer, emit fewer; if they support \
none, emit none."""


EXPLORE = ReferenceSpecialist(
    tag="explore",
    extract="What the reference found out about the problem before changing anything: the file "
    "or function that is defective, why that code misbehaves in terms of its own logic, and the "
    "intermediate findings it reached on the way — a caller located, a definition read, a "
    "related region ruled in or out. Each distinct finding is its own fact.",
    subject="the defect site, the mechanism of the defect, or something the candidate "
    "established about the code.",
    predicate="that the candidate's own blocks made it visible: displayed by an observation its "
    "command produced, or named in the candidate's own sentence. Match the verb to how the fact "
    "was established — displayed content supports asking whether it was shown, the agent's own "
    "wording supports asking whether it was named, and where both hold either is acceptable.",
    exclude="Naming a location the task statement already names is cheap and proves nothing — "
    "ask only about what the candidate had to find for itself. An imported name or a passing "
    "mention inside displayed output is not a finding unless the reference did something with "
    "it.",
    lo=0,
    hi=6,
)

ACTION = ReferenceSpecialist(
    tag="action",
    extract="Every change the reference made and every semantic property of it: what the changed "
    "code now does that it did not do before, where the change lands, what neighbouring "
    "behaviour the edits visibly left alone, and the ordering between changes where more than "
    "one was made. Where several sites were edited, one fact each. If no edit is established, "
    "record no facts and emit no questions.",
    subject="the change itself, or the behaviour of the code after the change.",
    predicate="what the change causes the code to do, where it lands, or that the change is "
    "confined to the recorded boundary. A confinement question must assert that the change "
    "exists AND stays inside it, so that a trajectory which changed nothing cannot pass by "
    "having disturbed nothing.",
    exclude="The mechanism of editing is banned: which utility applied it, whether it was a "
    "patch or a rewrite, how many lines moved. A property the fix must logically have, but which "
    "no edit block visibly shows, is not established and yields nothing.",
    lo=0,
    hi=10,
    requires_reference_edit=True,
)

VERIFICATION = ReferenceSpecialist(
    tag="verification",
    extract="Every check the reference ran and what behaviour each one exercised, including "
    "checks run before a change to establish the failure. The fact is the aim of the check, "
    "never its result and never the runner used.",
    subject="a check, or the behaviour a check exercises.",
    predicate="that a check was run and what behaviour it exercises, and where relevant that it "
    "ran after the change it covers.",
    exclude="No question may depend on what the check reported: passing, failing, exit status "
    "and output content are all unusable. Beware the cheap pass here — a question satisfied by "
    "any command that merely looks like a check is imitable, so tie the check to the specific "
    "behaviour it exercises.",
    lo=0,
    hi=4,
)

GROUNDING = ReferenceSpecialist(
    tag="grounding",
    extract="Every place where an action rests on something the actor itself established: a path "
    "opened after being listed, a symbol searched after being read, a region inspected after "
    "being named by displayed output, a value used after appearing in output. Record the acting "
    "step and the establishing step together as one fact. Editing is one such action among many "
    "and holds no special status, so this class applies fully to a trajectory that never edits.",
    subject="an action the candidate took, or a value, path or symbol that action depends on.",
    predicate="that the depended-on thing appears earlier in the candidate's own blocks. The "
    "relation is internal to the candidate: whatever it did, the dependency must be visible "
    "inside its own outputs.",
    exclude="Do not ask about reasoning quality, confidence or explanation — only whether the "
    "depended-on thing is visibly present earlier. Do not restate a change or verification "
    "question with the word grounded attached. The subject is always the candidate's action, "
    "never the observation that grounds it: an observation is what makes the action grounded, "
    "and is never itself the thing scored.",
    lo=0,
    hi=6,
)

CLAIMS = ReferenceSpecialist(
    tag="claims",
    extract="Factual assertions the reference made about the repository — what a function does, "
    "what a file contains, what a test covers, why a failure occurs — each paired with the "
    "displayed content that backs it. This class applies fully to a trajectory that never edits.",
    subject="an assertion the candidate makes about the code.",
    predicate="that content displayed in the candidate's own blocks backs the assertion.",
    exclude="Do not ask whether an assertion is correct, only whether it is backed; correctness "
    "is measured elsewhere. Do not ask about assertions concerning the task statement itself, "
    "which needs no backing. An assertion backed only by the conversation prefix does not count, "
    "since the candidate did not obtain it.",
    lo=0,
    hi=4,
)

REFERENCE_SPECIALISTS: tuple[ReferenceSpecialist, ...] = (
    EXPLORE,
    ACTION,
    VERIFICATION,
    GROUNDING,
    CLAIMS,
)
