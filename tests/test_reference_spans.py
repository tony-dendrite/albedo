"""What `normalise_span` folds, and what it must leave alone.

The line-prefix fold exists because a model that reads a file through `nl -ba` or `grep -n` quotes
the code without the numbering, which is correct of it. Without the fold, that faithfully copied
span then fails to occur in the observation it came from, and the evidence is rejected for a
formatting difference rather than a real one.
"""

from __future__ import annotations

from albedo_eval_service.evaluator.reference.questions import normalise_span


def test_a_span_quoted_without_nl_numbering_matches_the_numbered_observation():
    observation = "     6\ttype StreamConsumer struct {\n     7\t\tName string\n"
    span = "type StreamConsumer struct {\n\tName string"

    assert normalise_span(span) in normalise_span(observation)


def test_a_span_quoted_without_grep_numbering_matches_the_numbered_observation():
    observation = "./memprofiler/stream.go:5:func sizeOfStreamObject(obj *model.StreamObject) int {"
    span = "func sizeOfStreamObject(obj *model.StreamObject) int {"

    assert normalise_span(span) in normalise_span(observation)


def test_the_digits_then_colon_form_folds_too():
    assert normalise_span("  12:  size := 8") == "size := 8"


def test_a_span_that_already_matched_still_matches():
    """A span needing no fold must still match: the fold may only add matches, never remove one."""
    observation = "dists = []\n    for dist in original_dists:\n        name = dist.metadata.get()"
    span = "for dist in original_dists:"

    assert normalise_span(span) in normalise_span(observation)


def test_prose_beginning_with_a_number_is_left_alone():
    """Why the digits-then-single-space form is deliberately not folded."""
    assert normalise_span("1274 characters elided") == "1274 characters elided"


def test_data_rows_beginning_with_a_number_are_left_alone():
    assert normalise_span("0,listpack,stream,10852,10.6K") == "0,listpack,stream,10852,10.6K"


def test_whitespace_and_smart_quotes_still_fold():
    assert normalise_span("a   b\n\tc") == "a b c"
    assert normalise_span("say “hello”") == 'say "hello"'


def test_the_fold_happens_before_the_whitespace_collapse():
    """After collapsing, `     6\\ttype X` reads `6 type X` and the anchors have nothing to bite."""
    assert normalise_span("     6\ttype X") == "type X"
    # the same characters with the newline structure gone are NOT a line prefix any more
    assert normalise_span("6 type X") == "6 type X"
