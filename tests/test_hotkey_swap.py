from chain_guard.swap import SwapEvent, describe, find_swaps


def test_swap_detected_when_hotkey_changes_but_reg_block_does_not():
    prior = {190: ("hk-old", 8523241)}
    current = [(190, "hk-new", 8523241)]

    assert find_swaps(prior, current) == [SwapEvent(190, "hk-old", "hk-new", 8523241)]


def test_rereg_not_flagged_when_reg_block_bumped():
    prior = {5: ("hk-old", 8500000)}
    current = [(5, "hk-new", 8591000)]

    assert find_swaps(prior, current) == []


def test_unchanged_hotkey_not_flagged():
    prior = {7: ("hk-same", 8500000)}
    current = [(7, "hk-same", 8500000)]

    assert find_swaps(prior, current) == []


def test_null_prior_reg_block_skipped():
    # pre-backfill rows have no stored registration block — cannot judge, must not flag
    prior = {9: ("hk-old", None)}
    current = [(9, "hk-new", 8500000)]

    assert find_swaps(prior, current) == []


def test_uid_without_prior_state_skipped():
    prior = {}
    current = [(3, "hk-new", 8551113)]

    assert find_swaps(prior, current) == []


def test_mixed_metagraph_flags_only_the_swap():
    prior = {
        1: ("hk-a", 100),
        2: ("hk-b", 200),
        3: ("hk-c", 300),
    }
    current = [
        (1, "hk-a", 100),      # unchanged
        (2, "hk-b2", 999),     # re-registered
        (3, "hk-c2", 300),     # swapped
        (4, "hk-d", 400),      # new uid, no history
    ]

    assert find_swaps(prior, current) == [SwapEvent(3, "hk-c", "hk-c2", 300)]


def test_describe_shows_before_after_with_identical_reg_block():
    msg = describe(SwapEvent(190, "5FcPXrvJ", "5DUfpADA", 8523241).detail())

    assert "uid 190" in msg
    assert "BEFORE hotkey=5FcPXrvJ BlockAtRegistration=8523241" in msg
    assert "AFTER hotkey=5DUfpADA BlockAtRegistration=8523241" in msg
    assert "identical" in msg
