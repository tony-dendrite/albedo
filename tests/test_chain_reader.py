from chain_reader.chain import scan_commitments


class FakeNeuron:
    def __init__(self, hotkey, uid):
        self.hotkey = hotkey
        self.uid = uid


class FakeMeta:
    neurons = [
        FakeNeuron("hk-good", 7),
        FakeNeuron("hk-v5", 8),
        FakeNeuron("hk-four", 9),
        FakeNeuron("hk-hf", 10),
        FakeNeuron("hk-mutable", 11),
        FakeNeuron("hk-private", 12),
    ]


def _reveal(payload: str) -> str:
    return "0x" + (b"\x00" + payload.encode()).hex()


class FakeSubtensor:
    def query_map(self, module, name, params, **kwargs):
        assert (module, name) == ("Commitments", "RevealedCommitments")
        return [
            ("hk-good", [(_reveal("v7|alice/model|sha256:" + "a" * 64), 100)]),
            ("hk-v5", [(_reveal("v5|alice/model|sha256:" + "b" * 64), 101)]),
            ("hk-four", [(_reveal("v7|alice/model|sha256:" + "c" * 64 + "|hk-four"), 102)]),
            ("hk-hf", [(_reveal("v7|alice/model-hf|" + "d" * 40), 103)]),
            ("hk-mutable", [(_reveal("v7|alice/model|main"), 104)]),
            (
                "hk-private",
                [
                    (_reveal("r2activate:v1:" + "A" * 86), 105),
                    (_reveal("r2ready:v1:" + "e" * 64), 106),
                ],
            ),
        ]

    def metagraph(self, netuid):
        return FakeMeta()

    def get_block_hash(self, block):
        return f"0x{block}"


def test_scan_commitments_accepts_only_three_part_v7_payloads():
    commits, signals = scan_commitments(FakeSubtensor(), 1)

    assert len(commits) == 2
    by_hotkey = {c.hotkey: c for c in commits}

    commit = by_hotkey["hk-good"]
    assert commit.uid == 7
    assert commit.model_uri == "alice/model@sha256:" + "a" * 64
    assert commit.commit_payload == {
        "version": "v7",
        "repo": "alice/model",
        "digest": "sha256:" + "a" * 64,
        "author_hotkey": "hk-good",
        "spoofed": False,
    }

    hf_commit = by_hotkey["hk-hf"]
    assert hf_commit.uid == 10
    assert hf_commit.model_uri == "alice/model-hf@" + "d" * 40
    assert "hk-mutable" not in by_hotkey


def test_scan_commitments_collects_private_store_signals():
    _, signals = scan_commitments(FakeSubtensor(), 1)
    assert [(s.kind, s.hotkey, s.uid, s.block_number) for s in signals] == [
        ("activate", "hk-private", 12, 105),
        ("ready", "hk-private", 12, 106),
    ]
    assert signals[0].payload == "r2activate:v1:" + "A" * 86


def test_private_payloads_survive_the_real_chain_decode_path():
    from private_store.contracts import (
        activation_signal_payload,
        ready_signal_payload,
    )

    activate = activation_signal_payload(b"s" * 32)
    ready = ready_signal_payload("e" * 64)
    v7 = "v7|some-owner/some-model|sha256:" + "a" * 64

    class _Sub(FakeSubtensor):
        def query_map(self, module, name, params, **kwargs):
            return [
                ("hk-private", [(_reveal(activate), 200), (_reveal(ready), 201)]),
                ("hk-good", [(_reveal(v7), 202)]),
            ]

    commits, signals = scan_commitments(_Sub(), 1)
    assert {s.payload for s in signals} == {activate, ready}
    assert [c.model_uri for c in commits] == ["some-owner/some-model@sha256:" + "a" * 64]
    assert max(len(activate.encode()), len(ready.encode())) <= 128
