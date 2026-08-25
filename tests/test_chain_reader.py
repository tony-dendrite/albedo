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
        ]

    def metagraph(self, netuid):
        return FakeMeta()

    def get_block_hash(self, block):
        return f"0x{block}"


def test_scan_commitments_accepts_only_three_part_v7_payloads():
    commits = scan_commitments(FakeSubtensor(), 1)

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


def test_resolve_missing_coldkeys_only_for_registered_hotkeys(monkeypatch):
    import asyncio

    from chain_reader import db, reader

    stored = {}

    async def fake_missing(pool):
        return ["hk-reg", "hk-gone", "hk-no-owner"]

    async def fake_set(pool, hk, ck):
        stored[hk] = ck

    owners = {"hk-reg": "ck-1", "hk-no-owner": None}
    monkeypatch.setattr(db, "hotkeys_missing_coldkey", fake_missing)
    monkeypatch.setattr(db, "set_coldkey", fake_set)
    monkeypatch.setattr(reader.chain, "hotkey_owner", lambda st, hk, block: owners.get(hk))

    n = asyncio.run(reader.resolve_missing_coldkeys(None, object(), {"hk-reg", "hk-no-owner"}, 123))
    assert n == 1
    assert stored == {"hk-reg": "ck-1"}
