# Mining on Albedo (SN97) — how to commit a model

Albedo is a **king-of-the-hill** subnet. You upload a challenger model to Hippius, then
post an on-chain *reveal* pointing at it. The validator downloads your challenger and duels
it against the reigning king on SWE-ZERO coding trajectories. **Beat the king by enough and
you take the crown** (and the emissions).

This guide is the practical checklist: what you need, how to name things, what to verify,
how to commit, and how to confirm it landed.

---

## 1. Before you start — prerequisites

- [ ] **Hotkey registered on netuid 97** (`btcli subnet register --netuid 97`).
- [ ] A **Bittensor wallet** with that coldkey/hotkey locally (`~/.bittensor/wallets/`).
- [ ] A **Hippius Hub push token** — `HIPPIUS_HUB_TOKEN`. This is what lets you upload weights.
- [ ] Your **Hippius namespace** (your Hippius username/org) — your repos live under it.
- [ ] The repo installed: `uv pip install -e .` (add `[train]` for SFT).
- [ ] A model that actually **improves on the king**. The built-in `--noise` perturbation is
      only a pipeline smoke test — it will *not* dethrone a trained king.

Set the environment once per shell:

```bash
export ALBEDO_NETUID=97
export ALBEDO_NETWORK=finney
export BT_WALLET_NAME=<your-wallet>
export ALBEDO_CHALLENGER_NAMESPACE=<your-hippius-namespace>
export HIPPIUS_HUB_TOKEN=<your-hippius-token>
```

---

## 2. Naming rules (get these wrong → instant rejection)

**Repo name** must match the pattern in `chain.toml [chain].repo_pattern`:

```
^[^/]+/albedo-qwen3-4b-.+$
```

In plain terms: `<namespace>/albedo-qwen3-4b-<suffix>`, lowercase.

| Example                          | Valid? | Why |
|----------------------------------|--------|-----|
| `alice/albedo-qwen3-4b-v1`       | ✅     | matches pattern |
| `bob/albedo-qwen3-4b-sft-run3`   | ✅     | suffix can be anything non-empty |
| `alice/albedo-qwen3-4b`          | ❌     | missing the `-<suffix>` part |
| `alice/qwen3-4b-v1`              | ❌     | missing the `albedo-` prefix |
| `Alice/Albedo-Qwen3-4B-V1`       | ❌     | must be lowercase |

> The `qwen3-4b` in the name is not cosmetic — this competition is locked to the **Qwen3-4B
> size class**. Your challenger's architecture must match the king exactly (see §3).

**Reveal string** (what goes on chain) is built for you, in this format:

```
v4|<repo>|<digest>
```

The hotkey is **not** in the string — the wallet/hotkey that **signs the commit** is the
authority. Just make sure you commit with the hotkey you registered on SN97.

---

## 3. What to check before you commit

The validator runs an **admission gate** before it ever spends GPU time. If your challenger
fails any of these, it's rejected silently (no duel). Check them locally first:

1. **Repo name** matches the pattern above.
2. **Digest** is a Hippius `sha256:` blob (the upload tooling produces this — not `hf:`).
3. **`config.json` is present** in the uploaded snapshot.
4. **`architectures` == `["Qwen3ForCausalLM"]`** (same as the king).
5. **All arch-lock keys match the king exactly:**
   `vocab_size`, `model_type`, `max_position_embeddings`, `tie_word_embeddings`, `rope_theta`,
   `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
   `intermediate_size`, `head_dim`.
6. **No `auto_map`** key in `config.json`.
7. **No `quantization_config`** — quantized models are rejected.
8. **No `*.py` files** anywhere in the repo (no custom modeling code / `trust_remote_code`).
9. **At least one `.safetensors`** file.

Plus two runtime rules that aren't config checks:

- **One eval per hotkey** — you can't queue two challengers from the same hotkey at once.
- **The current king can't challenge itself.**

Both `miner.py` (`validate_local_config()`) and `scripts/upload_challenger.py`
(`check_arch_compat()`) mirror these checks so you catch a mismatch **before** uploading.

To see the king's current arch values, pull the dashboard:

```bash
python3 -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('https://us-east-1.hippius.com/albedo/dashboard.json'))
print('King repo:', d['king'].get('model_repo'))
print('King digest:', d['king'].get('king_digest') or d['king'].get('model_digest'))
"
```

---

## 4. Commit your model with `miner.py`

`miner.py` runs the whole pipeline end-to-end and **commits the reveal for you** — it's the
recommended one-command path:

> discover the king (from `dashboard.json`, else the `chain.toml` seed) → download the king →
> `train_or_perturb()` → `validate_local_config()` (the arch-lock gate from §3) → upload to
> Hippius → post the on-chain reveal.

```bash
python miner.py --hotkey <hotkey_name>
```

**Flags:**
| Flag | Default | Purpose |
|---|---|---|
| `--hotkey` | `h0` | Your hotkey name (under `BT_WALLET_NAME`); also the default repo suffix |
| `--suffix` | `<hotkey>` | Override the repo suffix → repo = `<namespace>/albedo-qwen3-4b-<suffix>` |
| `--noise` | `0.001` | Stddev for the built-in perturbation stub |

On success it prints: `reveal committed — validator picks up within ~30 s`.

### Smoke test (out of the box)

As shipped, `train_or_perturb()` just copies the king and adds gaussian noise — enough to prove
your wallet, token, namespace, upload, and reveal all work. **It will not dethrone a trained king.**

```bash
python miner.py --hotkey <hotkey_name> --noise 0.001
```

### Real mining (replace the stub)

To submit a model that can actually win, replace `train_or_perturb(king_dir, chal_dir, noise)`
in `miner.py` with your own SFT/RL loop — it just needs to write your trained challenger weights
into `chal_dir`. Typical prep first:

```bash
# 1. SFT data from past public duels (winning challenger turns)
python scripts/collect_traces.py --out data/traces.jsonl --min-delta 0.05
python scripts/inspect_dataset.py data/traces.jsonl          # sanity-check health

# 2. Fine-tune from the Qwen3-4B base
python scripts/train_sft.py --base Qwen/Qwen3-4B --data data/traces.jsonl --output ckpt/

# 3. Local format check (1 bash block, no injected verdict JSON, non-empty)
python scripts/sanity_check.py ckpt/
```

Then point `train_or_perturb()` at `ckpt/` (e.g. copy it into `chal_dir` instead of perturbing)
and run `python miner.py --hotkey <hotkey_name>`. It re-validates against the king, uploads, and
commits in one shot.

> Don't want to edit `miner.py`? `scripts/upload_challenger.py --model ckpt/ --repo <ns>/albedo-qwen3-4b-v1 --hotkey <ss58>`
> uploads a checkpoint and prints a reveal string to commit manually — but `miner.py` is the
> recommended path.

---

## 5. Verify the reveal landed on chain

After committing, confirm it's actually there with the read-only inspector:

```bash
python scripts/check_commits.py                       # all v4 commits on chain
python scripts/check_commits.py --hotkey <your-ss58>  # just yours
```

Read the `status` column:

- **`ok`** — valid reveal; the validator picks it up on its next chain scan (~20–30 s).
- **no rows for your hotkey** — the commit didn't land (or hasn't propagated). Wait a few
  seconds and re-run; if still missing, re-check the reveal string and the wallet/hotkey used.

---

## 6. Then watch the duel

```bash
# Dashboard: current king + queue depth
python3 -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('https://us-east-1.hippius.com/albedo/dashboard.json'))
print('King:', d['king'].get('model_repo'), '| Queue:', len(d.get('queue', [])))
"
```

**How dethroning works:** your challenger duels the king over 64 SWE-ZERO trajectories, scored
0–100 by an ensemble of 3 LLM judges. To take the crown you must clear **both** gates:

1. **Margin gate** — `challenger_score − king_score ≥ win_margin` (currently **2.0** points).
2. **Significance gate** — paired-bootstrap lower confidence bound > 0 at α = 0.05 (i.e. the win
   isn't statistical noise).

Beating the king by a hair is not enough — beat it clearly and consistently.

---

## Quick reference

| Thing | Value |
|---|---|
| Netuid | `97` (finney) |
| Repo pattern | `^[^/]+/albedo-qwen3-4b-.+$` (lowercase) |
| Reveal format | `v4\|<repo>\|<digest>` (hotkey = the commit signer, not in the string) |
| Base model / size class | `Qwen/Qwen3-4B` (arch-locked) |
| Win margin | `2.0` points (0–100 scale) |
| Trajectories per duel | 64 |
| Required env | `ALBEDO_NETUID`, `BT_WALLET_NAME`, `ALBEDO_CHALLENGER_NAMESPACE`, `HIPPIUS_HUB_TOKEN` |
| Commit (recommended) | `python miner.py --hotkey <name>` (train→upload→reveal in one shot) |
| Upload only (alt) | `python scripts/upload_challenger.py --model ckpt/ --repo <ns>/albedo-qwen3-4b-v1 --hotkey <ss58>` |
| Verify commit | `python scripts/check_commits.py --hotkey <ss58>` |

For the full system internals (validator/eval-server flow, judge transport, config reference),
see [`llms.txt`](llms.txt).
