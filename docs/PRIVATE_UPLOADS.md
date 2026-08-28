# Private model uploads (Albedo SN97)

Reference for a coding agent driving the `albedo` miner CLI. This is the **private
submission** path: your model is uploaded to the subnet's private bucket instead of a
public HuggingFace/Hippius repo, so competitors can't see or copy it. It only becomes
public if it wins and becomes king.

For model requirements (valid Qwen3.6-35B-A3B checkpoint, file allowlist, architecture
lock, win margin, etc.) see [MINING.md](MINING.md) — those are unchanged. This doc covers
**only how the upload works**.

## Mental model

You never hand the validator a public URL. Instead:

1. You commit on-chain that you want to submit (`r2activate`). This publishes a one-time
   public key.
2. The validator sees it, mints storage credentials scoped **only to your own folder**,
   encrypts them to that key, and drops them in a public "mailbox".
3. Your CLI fetches and decrypts the credentials, uploads your model + a signed manifest
   into your folder.
4. You commit on-chain that you're done (`r2ready`). This **freezes** the upload (your
   credentials are revoked), the validator verifies the files match your commitment, and
   evaluation proceeds exactly as before.

Any hotkey type works (sr25519 or ed25519) — the hotkey only signs the two on-chain
commits. A separate one-time key (generated locally) is what the credentials are encrypted
to and what signs your manifest.

## One command

```bash
albedo submit-private --path ./my-model --name my-model-v1
```

This runs the whole flow end to end: `activate` → wait for credentials → `upload` →
`ready`. Wallet/network flags match the rest of the CLI:

| flag | meaning | default |
| --- | --- | --- |
| `--path` | local model directory to upload | required |
| `--name` | model name recorded in the manifest | required |
| `--coldkey` | wallet (coldkey) name | `$ALBEDO_COLDKEY` |
| `--hotkey` | hotkey name | `$ALBEDO_HOTKEY` |
| `--netuid` | subnet id | `97` |
| `--network` | bittensor network | `$CHAIN_NETWORK` / `finney` |
| `--yes` | skip the on-chain confirm prompts | off |

## Configuration the agent must set

Two public values, published by the subnet operator (announcement / pinned config). The
CLI needs them in the environment or `.env`:

| env var | what it is |
| --- | --- |
| `R2_MAILBOX_PUBLIC_BASE_URL` | public URL of the credential mailbox the CLI polls |
| `R2_CHAIN_GENERATION` | submission-round identifier; **must match the validator** (default `albedo-mainnet-1`) |

Plus the usual wallet config (`ALBEDO_COLDKEY`, `ALBEDO_HOTKEY`, optionally
`ALBEDO_WALLET_PATH`). You do **not** configure the bucket, endpoint, or your folder path —
those arrive inside the encrypted credentials.

## The two on-chain commitments

Both are submitted with `set_reveal_commitment` (the CLI does this for you):

| commitment | format | meaning |
| --- | --- | --- |
| activate | `r2activate:v1:<base64url 32-byte pubkey>` | requests upload credentials; publishes your one-time submission public key |
| ready | `r2ready:v1:<64-hex manifest sha256>` | freezes the upload and triggers verification + eval |

## Local state (do not delete mid-flow)

`activate` writes files under `./.albedo-miner/<hotkey_ss58>/` (mode `0600`):

- `submission-key` — the secret half of your one-time key. The credentials are encrypted
  to its public half, so this file is what **decrypts** them, and it **signs** your
  manifest. It must survive from `activate` until `ready`, on the **same machine**.
- `upload-auth.json` — the decrypted credentials, written by the upload step. Removed
  automatically at `ready`.

If you lose `submission-key` between `activate` and `ready`, you cannot decrypt your
credentials and cannot recover this submission (the public key is already committed
on-chain and can't be changed). Since it's one submission per hotkey, you'd have to start
over with a fresh hotkey. Do **not** delete `.albedo-miner/` until `ready` completes.

## Running the steps separately

`submit-private` is the happy path. For manual control or recovery, run the steps
individually (same wallet/network flags):

```bash
albedo activate                                   # commit r2activate, write submission-key
albedo upload-private --path ./my-model --name my-model-v1   # fetch creds, upload model+manifest
albedo ready --manifest-sha256 <hash>             # commit r2ready (hash printed by upload-private)
```

- `activate` is idempotent: re-running reuses the existing `submission-key` (it does **not**
  generate a new one), so it's safe to retry.
- `upload-private` polls the mailbox for up to ~15 minutes, then uploads. If it fails
  partway (network drop), just run it again — it re-uploads the files and writes
  `manifest.json` last.
- `ready` needs the `manifest_sha256` that `upload-private` prints on success.

## Constraints and failure modes

- **24-hour window.** You must finish uploading and commit `r2ready` within 24h of
  `activate`, or the submission is abandoned (credentials revoked, folder wiped) and you
  start over.
- **One submission per hotkey** (same as the public flow).
- **Upload caps.** Total upload ≤ 100 GB and ≤ 4096 objects; exceeding either during the
  window abandons the submission. A normal checkpoint is far under both.
- **Verification (after `ready`)** rejects the submission if the uploaded files don't match
  your committed manifest, if a genesis/contract file was altered, if there are undeclared
  or missing files, or if the model fails the standard Qwen3.6-35B-A3B checks in
  [MINING.md](MINING.md). Fix locally and submit from a fresh hotkey.
- Your bytes stay private throughout. They are only copied to public HuggingFace if the
  model wins and becomes king; losing submissions are never exposed.

## Preflight (optional but recommended)

Validate the model locally before spending a submission:

```bash
albedo check-model --path ./my-model
```

This runs the same file-manifest and architecture checks the validator will, without
uploading or committing anything.
