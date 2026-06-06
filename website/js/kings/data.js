import { BITTENSOR_BLOCK_TIME_S } from "./config.js";

export function buildKingsList(d) {
  const chain   = d.king_chain || [];
  const history = d.history || [];
  const kings = [];
  const seenIds = new Set();

  chain.forEach(k => {
    seenIds.add(k.challenge_id);
    kings.push({ ...k, _rich: true, _inferred: false });
  });

  history
    .filter(h => h.accepted && !seenIds.has(h.challenge_id))
    .forEach(h => {
      seenIds.add(h.challenge_id);
      kings.push({
        challenge_id:  h.challenge_id,
        hotkey:        h.hotkey,
        uid:           h.uid ?? null,
        model_repo:    h.model_repo,
        king_digest:   h.model_digest,
        crowned_at:    h.completed_at,
        crowned_block: null,
        // A crowned challenger's reign = the champion it beat (king_reign_number) + 1,
        // so kingTitle() can name it (ALBEDO-I …) instead of falling back to base model.
        reign_number:  h.king_reign_number != null ? h.king_reign_number + 1 : null,
        weight:        null,
        registered:    null,
        judges:        h.judges || [],
        _rich:         false,
        _inferred:     false,
      });
    });

  kings.sort((a, b) => {
    const ta = a.crowned_at ? new Date(a.crowned_at).getTime() : 0;
    const tb = b.crowned_at ? new Date(b.crowned_at).getTime() : 0;
    return tb - ta;
  });

  const chainTimes = chain.map(k => k.crowned_at ? new Date(k.crowned_at).getTime() : Infinity);
  const oldestChainMs = chainTimes.length ? Math.min(...chainTimes) : Infinity;
  const knownMin = chain.reduce((mn, k) =>
    k.reign_number != null && k.reign_number < mn ? k.reign_number : mn, Infinity);

  if (knownMin !== Infinity) {
    let counter = knownMin - 1;
    for (let i = 0; i < kings.length; i++) {
      const k = kings[i];
      if (k.reign_number != null) continue;
      const kt = k.crowned_at ? new Date(k.crowned_at).getTime() : 0;
      if (kt < oldestChainMs && counter >= 0) {
        kings[i] = { ...kings[i], reign_number: counter, _inferred: true };
        counter--;
      }
    }
  }

  return kings;
}

export function applyDisplayStartBlock(d) {
  const startBlock = d.chain?.display_start_block;
  if (!startBlock || startBlock <= 0) return d;
  const ref = d.king || (d.king_chain || [])[0];
  const refBlock = ref?.crowned_block;
  const refAt    = ref?.crowned_at;
  if (!refBlock || !refAt) return d;
  const refMs    = new Date(refAt).getTime();
  const cutoffMs = refMs - (refBlock - startBlock) * BITTENSOR_BLOCK_TIME_S * 1000;
  return {
    ...d,
    king_chain: (d.king_chain || []).filter(k =>
      (k.crowned_block != null ? k.crowned_block >= startBlock : true) ||
      (k.crowned_at ? new Date(k.crowned_at).getTime() >= cutoffMs : true)
    ),
    history: (d.history || []).filter(h =>
      !h.completed_at || new Date(h.completed_at).getTime() >= cutoffMs
    ),
    _cutoff: { block: startBlock, ms: cutoffMs },
  };
}
