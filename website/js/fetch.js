import { DATA_ENDPOINTS, STATE_ENDPOINTS, BENCHMARK_ENDPOINTS, PULLED_SUITES, MANIFEST_ENDPOINTS, LLMS_URLS, REGISTRATION_ENDPOINTS } from "./config.js";

const registrationCacheKey = "albedo.registrationHistory.v2";

// Every feed is served with a validator and answers 304, so a conditional request is
// both always-fresh and cheap; a cache-busting query would force a full re-download
// on every poll instead.
const NO_CACHE = { cache: "no-cache" };

async function fetchFirstJson(endpoints) {
  for (const url of endpoints) {
    try {
      const r = await fetch(url, NO_CACHE);
      if (!r.ok) continue;
      return await r.json();
    } catch {}
  }
  return null;
}

export async function fetchDashboard() {
  return fetchFirstJson(DATA_ENDPOINTS);
}

export async function fetchState() {
  return fetchFirstJson(STATE_ENDPOINTS);
}

export async function fetchBenchmarks() {
  return fetchFirstJson(BENCHMARK_ENDPOINTS);
}

// One score file per pulled suite, keyed by suite so callers never have to know
// which file a suite came from.
export async function fetchPulledScores() {
  const entries = await Promise.all(PULLED_SUITES.map(async pulled => {
    const rows = await fetchFirstJson(pulled.scoreEndpoints);
    return [pulled.suite, Array.isArray(rows) ? rows : []];
  }));
  return new Map(entries);
}

export async function fetchBenchmarkRun(run) {
  if (!run?.detail_path) return null;
  for (const endpoint of BENCHMARK_ENDPOINTS) {
    const base = endpoint.slice(0, endpoint.lastIndexOf("/") + 1);
    try {
      const r = await fetch(base + run.detail_path, NO_CACHE);
      if (!r.ok) continue;
      const payload = await r.json();
      return payload?.run || payload;
    } catch {}
  }
  return null;
}

export async function fetchManifest() {
  return fetchFirstJson(MANIFEST_ENDPOINTS);
}

export async function fetchLlmsText() {
  for (const url of LLMS_URLS) {
    try {
      const r = await fetch(url, NO_CACHE);
      if (!r.ok) continue;
      return await r.text();
    } catch {}
  }
  return null;
}

export async function fetchRegistrationHistory() {
  for (const url of REGISTRATION_ENDPOINTS) {
    try {
      const r = await fetch(url, NO_CACHE);
      if (!r.ok) continue;
      const data = await r.json();
      try { localStorage.setItem(registrationCacheKey, JSON.stringify(data)); } catch {}
      return data;
    } catch {}
  }
  try { return JSON.parse(localStorage.getItem(registrationCacheKey)); } catch { return null; }
}

const PRED_MARKER = '"instance_id":';
const predsState = new Map();

function countMarkers(text) {
  let n = 0;
  for (let i = text.indexOf(PRED_MARKER); i !== -1; i = text.indexOf(PRED_MARKER, i + PRED_MARKER.length)) n++;
  return n;
}

// Ranged read of a file that is still being appended to: byte offsets only line up
// against the live object, so this one request must never be served from a cache.
async function readPredsTail(url, offset) {
  const from = Math.max(0, offset - PRED_MARKER.length + 1);
  const r = await fetch(url, { cache: "no-store", headers: from ? { Range: `bytes=${from}-` } : {} });
  if (!r.ok) return null;
  const text = await r.text();
  const end = Number(r.headers.get("content-range")?.split("/")[0]?.split("-")[1]);
  return {
    found: countMarkers(text),
    partial: from > 0 && r.status === 206,
    offset: Number.isFinite(end) ? end + 1 : from + text.length,
  };
}

export async function fetchPredsProgress(bases, runIds) {
  const ids = (Array.isArray(runIds) ? runIds : [runIds]).filter(Boolean);
  const heads = [];
  for (const runId of ids) {
    for (const base of bases) {
      const url = `${base}/${runId}/preds.json`;
      try {
        const head = await fetch(url, { method: "HEAD", ...NO_CACHE });
        if (!head.ok) continue;
        const size = Number(head.headers.get("content-length"));
        if (!Number.isFinite(size) || size <= 0) continue;
        const updatedAt = head.headers.get("last-modified") || null;
        const modified = new Date(updatedAt || 0).getTime();
        heads.push({ runId, url, size, updatedAt, modified: Number.isFinite(modified) ? modified : 0 });
        break;
      } catch {}
    }
  }
  if (!heads.length) return null;
  // the run being generated right now is the one whose preds file was written most recently
  const target = heads.sort((a, b) => b.modified - a.modified)[0];
  try {
    const prev = predsState.get(target.url);
    const state = prev && target.size >= prev.offset ? prev : { offset: 0, count: 0, first: null };
    if (target.size > state.offset) {
      const tail = await readPredsTail(target.url, state.offset);
      if (!tail) return null;
      state.count = tail.partial ? state.count + tail.found : tail.found;
      state.offset = tail.offset;
    }
    state.first ||= { at: Date.now(), count: state.count };
    predsState.set(target.url, state);
    const elapsed = Date.now() - state.first.at;
    const gained = state.count - state.first.count;
    return {
      runId: target.runId,
      count: state.count,
      updatedAt: target.updatedAt,
      rate: elapsed > 30000 && gained > 0 ? gained / elapsed : null,
    };
  } catch { return null; }
}

export async function fetchText(url) {
  try {
    const r = await fetch(url, NO_CACHE);
    if (!r.ok) return null;
    return await r.text();
  } catch { return null; }
}

export async function fetchJson(url) {
  const t = await fetchText(url);
  if (t == null) return null;
  try { return JSON.parse(t); } catch { return null; }
}
