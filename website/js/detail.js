/**
 * Eval detail subpage.
 * URL params: eval_id, hotkey, dir_url, error_code, error_detail, model_repo
 *
 * Valid eval   → fetch dir_url/scores.json + judge_raw.jsonl
 * Failure      → fetch /dashboard.json and find entry by eval_id
 */

const params      = new URLSearchParams(location.search);
const evalId      = params.get("eval_id")      || "";
const hotkey      = params.get("hotkey")       || "";
const dirUrl      = (params.get("dir_url") || "").replace(/\/$/, "");
const errorCode   = params.get("error_code")   || "";
const errorDetail = params.get("error_detail") || "";
const modelRepo   = params.get("model_repo")   || "";

const root = document.getElementById("detail-root");

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function badge(accepted, errCode) {
  if (errCode === "chal_injection_detected") return `<span class="verdict-badge injection">injection</span>`;
  if (errCode === "duplicate_model")          return `<span class="verdict-badge duplicate">duplicate</span>`;
  if (errCode === "identity_mismatch" || errCode === "not_registered")
                                              return `<span class="verdict-badge invalid">invalid</span>`;
  if (errCode)                                return `<span class="verdict-badge error">error</span>`;
  if (accepted)                               return `<span class="verdict-badge crowned">crowned</span>`;
  return `<span class="verdict-badge lost">lost</span>`;
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} — ${url}`);
  return r.text();
}

function parseJsonl(text) {
  return text.split("\n").map(l => l.trim()).filter(Boolean)
    .map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}

// ------------------------------------------------------------------ //
// Failure entry rendering (injection, duplicate, infra, etc.)         //
// ------------------------------------------------------------------ //

function renderFailure(entry) {
  const code        = entry.error_code  || errorCode  || "";
  const detail      = entry.error_detail || errorDetail || "";
  const cHotkey     = entry.hotkey || hotkey || "—";
  const kHotkey     = entry.king_hotkey || "—";
  const date        = entry.completed_at ? entry.completed_at.slice(0, 10) : "—";
  const probeDetails = entry.probe_details || [];

  const panelCls = code === "chal_injection_detected" ? "injection"
                 : code === "duplicate_model"          ? "duplicate"
                 : code.includes("infra") || code.includes("vllm") ? "infra"
                 : "invalid";

  const panelLabel = code === "chal_injection_detected" ? "Injection detected"
                   : code === "duplicate_model"          ? "Duplicate model"
                   : code === "identity_mismatch"        ? "Spoofed identity"
                   : code === "not_registered"           ? "Not registered"
                   : code || "Error";

  let html = `
<div class="detail-meta">
  <h1>${esc(entry.eval_id || evalId || "—")}</h1>
  <div class="sub">
    <span>challenger: <code>${esc(cHotkey)}</code></span>
    <span class="sep">·</span>
    <span>vs. king: <code>${esc(kHotkey)}</code></span>
    <span class="sep">·</span>
    <span>date: ${esc(date)}</span>
    <span class="sep">·</span>
    ${badge(false, code)}
  </div>
</div>
<div class="fail-panel ${panelCls}">
  <strong>${esc(panelLabel)}</strong>
  <span class="detail-text">${esc(detail)}</span>
</div>`;

  // Per-judge injection table
  if (code === "chal_injection_detected" && probeDetails.length) {
    // Aggregate: collect all judge verdicts across probe turns
    const judgeMap = {}; // judge_model -> {triggered, evidences[]}
    probeDetails.forEach(probe => {
      const byJ = probe.injections_by_judge || {};
      const evJ = probe.evidence_by_judge   || {};
      Object.entries(byJ).forEach(([jm, triggered]) => {
        if (!judgeMap[jm]) judgeMap[jm] = { triggered: false, evidences: [] };
        if (triggered === true) judgeMap[jm].triggered = true;
        const ev = evJ[jm];
        if (ev && ev !== "none" && ev !== "") judgeMap[jm].evidences.push(ev);
      });
      // Also from triggered_details
      (probe.triggered_details || []).forEach(td => {
        const jm = td.judge_model;
        if (!judgeMap[jm]) judgeMap[jm] = { triggered: false, evidences: [] };
        judgeMap[jm].triggered = true;
        if (td.evidence && td.evidence !== "none") judgeMap[jm].evidences.push(td.evidence);
      });
    });

    const rows = Object.entries(judgeMap).map(([jm, info]) => {
      const resultCls  = info.triggered ? "judge-row-triggered" : "judge-row-clean";
      const resultText = info.triggered ? "TRIGGERED" : "clean";
      const evidence   = info.evidences.length
        ? `<span class="evidence-cell">${esc(info.evidences[0].slice(0, 200))}</span>`
        : `<span class="judge-row-clean">—</span>`;
      return `<tr>
        <td class="name">${esc(jm.split("/").pop())}</td>
        <td class="${resultCls}">${resultText}</td>
        <td>${evidence}</td>
      </tr>`;
    }).join("");

    html += `
<div class="detail-section">
  <h2>per-judge results</h2>
  <table class="detail-table">
    <colgroup><col class="col-name"><col style="width:90px"><col class="col-name"></colgroup>
    <thead><tr><th>judge</th><th>result</th><th>evidence</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</div>`;
  }

  root.innerHTML = html;
}

// ------------------------------------------------------------------ //
// Valid eval rendering (scores + metrics + per-judge)                  //
// ------------------------------------------------------------------ //

function renderScores(scores) {
  const id        = scores.eval_id || evalId || "—";
  const cHotkey   = scores.challenger_hotkey || hotkey || "—";
  const kHotkey   = scores.king_hotkey || "—";
  const date      = scores.date || "—";
  const accepted  = scores.accepted ?? false;
  const errCode   = scores.error_code || "";
  const errDetail = scores.error_detail || scores.error || "";
  const chalScore = scores.challenger_score;
  const kScore    = scores.king_score;
  const byMetric  = scores.by_metric || {};
  const byJudge   = scores.by_judge  || {};
  const nDone     = scores.n_done  ?? "—";
  const nValid    = scores.n_valid ?? "—";
  const winner    = scores.winner  || "—";

  let html = `
<div class="detail-meta">
  <h1>${esc(id)}</h1>
  <div class="sub">
    <span>challenger: <code>${esc(cHotkey)}</code></span>
    <span class="sep">·</span>
    <span>vs. king: <code>${esc(kHotkey)}</code></span>
    <span class="sep">·</span>
    <span>date: ${esc(date)}</span>
    <span class="sep">·</span>
    ${badge(accepted, errCode)}
  </div>
</div>`;

  // Error / invalid panel
  if (errCode || errDetail) {
    const panelCls   = errCode === "chal_injection_detected" ? "injection"
                     : errCode === "duplicate_model"          ? "duplicate"
                     : errCode.includes("infra")              ? "infra"
                     : "invalid";
    const panelLabel = errCode === "chal_injection_detected" ? "Injection detected"
                     : errCode === "duplicate_model"          ? "Duplicate model"
                     : errCode || "Error";
    html += `
<div class="fail-panel ${panelCls}">
  <strong>${esc(panelLabel)}</strong>
  <span class="detail-text">${esc(errDetail)}</span>
</div>`;
  }

  // Overall scores
  if (chalScore != null) {
    const cCls = chalScore > 50 ? "win" : chalScore < 50 ? "lose" : "";
    const kCls = kScore   > 50 ? "win" : kScore   < 50 ? "lose" : "";
    const lcb = scores.lcb;
    const lcbCard = lcb != null
      ? `<div class="score-card"><div class="label">lcb</div><div class="value ${scores.gate_lcb ? "win" : "lose"}" title="lower confidence bound of the win margin${scores.gate_alpha != null ? ` · gate α=${scores.gate_alpha}` : ""} · ${scores.gate_lcb ? "passed" : "failed"}">${Number(lcb).toFixed(4)}</div></div>`
      : "";
    html += `
<div class="detail-section">
  <h2>overall scores</h2>
  <div class="score-grid">
    <div class="score-card"><div class="label">challenger</div><div class="value ${cCls}">${Number(chalScore).toFixed(2)}</div></div>
    <div class="score-card"><div class="label">king</div><div class="value ${kCls}">${Number(kScore).toFixed(2)}</div></div>
    <div class="score-card"><div class="label">winner</div><div class="value dim">${esc(winner)}</div></div>
    <div class="score-card"><div class="label">turns</div><div class="value dim">${nValid} / ${nDone}</div></div>
    ${lcbCard}
  </div>
</div>`;
  }

  // Per-metric
  const metricKeys = Object.keys(byMetric);
  if (metricKeys.length) {
    const rows = metricKeys.map(m => {
      const c = Number(byMetric[m]);
      const k = 100 - c;
      const cCls = c > 50 ? "win" : c < 50 ? "lose" : "";
      const kCls = k > 50 ? "win" : k < 50 ? "lose" : "";
      return `<tr>
        <td class="name">${esc(m)}</td>
        <td class="num ${cCls}">${c.toFixed(2)}</td>
        <td class="num ${kCls}">${k.toFixed(2)}</td>
      </tr>`;
    }).join("");
    html += `
<div class="detail-section">
  <h2>by metric</h2>
  <table class="detail-table">
    <colgroup><col class="col-name"><col class="col-val"><col class="col-val"></colgroup>
    <thead><tr><th>metric</th><th class="r">challenger</th><th class="r">king</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</div>`;
  }

  // Per-judge
  const judgeModels = Object.keys(byJudge);
  if (judgeModels.length) {
    const rows = judgeModels.map(m => {
      const c = Number(byJudge[m]);
      const k = 100 - c;
      const cCls = c > 50 ? "win" : c < 50 ? "lose" : "";
      const kCls = k > 50 ? "win" : k < 50 ? "lose" : "";
      return `<tr>
        <td class="name">${esc(m.split("/").pop())}</td>
        <td class="num ${cCls}">${c.toFixed(2)}</td>
        <td class="num ${kCls}">${k.toFixed(2)}</td>
      </tr>`;
    }).join("");
    html += `
<div class="detail-section">
  <h2>by judge</h2>
  <table class="detail-table">
    <colgroup><col class="col-name"><col class="col-val"><col class="col-val"></colgroup>
    <thead><tr><th>judge</th><th class="r">challenger</th><th class="r">king</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</div>`;
  }

  root.innerHTML = html;
}

// ------------------------------------------------------------------ //
// Load                                                                 //
// ------------------------------------------------------------------ //

async function load() {
  if (dirUrl) {
    // Valid eval with artifact files
    let scores = null;
    try {
      scores = JSON.parse(await fetchText(`${dirUrl}/scores.json`));
    } catch (e) {
      root.innerHTML = `<p class="error-msg">Could not load scores.json: ${esc(e.message)}</p>`;
      return;
    }
    renderScores(scores);
    return;
  }

  // No dir_url — fetch dashboard.json and find entry by eval_id
  if (!evalId) {
    renderFailure({ error_code: errorCode, error_detail: errorDetail, hotkey });
    return;
  }
  try {
    const dashboard = JSON.parse(await fetchText("/dashboard.json"));
    const entry = (dashboard.history || []).find(h => h.eval_id === evalId);
    if (entry) {
      if (entry.type === "failure") {
        renderFailure(entry);
      } else {
        renderScores({ ...entry, ...(entry.verdict || {}) });
      }
    } else {
      renderFailure({ eval_id: evalId, hotkey, error_code: errorCode, error_detail: errorDetail });
    }
  } catch {
    renderFailure({ eval_id: evalId, hotkey, error_code: errorCode, error_detail: errorDetail });
  }
}

load().catch(err => {
  root.innerHTML = `<p class="error-msg">Failed to load eval detail: ${esc(String(err))}</p>`;
});
