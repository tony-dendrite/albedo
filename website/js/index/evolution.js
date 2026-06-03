import { EVO_SCALE_MAX } from "./config.js";
import { fmtScore3 } from "./format.js";
import { judgeMeta, kingTitleName, modelLinkHtml } from "./model.js";
import { kingDateShort } from "./format.js";

let evoJudgeFilter = "all";
let evoRenderCtx = null;

export function kingBarScores(entry, index, chain, currentEval) {
  if (index === 0 && currentEval?.judges?.length) {
    return currentEval.judges.map(j => ({
      model:      j.model,
      score:      j.king_mean,
      chal_score: j.chal_mean,
      n:          j.n,
      live:       true,
    }));
  }
  if (entry.judges?.length) {
    return entry.judges.map(j => ({
      model:      j.model,
      score:      j.king_mean,
      chal_score: j.chal_mean,
      n:          j.n,
      live:       false,
    }));
  }
  const models = chain?.judge_models || [];
  return models.map(m => ({ model: m, score: null, chal_score: null, n: 0, live: false }));
}

function evoBarPx(score) {
  if (score == null) return 0;
  const root = getComputedStyle(document.documentElement);
  const barH = parseFloat(root.getPropertyValue("--evo-bar-h")) || 232;
  return Math.min(barH, Math.max(0, (Number(score) / EVO_SCALE_MAX) * barH));
}

function fmtBeatDelta(score, prev) {
  if (score == null || prev == null || prev <= 0) return null;
  const gain = Number(score) - Number(prev);
  if (gain <= 0) return null;
  return `+${((gain / Number(prev)) * 100).toFixed(1)}%`;
}

function evoBarLayers(score, prevScore) {
  const currPx = evoBarPx(score);
  const prevPx = prevScore == null ? 0 : evoBarPx(prevScore);
  const gainPx = prevScore == null ? 0 : Math.max(0, currPx - prevPx);
  const beat = gainPx > 0.5;
  const seed = prevScore == null;
  return { currPx, prevPx, gainPx, beat, seed };
}

function judgeColumns(kc, chain, currentEval) {
  const models = [];
  const seen = new Set();
  const add = m => { if (m && !seen.has(m)) { seen.add(m); models.push(m); } };
  (chain?.judge_models || []).forEach(add);
  if (currentEval?.judges) currentEval.judges.forEach(j => add(j.model));
  kc.forEach((e, i) => kingBarScores(e, i, chain, currentEval).forEach(s => add(s.model)));
  return models;
}

function renderEvolutionTower(s, prevScore, filter) {
  const meta = judgeMeta(s.model);
  const hidden = filter !== "all" && filter !== s.model ? " hidden" : "";
  const liveCls = s.live ? " live" : "";
  if (s.score == null) {
    return `<div class="evo-tower${hidden}" data-judge="${s.model}" title="${s.model}">
      <div class="evo-tower-metrics"><span class="evo-tower-val">—</span></div>
      <div class="evo-bar missing"><div class="evo-bar-baseline"></div></div>
      <span class="evo-tower-letter">${meta.letter}</span>
    </div>`;
  }
  const { currPx, prevPx, gainPx, beat, seed } = evoBarLayers(s.score, prevScore);
  const delta = seed ? null : fmtBeatDelta(s.score, prevScore);
  const deltaHtml = seed
    ? `<span class="evo-tower-delta na">—</span>`
    : (beat ? `<span class="evo-tower-delta beat">${delta}</span>` : "");
  const barCls = ["evo-bar", seed ? "is-seed" : "", beat ? "" : "no-gain"].filter(Boolean).join(" ");

  // Head-to-head labels: winner score on top (how this king won), prev-king score below
  // (how they defended). Each value is a short number that fits the 44px tower.
  const prevKingLabel = s.chal_score != null
    ? `<span class="evo-tower-val evo-tower-king-val">${fmtScore3(s.score)}</span>`
    : "";
  const mainVal = s.chal_score != null ? fmtScore3(s.chal_score) : fmtScore3(s.score);
  const tip = s.model
    + (s.n ? ` · n=${s.n}` : "")
    + (s.chal_score != null ? ` · winner ${fmtScore3(s.chal_score)} / prev-king ${fmtScore3(s.score)}` : "")
    + (beat && delta ? ` · beat prev by ${delta}` : "");
  const style = `--curr-h:${currPx}px;--prev-h:${prevPx}px;--gain-h:${gainPx}px`;
  return `<div class="evo-tower${hidden}${liveCls}" data-judge="${s.model}" title="${tip}">
    <div class="evo-tower-metrics">
      <span class="evo-tower-val">${mainVal}</span>
      ${prevKingLabel}
      ${deltaHtml}
    </div>
    <div class="${barCls}" style="${style}">
      <div class="evo-bar-baseline"></div>
      <div class="evo-bar-outline"></div>
      <div class="evo-bar-gain"></div>
      <div class="evo-bar-cap"></div>
    </div>
    <span class="evo-tower-letter">${meta.letter}</span>
  </div>`;
}

function renderEvolutionFilters(judges) {
  const filters = document.getElementById("evolution-filters");
  if (!judges.length) { filters.hidden = true; return; }
  filters.hidden = false;
  const allActive = evoJudgeFilter === "all" ? " active" : "";
  const btns = [`<button type="button" class="evo-filter${allActive}" data-filter="all">all</button>`];
  judges.forEach(m => {
    const meta = judgeMeta(m);
    const active = evoJudgeFilter === m ? " active" : "";
    btns.push(`<button type="button" class="evo-filter${active}" data-filter="${m}" title="${m}">
      <span class="evo-filter-key">${meta.letter}</span>${meta.label}
    </button>`);
  });
  filters.innerHTML = btns.join("");
  filters.querySelectorAll(".evo-filter").forEach(btn => {
    btn.onclick = () => {
      evoJudgeFilter = btn.dataset.filter;
      if (evoRenderCtx) renderEvolution(evoRenderCtx.kc, evoRenderCtx.chain, evoRenderCtx.currentEval);
    };
  });
}

export function renderEvolution(kc, chain, currentEval) {
  evoRenderCtx = { kc, chain, currentEval };
  const scroll = document.getElementById("evolution-scroll");
  if (!kc || kc.length === 0) {
    scroll.innerHTML = '<div class="empty">no kings yet.</div>';
    document.getElementById("evolution-filters").hidden = true;
    return;
  }
  const judges = judgeColumns(kc, chain, currentEval);
  const ordered = kc.slice().reverse();

  const groups = ordered.map((e, displayIdx) => {
    const dataIdx = kc.length - 1 - displayIdx;
    const scores = kingBarScores(kc[dataIdx], dataIdx, chain, currentEval);
    const byModel = Object.fromEntries(scores.map(s => [s.model, s]));
    let prevScores = {};
    if (displayIdx > 0) {
      const prevIdx = kc.length - 1 - (displayIdx - 1);
      const prevKingScores = kingBarScores(kc[prevIdx], prevIdx, chain, currentEval);
      // Compare how well each king WON (chal_score) vs how well the previous king won,
      // so the delta reflects improvement in winning margin, not in how badly each lost.
      prevScores = Object.fromEntries(
        prevKingScores.map(s => [s.model, s.chal_score ?? s.score])
      );
    }
    const towers = judges.map(m => {
      const s = byModel[m] || { model: m, score: null, chal_score: null, n: 0, live: false };
      // prevScore for the delta is previous king's chal_score (winning performance).
      return renderEvolutionTower(s, prevScores[m], evoJudgeFilter);
    }).join("");
    const dim = e.registered ? "" : " dim";
    const current = dataIdx === 0 ? " is-current" : "";
    const repo = e.model_repo || "";
    const digest = e.king_digest || e.model_digest || "";
    const name = modelLinkHtml(repo, digest, kingTitleName(e.reign_number));

    // Final score: mean chal and king across all judges for this king's crowning battle.
    const judgeEntries = e.judges || [];
    let finalHtml = "";
    if (judgeEntries.length) {
      const avgChal = judgeEntries.reduce((s, j) => s + (j.chal_mean || 0), 0) / judgeEntries.length;
      const avgKing = judgeEntries.reduce((s, j) => s + (j.king_mean || 0), 0) / judgeEntries.length;
      const cPct = (avgChal * 100).toFixed(1);
      const kPct = (avgKing * 100).toFixed(1);
      const winCls  = avgChal > 0.5 ? " win"  : avgChal < 0.5 ? " lose" : "";
      const loseCls = avgKing  > 0.5 ? " win"  : avgKing  < 0.5 ? " lose" : "";
      finalHtml = `<div class="evo-king-final">
        <span class="evo-final-chal${winCls}">${cPct}</span>
        <span class="evo-final-sep"> / </span>
        <span class="evo-final-king${loseCls}">${kPct}</span>
      </div>`;
    }

    return `<div class="evo-king${dim}${current}">
      <div class="evo-towers">${towers}</div>
      ${finalHtml}
      <div class="evo-king-name">${name}</div>
      <div class="evo-king-date">${kingDateShort(e.crowned_at)}</div>
    </div>`;
  }).join("");

  scroll.innerHTML = `<div class="evo-chart">${groups}</div>`;
  renderEvolutionFilters(judges);
  requestAnimationFrame(() => { scroll.scrollLeft = scroll.scrollWidth; });
}
