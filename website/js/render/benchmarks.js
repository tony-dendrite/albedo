import { el, mount } from "../dom.js";
import { pct, fmtRelative } from "../format.js";
import { modelRepo, kingTitleName } from "../model.js";
import { PULLED_SUITES, PREDS_STALE_MS } from "../config.js";
import { benchmarkRegistry, mergeDistributedResults, distributedRunFor, distributedProgress } from "../results.js";

const MODEL_SCORE_SUITE = "model_score";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  swe_rebench_2026_03: "SWE-rebench",
  model_score: "SWE-bench Verified"
};

// const BENCHMARK_ORDER = ["tau2_airline", "tau2_retail", "tau2_telecom", "swe_rebench_2026_03", MODEL_SCORE_SUITE];
let BENCHMARK_ORDER = ["swe_rebench_2026_03", MODEL_SCORE_SUITE];

function applyBenchmarkRegistry(manifest) {
  const registry = benchmarkRegistry(manifest);
  BENCHMARK_ORDER = registry.map(entry => entry.suite);
  for (const entry of registry) BENCHMARK_LABELS[entry.suite] = entry.niceName;
  if (!BENCHMARK_ORDER.includes(benchSort)) benchSort = BENCHMARK_ORDER[0];
}

const ACTIVE_STATES = new Set(["QUEUED", "CLAIMED", "LOADING_MODEL", "RUNNING", "SCORING"]);
const LEADERBOARD_ROWS = 5;
const PAGE_SIZES = [5, 10, 25, 50];
// genesis is re-uploaded under the king-genesis repo, like every other king
const GENESIS_REPO = "dendriteholdings/albedo-qwen3.6-35b-king-genesis";

// leaderboard view: "top" = top-5 sortable by benchmark (default), "all" = full king history
let benchMode = localStorage.getItem("benchLeaderMode") || "top";
if (!["top", "all"].includes(benchMode)) benchMode = "top";
// which benchmark the top-5 is sorted by (descending only)
let benchSort = localStorage.getItem("benchLeaderboardSort") || BENCHMARK_ORDER[0];
if (!BENCHMARK_ORDER.includes(benchSort)) benchSort = BENCHMARK_ORDER[0];
let historyPage = Math.max(1, Number(localStorage.getItem("benchPanelHistoryPage")) || 1);
let historyPageSize = Number(localStorage.getItem("benchPanelHistoryPageSize")) || 10;
if (!PAGE_SIZES.includes(historyPageSize)) historyPageSize = 10;

function benchmarkLabel(suite) {
  return BENCHMARK_LABELS[suite] || suite || "—";
}

function modelName(model) {
  if (isGenesis(model)) return GENESIS_REPO;
  return model?.model_repo || modelRepo(model?.model_uri) || model?.model_uri || "—";
}

const ROMAN_VALUES = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };

function romanToInt(value) {
  let total = 0;
  let previous = 0;
  for (const char of value.toUpperCase().split("").reverse()) {
    const current = ROMAN_VALUES[char] || 0;
    total += current < previous ? -current : current;
    previous = Math.max(previous, current);
  }
  return total;
}

// benchmarks.json labels ("King <N>") map 1:1 to chain reigns — each king-<N> repo's
// albedo.md names the hippius repo/hotkey of chain king N. Display the reign name
// (ALBEDO-<roman>) used everywhere else on the site.
function modelLabel(model) {
  const label = model?.label || "—";
  if (/^genesis$/i.test(label)) return kingTitleName(0);
  const match = /^King\s+([IVXLCDM]+)$/i.exec(label);
  if (!match) return label;
  return kingTitleName(romanToInt(match[1]));
}

function modelKingNumber(model) {
  const labelMatch = /^King\s+([IVXLCDM]+)$/i.exec(model?.label || "");
  const repoMatch = /-king-([IVXLCDM]+)$/i.exec(model?.model_repo || "");
  const numeral = labelMatch?.[1] || repoMatch?.[1];
  return numeral ? romanToInt(numeral) : null;
}

function hfRepoUrl(model) {
  if (isGenesis(model)) return `https://huggingface.co/${GENESIS_REPO}`;
  return model?.model_repo ? `https://huggingface.co/${model.model_repo}` : null;
}

function completedRuns(model) {
  return (model?.runs || []).filter(run => run.score != null || Number(run.task_count || 0) > 0 || run.finished_at);
}

function progressKey(modelRepo, suite) {
  return `${modelRepo || ""}\n${suite || ""}`;
}

function activeState(item) {
  return String(item?.phase || item?.state || "").toUpperCase();
}

function isActiveProgress(item) {
  return ACTIVE_STATES.has(activeState(item));
}

function activeProgressByModelSuite(data) {
  const out = new Map();
  for (const source of [...(data?.jobs || []), ...(data?.workers || [])]) {
    if (!source?.model_repo || !source?.suite || !isActiveProgress(source)) continue;
    out.set(progressKey(source.model_repo, source.suite), source);
  }
  return out;
}

function hasActiveProgress(model, activeProgress) {
  return BENCHMARK_ORDER.some(suite => activeProgress.has(progressKey(model?.model_repo || model?.id, suite)));
}

function latestRun(model) {
  return completedRuns(model).sort((a, b) => {
    const at = new Date(a.finished_at || a.started_at || "").getTime();
    const bt = new Date(b.finished_at || b.started_at || "").getTime();
    if (Number.isFinite(bt - at) && bt !== at) return bt - at;
    return Number(b.run_attempt || 0) - Number(a.run_attempt || 0);
  })[0] || null;
}

function latestRunTime(model) {
  const run = latestRun(model);
  return run?.finished_at || run?.started_at || model?.activated_at || model?.discovered_at || "";
}

export function sortModels(models) {
  return [...(models || [])].sort((a, b) => {
    if (isGenesis(a) !== isGenesis(b)) return isGenesis(a) ? 1 : -1;
    const aKing = modelKingNumber(a);
    const bKing = modelKingNumber(b);
    if (aKing != null || bKing != null) {
      if (aKing == null) return 1;
      if (bKing == null) return -1;
      if (aKing !== bKing) return bKing - aKing;
    }
    const orderDelta = Number(a.model_order ?? 999999) - Number(b.model_order ?? 999999);
    if (orderDelta) return orderDelta;
    const timeDelta = new Date(latestRunTime(b)).getTime() - new Date(latestRunTime(a)).getTime();
    return Number.isFinite(timeDelta) ? timeDelta : 0;
  });
}

function isGenesis(model) {
  const identity = `${model?.label || ""} ${model?.model_repo || ""}`.toLowerCase();
  return identity.includes("genesis") || identity.includes("qwen/qwen3.6-35b-a3b");
}

// Reign a model stands for, on the scale the pulled score files use: genesis is 0.
function reignNumber(model) {
  return isGenesis(model) ? 0 : modelKingNumber(model);
}

// A pulled run_id is the reign name: "king-genesis" or "king-<roman>".
export function pulledRunId(model) {
  if (!model) return null;
  if (isGenesis(model)) return "king-genesis";
  const numeral = modelLabel(model).split("-").pop();
  return /^[IVXLCDM]+$/.test(numeral) ? `king-${numeral}` : null;
}

function runIdReign(runId) {
  if (/^king-genesis$/i.test(runId)) return 0;
  const numeral = /^king-([IVXLCDM]+)$/i.exec(runId)?.[1];
  return numeral ? romanToInt(numeral) : null;   // ignores rows for non-king models
}

// The research bucket also holds shadow runs from before the service took a suite
// over, so a score row only counts from that suite's first published reign.
function pulledApplies(pulled, number) {
  return number != null && number >= pulled.fromKing;
}

export function pulledRunFor(run) {
  return run?.pulled_key ? PULLED_SUITES.find(pulled => pulled.key === run.pulled_key) || null : null;
}

function pulledRun(pulled, row) {
  const score = Number(row?.score);
  if (!Number.isFinite(score)) return null;
  const id = `pulled:${pulled.key}:${row.run_id}`;
  return {
    id,
    run_id: id,
    suite: pulled.suite,
    score: score / 100,          // score files carry percent, runs carry a fraction
    state: "SUCCEEDED",
    task_count: row.total,
    passed_count: row.resolved,
    score_meta: `${row.resolved ?? "—"}/${row.total ?? "—"} resolved`,
    pulled_key: pulled.key,
    pulled_run_id: row.run_id,
    pulled_model: row.model,
  };
}

// Fold the pulled score files into the benchmarks.json runs, per suite, so the rest of
// the panel never has to care which source a score came from.
export function mergePulledScores(data, scoresBySuite) {
  if (!scoresBySuite?.size) return data;
  const byReign = new Map();
  for (const pulled of PULLED_SUITES) {
    for (const row of scoresBySuite.get(pulled.suite) || []) {
      if (!row?.run_id) continue;
      const number = runIdReign(String(row.run_id));
      if (!pulledApplies(pulled, number)) continue;
      const run = pulledRun(pulled, row);
      if (!run) continue;
      const bucket = byReign.get(number) || { runId: String(row.run_id), row, runs: [] };
      bucket.runs.push(run);
      byReign.set(number, bucket);
    }
  }
  if (!byReign.size) return data;

  const merged = new Set();
  const models = (data?.models || []).map(model => {
    const number = reignNumber(model);
    const bucket = number == null ? null : byReign.get(number);
    if (!bucket) return model;
    merged.add(number);
    const replaced = new Set(bucket.runs.map(run => run.suite));
    return { ...model, runs: [...(model.runs || []).filter(run => !replaced.has(run.suite)), ...bucket.runs] };
  });

  // A reign the benchmarking service scored but never registered as a model would
  // otherwise be missing from the site entirely, so stand one up from the score row.
  for (const [number, bucket] of byReign) {
    if (number === 0 || merged.has(number)) continue;
    models.push({
      id: `pulled:${bucket.runId}`,
      label: `King ${bucket.runId.replace(/^king-/i, "").toUpperCase()}`,
      model_repo: String(bucket.row.model || "").replace(/^hosted_vllm\//, ""),
      runs: bucket.runs,
    });
  }
  return { ...data, models };
}

function panelModels(data, liveRunIds = new Set()) {
  const activeProgress = activeProgressByModelSuite(data);
  // A pulled suite queues no job here, so a reign whose only sign of life is its
  // predictions file still has to reach the panel, or its progress never shows.
  const active = model => hasActiveProgress(model, activeProgress) || liveRunIds.has(pulledRunId(model));
  const models = (data?.models || []).filter(model => completedRuns(model).length || active(model));
  const sorted = sortModels(models).filter(model => hasPanelScores(model) || active(model));
  return { models, sorted, selected: sorted.find(model => !isGenesis(model)) || sorted[0] || null };
}

// Reigns whose pulled score has not landed yet: their preds file is what the tile
// shows progress from, one candidate list per suite.
export function liveScoreCandidates(data, scoresBySuite) {
  const sorted = sortModels(mergePulledScores(data, scoresBySuite)?.models || []);
  return new Map(PULLED_SUITES.map(pulled => [
    pulled.suite,
    sorted
      .filter(model => !isGenesis(model)
        && pulledApplies(pulled, reignNumber(model))
        && suiteScores(model)[pulled.suite]?.score == null)
      .map(pulledRunId)
      .filter(Boolean)
      .slice(0, 6),
  ]));
}

function scoreTotal(rows, fallback) {
  const totals = (rows || []).map(row => Number(row?.total)).filter(n => Number.isFinite(n) && n > 0);
  return totals.length ? Math.max(...totals) : fallback;
}

function livePreds(live, rows, pulled) {
  if (!live?.count) return null;
  const total = scoreTotal(rows, pulled.totalFallback);
  const updated = live.updatedAt ? new Date(live.updatedAt).getTime() : NaN;
  const ratio = Math.min(1, live.count / total);
  const fresh = Number.isFinite(updated) ? Date.now() - updated < PREDS_STALE_MS : true;
  return {
    count: live.count,
    total,
    ratio,
    fresh,
    scoring: !fresh && ratio >= 0.95,
    updatedAt: live.updatedAt,
  };
}

function detailHref(model, runId = null) {
  const qs = new URLSearchParams();
  if (model?.id) qs.set("model_id", model.id);
  if (runId) qs.set("run_id", runId);
  return `./benchmark.html?${qs.toString()}`;
}

function runHref(model, entry) {
  return entry?.run_id && !entry.no_detail ? detailHref(model, entry.run_id) : detailHref(model);
}

function runTime(run) {
  return new Date(run?.finished_at || run?.started_at || "").getTime() || 0;
}

export function suiteScores(model) {
  const scores = { ...(model?.latest_scores || {}) };
  const passes = {};
  for (const run of model?.runs || []) {
    if (!run?.suite || run.score == null) continue;
    (passes[run.suite] ||= []).push(run);
  }
  for (const [suite, runs] of Object.entries(passes)) {
    const latest = runs.reduce((a, b) => runTime(b) > runTime(a) ? b : a);
    scores[suite] = {
      ...latest,
      score: runs.reduce((sum, run) => sum + Number(run.score), 0) / runs.length,
      pass_count: runs.length,
      run_id: latest.run_id || latest.id,
    };
  }
  return scores;
}

function hasPanelScores(model) {
  const scores = suiteScores(model);
  return BENCHMARK_ORDER.some(suite => scores[suite]?.score != null)
    || (model?.runs || []).some(run => run?.source === "distributed" && BENCHMARK_ORDER.includes(run.suite));
}

function panelScore(value) {
  return `${pct(value, 1)}%`;
}

function baselineComparison(entry, baseline) {
  if (baseline?.score == null) return { label: "genesis —", delta: "—", cls: "" };
  if (entry?.score == null) return { label: `genesis ${panelScore(baseline.score)}`, delta: "—", cls: "" };
  const delta = (Number(entry.score) - Number(baseline.score)) * 100;
  return {
    label: `genesis ${panelScore(baseline.score)}`,
    delta: `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`,
    cls: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
  };
}

function previousComparison(entry, sorted, selected, suite) {
  if (entry?.score == null) return { delta: "—", cls: "flat" };
  const previous = sorted
    .slice(sorted.indexOf(selected) + 1)
    .find(model => !isGenesis(model) && suiteScores(model)[suite]?.score != null);
  if (!previous) return { delta: "—", cls: "flat" };
  const delta = (Number(entry.score) - Number(suiteScores(previous)[suite].score)) * 100;
  return {
    delta: `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`,
    cls: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
  };
}

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

const SPARK_KINGS = 20;

function renderSpark(sorted, suite, baselineScore = null, width = 360) {
  // a fixed window of the last SPARK_KINGS kings: a new king adds a slot even before it has a
  // score, so the line shifts left as reigns change instead of stretching over all history
  const slots = sorted.filter(model => !isGenesis(model)).slice(0, SPARK_KINGS).reverse();
  const points = slots
    .map((model, slot) => {
      const entry = suiteScores(model)[suite];
      return { slot, label: modelLabel(model), score: entry?.score, href: runHref(model, entry) };
    })
    .filter(point => point.score != null);
  const FLOOR = 54, TOP = 8;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} 64`, preserveAspectRatio: "xMidYMid", role: "img" });

  // line below graph
  svg.append(svgEl("line", { x1: 6, y1: FLOOR, x2: width - 6, y2: FLOOR, stroke: "currentColor", "stroke-width": 1, opacity: 0.15 }));

  if (!points.length) {
    svg.append(svgEl("text", { x: width / 2, y: 34, "text-anchor": "middle", "font-size": 8, fill: "currentColor", opacity: 0.45 }, "no score"));
    return svg;
  }
  const vals = points.map(p => p.score);
  const scaleVals = baselineScore != null ? [...vals, baselineScore] : vals;
  const min = 0;
  const max = (Math.max(...scaleVals) || 0.01) * 1.12;
  const yOf = v => FLOOR - ((v - min) / (max - min)) * (FLOOR - TOP);
  const xOf = slot => 6 + (slot / (SPARK_KINGS - 1)) * (width - 12);
  const coords = points.map(point => ({ x: xOf(point.slot), y: yOf(point.score), point }));

  // soft area fill under the trend
  if (coords.length > 1) {
    const d = `M ${coords[0].x.toFixed(1)},${FLOOR} `
      + coords.map(c => `L ${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ")
      + ` L ${coords[coords.length - 1].x.toFixed(1)},${FLOOR} Z`;
    svg.append(svgEl("path", { d, fill: "currentColor", opacity: 0.08 }));
  }
  // genesis baseline reference (dashed gold) — points above it beat genesis
  if (baselineScore != null) {
    const by = yOf(baselineScore);
    svg.append(svgEl("line", {
      x1: 6, y1: by.toFixed(1), x2: width - 6, y2: by.toFixed(1),
      stroke: "var(--color-gold)", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.55,
    }, svgEl("title", {}, `genesis ${panelScore(baselineScore)}`)));
  }
  if (coords.length > 1) {
    svg.append(svgEl("polyline", {
      points: coords.map(c => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" "),
      fill: "none", stroke: "currentColor", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }
  const bestIdx = vals.indexOf(Math.max(...vals));   // best model overall — highlighted gold
  coords.forEach((c, i) => {
    const last = i === coords.length - 1;
    const best = i === bestIdx;
    svg.append(svgEl("circle", {
      cx: c.x.toFixed(1), cy: c.y.toFixed(1), r: best ? 3.4 : last ? 3 : 2.2,
      fill: best ? "var(--color-gold)" : "currentColor",
      opacity: best || last ? 1 : 0.4,
    }));
  });
  return withSparkHover(svg, coords, width, bestIdx, { top: TOP, floor: FLOOR });
}

function withSparkHover(svg, coords, width, bestIdx, { top, floor }) {
  const guide = svgEl("line", {
    y1: top - 4, y2: floor, stroke: "currentColor", "stroke-width": 1, opacity: 0.35,
    visibility: "hidden", "pointer-events": "none",
  });
  svg.append(guide);
  const tip = el("div", { class: "spark-tip", hidden: true });
  const wrap = el("div", { class: "spark" }, svg, tip);
  const hide = () => { guide.setAttribute("visibility", "hidden"); tip.hidden = true; };
  const nearest = event => {
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return null;
    const x = (event.clientX - rect.left) * (width / rect.width);
    let index = 0;
    coords.forEach((c, i) => { if (Math.abs(c.x - x) < Math.abs(coords[index].x - x)) index = i; });
    return { index, rect };
  };
  svg.addEventListener("click", event => {
    const hit = nearest(event);
    if (hit) location.href = coords[hit.index].point.href;
  });
  svg.addEventListener("pointermove", event => {
    const hit = nearest(event);
    if (!hit) return;
    const { index, rect } = hit;
    const c = coords[index];
    guide.setAttribute("x1", c.x.toFixed(1));
    guide.setAttribute("x2", c.x.toFixed(1));
    guide.setAttribute("visibility", "visible");
    tip.replaceChildren(
      el("b", { class: index === bestIdx ? "best" : "" }, panelScore(c.point.score)),
      el("span", {}, c.point.label));
    tip.hidden = false;
    const left = (c.x / width) * rect.width;
    tip.style.left = `${left}px`;
    tip.dataset.side = left > rect.width / 2 ? "left" : "right";
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("pointercancel", hide);
  return wrap;
}

function progressLabel(preds) {
  if (preds.distributed) return preds.status;
  if (preds.fresh) return "running";
  return preds.scoring ? "scoring" : "stalled";
}

function renderProgress(preds, label) {
  const percent = (preds.ratio * 100).toFixed(1);
  const state = preds.distributed
    ? [`${percent}%`, preds.status, `${preds.completed} completed`, preds.errored ? `${preds.errored} errored` : null]
    : preds.fresh
      ? [`${percent}%`, "generating"]
    : preds.scoring
      ? [`${percent}%`, "awaiting score"]
      : [`${percent}%`, `idle ${fmtRelative(preds.updatedAt)}`];
  return el("div", { class: "bench-tile-progress" },
    el("div", { class: preds.fresh ? "bench-tile-progress-bar live" : "bench-tile-progress-bar" },
      el("i", { style: `width:${percent}%` })),
    el("div", { class: "bench-tile-progress-note" }, [label, ...state].filter(Boolean).join(" · ")));
}

function renderTile(model, suite, sorted, baseline, activity, preds) {
  const entry = suiteScores(model)[suite];
  const distributed = distributedRunFor(model, suite);
  const scored = entry?.score != null;
  // A running distributed benchmark can publish a meaningful partial score. Keep its
  // progress visible—and use the running theme—until the producer marks it complete.
  const distributedLive = distributed && !["complete", "failed"].includes(distributed.distributed_status);
  const progress = distributedLive ? distributedProgress(distributed) : (scored ? null : preds);
  const genesis = baselineComparison(entry, baseline);
  const previous = previousComparison(entry, sorted, model, suite);
  const running = scored || distributed ? null : activity?.running;
  const queued = scored || distributed ? [] : (activity?.queued || []);
  const selectedRun = entry || distributed;
  const href = selectedRun?.run_id && !selectedRun.no_detail ? detailHref(model, selectedRun.run_id) : null;
  const runNote = running
    ? [runningLabel(running, activity.labelByRepo), progressNote(running)].filter(Boolean).join(" · ")
    : queued.length ? `${queued.length} pending` : "";
  const live = Boolean(running) || Boolean(progress?.fresh);

  const chartSvgElement = el("div", { class: "bench-tile-chart" }, renderSpark(sorted, suite, baseline?.score));
  let chartWidth = 0;
  const chartObserver = new ResizeObserver(entries => {
    const w = Math.round(entries[0].contentRect.width);
    if (!w || w === chartWidth) return;
    chartWidth = w;
    chartSvgElement.replaceChildren(renderSpark(sorted, suite, baseline?.score, w));
  });
  chartObserver.observe(chartSvgElement);

  return el("article", {
    class: "bench-tile",
    "data-status": progress ? "progress" : scored ? "completed" : "missing",
    "data-activity": live ? "running" : "idle",
  },
    el("div", { class: "bench-tile-head" },
      el("div", { class: "bench-tile-name" }, benchmarkLabel(suite)),
      el("span", { class: live ? "bench-tile-activity live" : "bench-tile-activity" }, live ? "running" : "idle")),
    el("div", { class: "bench-tile-main" },
      el("div", { class: "bench-tile-score-wrap" },
        el(href ? "a" : "span", { class: "bench-tile-score", href },
          scored ? `${entry.partial_score ? "partial " : ""}${panelScore(entry.score)}` : progress ? progressLabel(progress) : "missing"),
        scored
          ? el("span", { class: "bench-tile-pass-count" },
              entry.score_meta || `avg · ${entry.pass_count || 1} ${entry.pass_count === 1 ? "pass" : "passes"}`)
          : progress
            ? el("span", { class: "bench-tile-pass-count" }, `${progress.count} / ${progress.total} predictions`)
            : null),
      el("div", { class: `bench-tile-change ${previous.cls}` },
        el("strong", {}, previous.delta),
        el("span", {}, "since last"))),
    chartSvgElement,
    el("div", { class: "bench-tile-status" },
      el("span", {}, genesis.label),
      el("span", { class: `bench-delta ${genesis.cls}`, title: "delta vs genesis" }, genesis.delta)),
    progress ? renderProgress(progress, progress.kingLabel || modelLabel(model)) : runNote ? el("div", { class: "bench-tile-run-note" }, runNote) : null);
}

function runningLabel(item, labelByRepo) {
  if (labelByRepo?.has(item?.model_repo)) return labelByRepo.get(item.model_repo);
  if (item?.label) return modelLabel({ label: item.label });
  return (item?.model_repo || "").split("/").pop() || "—";
}

function progressNote(item) {
  const done = Number(item?.progress_done);
  const total = Number(item?.progress_total);
  if (Number.isFinite(done) && Number.isFinite(total) && total > 0) return `${done}/${total}`;
  return null;
}

function suiteActivity(data) {
  const models = data?.models || [];
  const labelByRepo = new Map(models.filter(m => m.model_repo).map(m => [m.model_repo, modelLabel(m)]));
  const orderByRepo = new Map(models.filter(m => m.model_repo).map(m => [m.model_repo, Number(m.model_order ?? 999999)]));

  const runningBySuite = new Map();
  for (const worker of data?.workers || []) {
    if (worker?.suite && worker?.model_repo && isActiveProgress(worker)) runningBySuite.set(worker.suite, worker);
  }
  const queuedBySuite = new Map(BENCHMARK_ORDER.map(suite => [suite, []]));
  for (const job of data?.jobs || []) {
    if (!queuedBySuite.has(job?.suite) || !isActiveProgress(job)) continue;
    if (activeState(job) === "QUEUED") queuedBySuite.get(job.suite).push(job);
    else if (!runningBySuite.has(job.suite)) runningBySuite.set(job.suite, job);
  }
  return new Map(BENCHMARK_ORDER.map(suite => {
    const queued = [...(queuedBySuite.get(suite) || [])].sort((a, b) =>
      (orderByRepo.get(a.model_repo) ?? 999999) - (orderByRepo.get(b.model_repo) ?? 999999));
    return [suite, { running: runningBySuite.get(suite), queued, labelByRepo }];
  }));
}

function benchScoreOf(model, suite) {
  const s = suiteScores(model)[suite]?.score;
  return Number.isFinite(s) ? s : null;
}

// leaderboard cells: the king opens its Hugging Face model, a score opens that king's run
function kingLink(model) {
  const repoUrl = hfRepoUrl(model);
  return repoUrl
    ? el("a", { href: repoUrl, target: "_blank", rel: "noopener", title: modelName(model) }, modelLabel(model))
    : el("span", { title: modelName(model) }, modelLabel(model));
}

function runCell(model, suite, entry, extraClass = "", note = "") {
  const passes = entry.score_meta || `${entry.pass_count || 1} pass average`;
  const title = [`${modelLabel(model)} on ${benchmarkLabel(suite)}`, note, passes, "open run and trajectories"]
    .filter(Boolean).join(" · ");
  return el("td", { class: `r bench-run-cell${extraClass}` },
    el("a", { href: runHref(model, entry), title }, panelScore(entry.score)));
}

// each benchmark's best score among the kings (genesis is the reference, not a king)
function bestKingScores(sorted) {
  return Object.fromEntries(BENCHMARK_ORDER.map(suite => [suite, Math.max(
    ...sorted.filter(model => !isGenesis(model)).map(model => benchScoreOf(model, suite) ?? -Infinity))]));
}

function scoreCell(model, suite, entry, best) {
  const top = !isGenesis(model) && entry.score === best[suite];
  return runCell(model, suite, entry, top ? " bench-best" : "", top ? "best king score" : "");
}

function renderLeaderboard(sorted, selectedModel, baselineScores, rerender) {
  const best = bestKingScores(sorted);
  const setSort = suite => {
    benchSort = suite;
    localStorage.setItem("benchLeaderboardSort", suite);
    rerender();
  };
  // top N by the active benchmark, descending; genesis ranks by its own score
  const order = [...sorted]
    .sort((a, b) => (benchScoreOf(b, benchSort) ?? -Infinity) - (benchScoreOf(a, benchSort) ?? -Infinity));
  const ranked = order.slice(0, LEADERBOARD_ROWS);
  // genesis is the reference point, so it stays visible below the top rows with its real rank
  const genesisRank = order.findIndex(isGenesis);
  const shown = ranked.map((model, i) => [model, i + 1]);
  if (genesisRank >= LEADERBOARD_ROWS) shown.push([order[genesisRank], genesisRank + 1]);

  const headCell = suite => el("th", {
    class: `r bench-sort-th${suite === benchSort ? " active" : ""}`,
    onClick: () => setSort(suite),
    title: `sort by ${benchmarkLabel(suite)} (descending)`,
  }, benchmarkLabel(suite));

  const rows = shown.flatMap(([model, rank], i) => [
    // ranks skipped between the top rows and genesis read as a gap, not as consecutive places
    i > 0 && rank - shown[i - 1][1] > 1
      ? el("tr", { class: "bench-rank-gap" }, el("td", { colspan: 2 + BENCHMARK_ORDER.length }, "⋯"))
      : null,
    leaderboardRow(model, rank),
  ]).filter(Boolean);

  function leaderboardRow(model, rank) {
    const scores = suiteScores(model);
    const genesis = isGenesis(model);
    return el("tr", { class: genesis ? "bench-genesis-row" : "" },
      el("td", { class: "bench-rank" }, String(rank)),
      el("td", { class: "bench-king-col" },
        kingLink(model),
        genesis ? el("span", { class: "bench-baseline-tag" }, "baseline") : null),
      BENCHMARK_ORDER.map(suite => {
        const entry = scores[suite];
        if (entry?.score == null) return el("td", { class: "r" }, el("span", { class: "muted-dash" }, "—"));
        return scoreCell(model, suite, entry, best);
      }));
  }

  return el("div", { class: "bench-history" },
    el("div", { class: "bench-leaderboard-cap" },
      el("span", {}, `top ${ranked.length} · by ${benchmarkLabel(benchSort)}`),
      el("span", { class: "bench-leaderboard-hint" }, "click a benchmark to sort · yellow is the best king score")),
    sorted.length
      ? el("div", { class: "data-table-wrap" },
          el("table", { class: "data-table bench-leaderboard" },
            el("thead", {}, el("tr", {},
              el("th", { class: "bench-rank" }, "#"),
              el("th", {}, "king"),
              BENCHMARK_ORDER.map(headCell))),
            el("tbody", {}, rows)))
      : el("div", { class: "bench-history-empty" }, "no benchmark history yet"));
}

// "all" mode: the full king benchmark history, in reign order, paginated (the classic view).
function renderKingHistory(sorted, selectedModel, rerender) {
  const best = bestKingScores(sorted);
  const pages = Math.max(1, Math.ceil(sorted.length / historyPageSize));
  historyPage = Math.min(Math.max(1, historyPage), pages);
  const shown = sorted.slice((historyPage - 1) * historyPageSize, historyPage * historyPageSize);
  const setPage = page => {
    historyPage = page;
    localStorage.setItem("benchPanelHistoryPage", String(historyPage));
    rerender();
  };
  const pager = el("div", { class: "bench-history-pager" },
    el("div", { class: "bench-history-pager-left" },
      el("button", { type: "button", disabled: historyPage <= 1, onClick: () => setPage(historyPage - 1) }, "prev"),
      el("span", {}, `page ${historyPage} / ${pages} · ${sorted.length} kings`),
      el("button", { type: "button", disabled: historyPage >= pages, onClick: () => setPage(historyPage + 1) }, "next")),
    el("span", { class: "bench-leaderboard-hint" }, "yellow is the best king score"),
    el("label", { class: "bench-history-pager-right" }, "rows",
      el("select", { onChange: e => {
        historyPageSize = Number(e.target.value);
        localStorage.setItem("benchPanelHistoryPageSize", String(historyPageSize));
        setPage(1);
      } }, PAGE_SIZES.map(size => el("option", { value: size, selected: size === historyPageSize }, String(size))))));

  const rows = shown.map(model => {
    const scores = suiteScores(model);
    const reign = reignNumber(model);
    return el("tr", {},
      el("td", { class: "bench-rank" }, reign == null ? "—" : String(reign)),
      el("td", { class: "bench-king-col" }, kingLink(model)),
      BENCHMARK_ORDER.map(suite => {
        const entry = scores[suite];
        if (entry?.score == null) return el("td", { class: "r" }, el("span", { class: "muted-dash" }, "—"));
        return scoreCell(model, suite, entry, best);
      }));
  });

  return el("div", { class: "bench-history" },
    pager,
    sorted.length
      ? el("div", { class: "data-table-wrap" },
          el("table", { class: "data-table bench-leaderboard" },
            el("thead", {}, el("tr", {},
              el("th", { class: "bench-rank", title: "reign number (genesis is 0)" }, "#"),
              el("th", {}, "king"),
              BENCHMARK_ORDER.map(suite => el("th", { class: "r" }, benchmarkLabel(suite))))),
            el("tbody", {}, rows)))
      : el("div", { class: "bench-history-empty" }, "no benchmark history yet"));
}

export function renderBenchmarks(container, metaNode, data, scoresBySuite = null, liveBySuite = null, resultsManifest = null) {
  applyBenchmarkRegistry(resultsManifest);
  data = mergeDistributedResults(mergePulledScores(data, scoresBySuite), resultsManifest);
  const liveRunIds = new Set([...(liveBySuite?.values() || [])].map(live => live?.runId).filter(Boolean));
  const { models, sorted, selected } = panelModels(data, liveRunIds);
  if (!models.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark data yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  if (!sorted.length) {
    mount(container, el("div", { class: "empty" }, "no benchmark scores yet."));
    if (metaNode) metaNode.textContent = "no data";
    return;
  }
  const baselineScores = suiteScores((data?.models || []).find(isGenesis));
  const activity = suiteActivity(data);
  const predsBySuite = new Map(PULLED_SUITES.map(pulled => {
    const live = liveBySuite?.get(pulled.suite) || null;
    const preds = livePreds(live, scoresBySuite?.get(pulled.suite), pulled);
    // A tile can be showing progress for a reign other than the one on screen.
    if (preds && live?.runId !== pulledRunId(selected)) {
      const runningModel = (data?.models || []).find(model => pulledRunId(model) === live.runId);
      preds.kingLabel = runningModel ? modelLabel(runningModel) : String(live.runId || "").replace(/^king-/i, "King ");
    }
    return [pulled.suite, preds];
  }));
  const rerender = () => renderBenchmarks(container, metaNode, data, scoresBySuite, liveBySuite, resultsManifest);
  const scores = suiteScores(selected);
  const done = BENCHMARK_ORDER.filter(suite => scores[suite]?.score != null).length;

  mount(container,
    el("section", { class: "bench-panel" },
      el("div", { class: "bench-panel-head" },
        el("span", {}, "benchmark panel"),
        el("div", { class: "bench-panel-tools" },
          el("button", { class: "bench-history-toggle", type: "button", onClick: () => {
            benchMode = benchMode === "top" ? "all" : "top";
            localStorage.setItem("benchLeaderMode", benchMode);
            rerender();
          } }, benchMode === "top" ? "show all kings" : "top 5"),
          el("span", { class: "bench-panel-meta" },
            `${done}/${BENCHMARK_ORDER.length} scores · ${modelLabel(selected)}`))),
      el("div", { class: "bench-tile-grid" }, BENCHMARK_ORDER.map(suite =>
        renderTile(selected, suite, sorted, baselineScores[suite], activity.get(suite),
          predsBySuite.get(suite) || null))),
      benchMode === "all"
        ? renderKingHistory(sorted, selected, rerender)
        : renderLeaderboard(sorted, selected, baselineScores, rerender)));
  if (metaNode) metaNode.textContent = `${models.length} models · ${data.counts?.runs ?? 0} benchmark runs`;
}
