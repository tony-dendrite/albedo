import { fetchBenchmarkRun, fetchBenchmarks, fetchPulledScores, fetchJson } from "../fetch.js";
import { mergePulledScores, pulledRunFor } from "../render/benchmarks.js";
import { el, mount } from "../dom.js";
import { fmt, fmtDateTime, shortDigest } from "../format.js";
import { modelRepo, kingTitleName } from "../model.js";

const BENCHMARK_LABELS = {
  tau2_airline: "Tau2 Airline",
  tau2_retail: "Tau2 Retail",
  tau2_telecom: "Tau2 Telecom",
  tau2_banking_knowledge: "Tau2 Banking",
  swe_rebench_2026_03: "SWE-rebench",
  model_score: "SWE-bench Verified",
};
const TAU2_BENCH_VERSION = "τ²-bench 1.0.0";
const SWE_REBENCH_VERSION = "SWE-rebench 2026-03";
const SWE_VERIFIED_VERSION = "SWE-bench Verified";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const modelId = params.get("model_id");
const runId = params.get("run_id");

function benchmarkLabel(suite) {
  return BENCHMARK_LABELS[suite] || suite || "—";
}

function modelName(model) {
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

function hfRepoUrl(model) {
  return model?.model_repo ? `https://huggingface.co/${model.model_repo}` : null;
}

function completedRuns(model) {
  return (model?.runs || []).filter(run => run.score != null || Number(run.task_count || 0) > 0 || run.finished_at);
}

function latestRun(model) {
  return completedRuns(model).sort((a, b) => {
    const at = new Date(a.finished_at || a.started_at || "").getTime();
    const bt = new Date(b.finished_at || b.started_at || "").getTime();
    if (Number.isFinite(bt - at) && bt !== at) return bt - at;
    return Number(b.run_attempt || 0) - Number(a.run_attempt || 0);
  })[0] || null;
}

function runStateClass(run) {
  const state = String(run?.state || "").toLowerCase();
  if (state === "succeeded") return "ok";
  if (state.includes("fail") || state.includes("missing")) return "bad";
  return "live";
}

function isGenesis(model) {
  const identity = `${model?.label || ""} ${model?.model_repo || ""}`.toLowerCase();
  return identity.includes("genesis") || identity.includes("qwen/qwen3.6-35b-a3b");
}

function genesisScores(models) {
  const genesis = (models || []).find(isGenesis);
  const out = new Map();
  if (!genesis) return out;
  for (const run of completedRuns(genesis)) {
    if (run.score != null) out.set(run.suite, Number(run.score));
  }
  return out;
}

function scoreDelta(model, run, baseline) {
  if (!run || run.score == null || isGenesis(model)) return null;
  const base = baseline?.get(run.suite);
  if (base == null) return null;
  return Number(run.score) - Number(base);
}

function deltaCell(delta) {
  if (delta == null || !Number.isFinite(delta)) return null;
  const cls = delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const sign = delta > 0 ? "+" : "";
  return el("span", { class: `bench-delta ${cls}`, title: "delta vs genesis" }, `${sign}${fmt(delta, 3)}`);
}

function score(run, delta = null) {
  return el("span", { class: "bench-score-wrap" },
    el("span", { class: `bench-score ${runStateClass(run)}` }, run?.score == null ? "—" : fmt(run.score, 3)),
    deltaCell(delta));
}

function taskSummary(run) {
  const passed = run?.passed_count ?? "—";
  const total = run?.task_count ?? "—";
  return `${passed}/${total}`;
}

function detailHref(model, run) {
  const qs = new URLSearchParams();
  qs.set("model_id", model.id);
  if (run?.id) qs.set("run_id", run.id);
  return `./benchmark.html?${qs.toString()}`;
}

function kv(k, v, cls = "") {
  return el("div", { class: "kv" }, el("span", { class: "k" }, k), el("span", { class: cls ? `v ${cls}` : "v" }, v));
}

function cleanUserSimulator(run) {
  const llm = run?.environment?.user_llm || run?.harness_config?.user_llm || "gpt-5.2";
  return String(llm).includes("gpt-5.2") ? "gpt-5.2" : String(llm).replace(/^openrouter\/openai\//, "");
}

function benchVersion(run) {
  const version = run?.environment?.benchmark_version || run?.harness_config?.bench_version;
  if (version) return version;
  const ref = run?.environment?.benchmark_repo_ref || run?.harness_config?.repo_ref;
  if (ref) return `${TAU2_BENCH_VERSION} @ ${shortDigest(ref)}`;
  if (run?.suite === "swe_rebench_2026_03") return SWE_REBENCH_VERSION;
  if (run?.suite === "model_score") return SWE_VERIFIED_VERSION;
  return String(run?.suite || "").startsWith("tau2_") ? TAU2_BENCH_VERSION : "—";
}

function methodologyNotes(model, run) {
  const pulled = pulledRunFor(run);
  if (pulled) {
    return [
      `Evaluated using ${modelName(model)} as ${run.pulled_run_id}.`,
      `Agent harness: mini-swe-agent. Metric: Pass@1.`,
      `Scores, predictions and trajectories are published by the benchmarking service;`,
      run.report_uri
        ? `per-instance outcomes come from its ${pulled.key} grading report.`
        : `no ${pulled.key} grading report was published for this run, so instances are listed from its predictions without per-instance outcomes.`,
    ].join(" ");
  }
  if (run?.suite === "swe_rebench_2026_03") {
    const env = run?.environment || {};
    const metrics = run?.metrics || {};
    return [
      `Evaluated using ${modelName(model)} on ${env.dataset || "nebius/SWE-rebench-leaderboard"}.`,
      `Data file: ${env.data_files || "data/2026_03-00000-of-00001.parquet"}.`,
      `Agent harness: ${metrics.agent_harness || "mini-swe-agent"}.`,
      `Instances: ${run?.task_count || metrics.expected_prediction_count || "—"}.`,
      "Metric: Pass@1.",
    ].join(" ");
  }
  const cfg = run?.harness_config || {};
  const env = run?.environment || {};
  const parts = [
    `Evaluated using ${modelName(model)} with reasoning_effort: none.`,
    `User simulator: ${cleanUserSimulator(run)} with reasoning_effort: ${cfg.user_reasoning_effort || "low"}.`,
    `${cfg.num_trials || 1} trials.`,
    `Seed: ${cfg.seed || 300}.`,
    `Domain: ${env.domain || suiteDomain(run?.suite)}.`,
  ];
  if ((env.domain || run?.suite || "").includes("banking")) {
    parts.push("Banking domain evaluated with retrieval_config: qwen_embeddings.");
  }
  return parts.join(" ");
}

function suiteDomain(suite) {
  return String(suite || "").replace(/^tau2_/, "") || "—";
}

function renderMethodology(model, run) {
  const agentHarness = pulledRunFor(run) || run?.suite === "swe_rebench_2026_03";
  const actorKey = agentHarness ? "Agent Harness" : "User Simulator";
  const actorValue = agentHarness ? (run?.metrics?.agent_harness || "mini-swe-agent") : cleanUserSimulator(run);
  return el("div", { class: "detail-section" },
    el("h2", {}, "methodology"),
    el("div", { class: "kv-grid" },
      kv(actorKey, actorValue),
      kv("Evaluation Date", fmtDateTime(run?.finished_at || run?.started_at)),
      kv("Bench Version", benchVersion(run)),
      kv("Notes", methodologyNotes(model, run))));
}

function taskArtifactTasks(run) {
  return (run?.task_results || []).filter(task => /^https?:\/\//.test(String(task.artifact_uri || "")));
}

// The pulled suites publish no task rows: their per-instance outcome lives in the
// SWE-bench grading report, and each trajectory sits beside the run's preds file.
// Both are addressed relative to this page, so resolve them before the viewer's
// http-only artifact filter sees them.
function absolute(path) {
  try {
    return new URL(path, location.href).href;
  } catch {
    return path;
  }
}

function reportUrl(pulled, run) {
  const base = pulled.reportEndpoints?.[0];
  const model = String(run?.pulled_model || "").replaceAll("/", "__");
  if (!base || !model || !run?.pulled_run_id) return null;
  return absolute(`${base}/${run.pulled_run_id}/${model}.${pulled.key}_${run.pulled_run_id}.json`);
}

function trajectoryUrl(pulled, run, instanceId) {
  const base = pulled.predsEndpoints?.[0];
  if (!base || !run?.pulled_run_id) return null;
  return absolute(`${base}/${run.pulled_run_id}/${instanceId}/${instanceId}.traj.json`);
}

const REPORT_STATES = [
  ["resolved_ids", "RESOLVED", 1],
  ["unresolved_ids", "UNRESOLVED", 0],
  ["empty_patch_ids", "EMPTY_PATCH", 0],
  ["error_ids", "ERROR", null],   // graded, but the harness reached no verdict
];

function reportTaskResults(pulled, run, report) {
  const states = new Map();
  for (const [key, state, score] of REPORT_STATES) {
    for (const id of report?.[key] || []) {
      if (!states.has(id)) states.set(id, { state, score });
    }
  }
  return [...states.keys()].sort().map(id => ({
    task_name: id,
    state: states.get(id).state,
    score: states.get(id).score,
    artifact_uri: trajectoryUrl(pulled, run, id),
  }));
}

function predsUrl(pulled, run) {
  const base = pulled.predsEndpoints?.[0];
  if (!base || !run?.pulled_run_id) return null;
  return absolute(`${base}/${run.pulled_run_id}/preds.json`);
}

function predsTaskResults(pulled, run, preds) {
  const ids = Array.isArray(preds) ? preds.map(p => p?.instance_id) : Object.keys(preds || {});
  return ids.filter(Boolean).sort().map(id => ({
    task_name: id,
    state: "SUBMITTED",
    score: null,
    artifact_uri: trajectoryUrl(pulled, run, id),
  }));
}

// A score row only exists once the service has graded the run, so the report is the
// source for both the task rows and their outcome; the preds file is the fallback.
async function loadPulledRun(run) {
  const pulled = pulledRunFor(run);
  if (!pulled) return run;
  const url = reportUrl(pulled, run);
  const report = url ? await fetchJson(url) : null;
  if (!report) {
    const preds = await fetchJson(predsUrl(pulled, run));
    return preds ? { ...run, task_results: predsTaskResults(pulled, run, preds) } : run;
  }
  return {
    ...run,
    report_uri: url,
    task_results: reportTaskResults(pulled, run, report),
    task_count: report.total_instances ?? run.task_count,
    passed_count: report.resolved_instances ?? run.passed_count,
  };
}

function renderTrajectoryViewer(run) {
  const tasks = taskArtifactTasks(run);
  if (!tasks.length) {
    return el("div", { class: "detail-section" }, el("h2", {}, "trajectory"), el("div", { class: "empty" }, "no trajectory artifacts yet."));
  }
  return el("div", { class: "detail-section" },
    el("h2", {}, "trajectory"),
    el("div", { class: "trajectory-shell" },
      el("div", { class: "trajectory-toolbar" },
        el("select", { id: "trajectory-task-select" }, tasks.map((task, index) =>
          el("option", { value: String(index) }, `${task.task_name || "task"} ${task.trial_name || ""}`.trim()))),
        el("a", { id: "trajectory-open", href: tasks[0].artifact_uri, target: "_blank", rel: "noopener" }, "open json")),
      el("div", { class: "trajectory-layout" },
        el("div", { id: "trajectory-messages", class: "trajectory-messages" }, el("div", { class: "empty" }, "loading trajectory…")),
        el("div", { id: "trajectory-meta", class: "trajectory-meta" }))));
}

function renderTrajectoryMessages(payload) {
  const messages = Array.isArray(payload?.messages) ? payload.messages : [];
  if (!messages.length) {
    const error = payload?.error || payload?.info?.error || "no messages recorded.";
    return el("div", { class: "trajectory-error" }, String(error));
  }
  return messages.map((message, index) => {
    const role = message?.role || message?.sender || message?.source || `step ${index + 1}`;
    const content = message?.content ?? message?.message ?? message?.text ?? JSON.stringify(message);
    return el("div", { class: "trajectory-message" },
      el("div", { class: "trajectory-role" }, String(role)),
      el("pre", {}, typeof content === "string" ? content : JSON.stringify(content, null, 2)));
  });
}

function renderPulledTrajectoryMeta(payload, task, links) {
  const stats = payload?.info?.model_stats || {};
  const patch = String(payload?.info?.submission || "");
  return el("div", { class: "kv-grid trajectory-kv" },
    kv("task", payload?.instance_id || task?.task_name || "—"),
    kv("score", task?.score == null ? "— (no grading report)" : fmt(task.score, 3)),
    kv("state", task?.state || "—"),
    kv("termination", payload?.info?.exit_status || "—"),
    kv("api calls", stats.api_calls == null ? "—" : String(stats.api_calls)),
    kv("agent cost", stats.instance_cost == null ? "—" : fmt(Number(stats.instance_cost), 4)),
    kv("patch", patch.trim() ? `${patch.length} chars, ${patch.split("\n").length} lines` : "empty"),
    el("div", { class: "kv" }, el("span", { class: "k" }, "artifacts"), el("span", { class: "v" }, links.length ? links : "—")));
}

function renderTrajectoryMeta(payload, task, pulled) {
  const links = [];
  if (task?.artifact_uri) links.push(el("a", { href: task.artifact_uri, target: "_blank", rel: "noopener" }, "trajectory json"));
  if (pulled) return renderPulledTrajectoryMeta(payload, task, links);
  const reward = payload?.reward_breakdown || payload?.reward_info?.reward_breakdown;
  const agentCost = payload?.agent_cost || messageCost(payload, "assistant");
  const userCost = payload?.user_cost || messageCost(payload, "user");
  return el("div", { class: "kv-grid trajectory-kv" },
    kv("task", payload?.task_id || task?.task_name || "—"),
    kv("score", payload?.score == null ? "—" : fmt(payload.score, 3)),
    kv("state", payload?.state || task?.state || "—"),
    kv("termination", payload?.termination_reason || payload?.info?.exit_status || task?.metrics?.termination_reason || "—"),
    kv("duration", payload?.duration == null ? "—" : `${fmt(payload.duration, 2)}s`),
    kv("agent cost", agentCost == null ? "—" : fmt(agentCost, 4)),
    kv("user cost", userCost == null ? "—" : fmt(userCost, 4)),
    kv("reward", reward ? JSON.stringify(reward) : "—"),
    el("div", { class: "kv" }, el("span", { class: "k" }, "artifacts"), el("span", { class: "v" }, links.length ? links : "—")));
}

function messageCost(payload, role) {
  const total = (payload?.messages || []).reduce((sum, message) => {
    if (message?.role !== role) return sum;
    return sum + Number(message?.cost || 0) + Number(message?.raw_data?.usage?.cost || 0);
  }, 0);
  return total || null;
}

function wireTrajectory(run) {
  const tasks = taskArtifactTasks(run);
  const select = $("trajectory-task-select");
  const open = $("trajectory-open");
  const messages = $("trajectory-messages");
  const meta = $("trajectory-meta");
  if (!tasks.length || !select || !open || !messages || !meta) return;

  async function loadTask() {
    const task = tasks[Number(select.value) || 0];
    open.href = task.artifact_uri;
    mount(messages, el("div", { class: "empty" }, "loading trajectory…"));
    mount(meta);
    const payload = await fetchJson(task.artifact_uri);
    if (!payload) {
      mount(messages, el("div", { class: "trajectory-error" }, "could not load trajectory artifact."));
      return;
    }
    mount(messages, renderTrajectoryMessages(payload));
    mount(meta, renderTrajectoryMeta(payload, task, pulledRunFor(run)));
  }

  select.addEventListener("change", loadTask);
  loadTask();
}

function renderTaskTable(run) {
  const pulled = Boolean(pulledRunFor(run));
  const rows = (run?.task_results || []).map(task => {
    const info = task.metrics?.exception_info;
    return el("tr", {},
      el("td", { class: "model" }, el("span", { class: "model-cell", title: task.task_name }, task.task_name || "—")),
      el("td", {}, el("span", { class: `bench-state ${String(task.state || "").toLowerCase()}` }, task.state || "—")),
      el("td", { class: "r" }, task.score == null ? "—" : fmt(task.score, 3)),
      pulled ? false : el("td", { class: "when" }, fmtDateTime(task.finished_at)),
      pulled ? false : el("td", { class: "fail-reason-cell" }, info ? `${info.exception_type || "error"}: ${info.exception_message || ""}` : ""));
  });
  return el("div", { class: "data-table-wrap bench-task-wrap" },
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "task"), el("th", {}, "state"), el("th", { class: "r" }, "score"),
        pulled ? false : el("th", {}, "finished"), pulled ? false : el("th", {}, "error"))),
      el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: pulled ? "3" : "5" }, "no task rows.")))));
}

function render(model, selected, baseline) {
  const repo = hfRepoUrl(model);
  mount($("b-title"), repo ? el("a", { href: repo, target: "_blank", rel: "noopener" }, `${modelLabel(model)} · ${modelName(model)}`) : `${modelLabel(model)} · ${modelName(model)}`);
  $("b-sub").textContent = `${model.model_repo || model.id || "—"} · ${shortDigest(model.artifact_sha256 || model.model_hash)}`;

  const runs = completedRuns(model);
  const tabs = runs.length ? el("div", { class: "bench-run-tabs detail-section" }, runs.map(run =>
    el("a", { href: detailHref(model, run), class: run.id === selected?.id ? "active" : "" }, benchmarkLabel(run.suite)))) : null;

  mount($("b-body"),
    tabs,
    selected ? el("div", { class: "kv-grid" },
      kv("benchmark", benchmarkLabel(selected.suite)),
      kv("state", selected.state || "—", runStateClass(selected)),
      kv("score", score(selected, scoreDelta(model, selected, baseline))),
      kv("tasks", taskSummary(selected)),
      kv("worker", selected.worker_id || "—"),
      kv("finished", fmtDateTime(selected.finished_at))) : el("div", { class: "empty" }, "no completed benchmark runs yet."),
    selected ? renderMethodology(model, selected) : false,
    selected ? renderTrajectoryViewer(selected) : false,
    selected ? el("div", { class: "detail-section" }, el("h2", {}, "task results"), renderTaskTable(selected)) : false,
    el("div", { class: "detail-section" }, el("h2", {}, "model"),
      el("div", { class: "kv-grid" },
        kv("repo", model.model_repo || "—"),
        kv("label", modelLabel(model)),
        kv("artifact", model.artifact_uri || "—"),
        kv("model uri", model.model_uri || "—"),
        kv("activated", fmtDateTime(model.activated_at)))))
  if (selected) wireTrajectory(selected);
}

async function load() {
  const [raw, scores] = await Promise.all([fetchBenchmarks(), fetchPulledScores()]);
  if (!raw) {
    mount($("b-body"), el("div", { class: "empty" }, "could not load benchmark data."));
    return;
  }
  const data = mergePulledScores(raw, scores);
  const models = data.models || [];
  const model = models.find(m => m.id === modelId)
    || models.find(m => completedRuns(m).some(r => r.id === runId));
  if (!model) {
    mount($("b-body"), el("div", { class: "empty" }, "benchmark model not found."));
    return;
  }
  let selected = completedRuns(model).find(run => run.id === runId) || latestRun(model);
  if (selected?.detail_path) {
    const detail = await fetchBenchmarkRun(selected);
    if (detail) selected = { ...selected, ...detail };
  } else if (pulledRunFor(selected)) {
    selected = await loadPulledRun(selected);
  }
  render(model, selected, genesisScores(models));
}

load();
