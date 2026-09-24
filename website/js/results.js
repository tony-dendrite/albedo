
const LEGACY_BENCHMARKS = [
  { suite: "swe_rebench_2026_03", niceName: "SWE-rebench", order: 1 },
  { suite: "model_score", niceName: "SWE-bench Verified", order: 2 },
];

export function benchmarkRegistry(manifest) {
  const bySuite = new Map(LEGACY_BENCHMARKS.map(entry => [entry.suite, entry]));
  for (const benchmark of manifest?.benchmarks || []) {
    if (!benchmark.enabled) continue;
    const suite = benchmark.legacy_suite || benchmark.name;
    bySuite.set(suite, { suite, niceName: benchmark.nice_name || benchmark.name, order: benchmark.order ?? 9999 });
  }
  return [...bySuite.values()].sort((a, b) => a.order - b.order);
}

function distributedRun(benchmark, row) {
  const status = String(row.status || "pending").toLowerCase();
  const partial = row.score != null && status !== "complete";
  const id = `distributed:${benchmark.name}:${row.model_key}`;
  return {
    id,
    run_id: id,
    suite: benchmark.legacy_suite || benchmark.name,
    state: status.toUpperCase(),
    score: row.score == null ? null : row.score / 100,
    task_count: row.total ?? row.progress?.total ?? null,
    passed_count: row.resolved ?? null,
    finished_at: status === "complete" ? row.updated_at : null,
    score_meta: row.resolved != null && row.total != null
      ? `${partial ? "partial · " : ""}${row.resolved}/${row.total} resolved`
      : null,
    partial_score: partial,
    source: "distributed",
    distributed_benchmark: benchmark.name,
    distributed_model_key: row.model_key,
    distributed_status: status,
    distributed_progress: row.progress || {},
    distributed_updated_at: row.updated_at || null,
    benchmark_nice_name: benchmark.nice_name || benchmark.name,
  };
}

// The site lists genesis under its upstream repo; the service publishes it as king-genesis.
const modelKey = model => (model.label === "genesis" ? "genesis" : model.model_repo);
const rowKey = row => (row.reign === 0 ? "genesis" : row.model_repo);

// External reference models (GLM 5.2 via OpenRouter, ...) are published with reign null under their
// own slug. They are not kings: they join the data flagged `reference`, never matched to a king.
const isReferenceRow = row => row.reign == null && !String(row.model_key || "").startsWith("king-");

// "z-ai/glm-5.2" -> "GLM 5.2"
function referenceLabel(repo) {
  return String(repo || "").split("/").pop().split("-")
    .map(part => (/^[a-z]+$/i.test(part) ? part.toUpperCase() : part))
    .join(" ");
}

export function mergeDistributedResults(data, manifest) {
  const runsByRepo = new Map();
  const references = new Map();
  for (const benchmark of manifest?.benchmarks || []) {
    if (!benchmark.enabled) continue;
    for (const row of manifest.results?.[benchmark.name] || []) {
      if (!row.model_repo || !row.model_key) continue;
      const run = distributedRun(benchmark, row);
      if (isReferenceRow(row)) {
        const ref = references.get(row.model_key) || {
          id: `reference:${row.model_key}`,
          label: referenceLabel(row.model_repo),
          model_repo: row.model_repo,
          reference: true,
          runs: [],
        };
        ref.runs.push(run);
        references.set(row.model_key, ref);
      } else {
        runsByRepo.set(rowKey(row), [...(runsByRepo.get(rowKey(row)) || []), run]);
      }
    }
  }
  if (!runsByRepo.size && !references.size) return data;
  const models = (data?.models || []).filter(model => !model.reference).map(model => {
    const runs = runsByRepo.get(modelKey(model));
    if (!runs) return model;
    const replaced = new Set(runs.map(run => run.suite));
    return { ...model, runs: [...(model.runs || []).filter(run => !replaced.has(run.suite)), ...runs] };
  });
  return { ...data, models: [...models, ...references.values()] };
}

export function distributedRunFor(model, suite) {
  return (model?.runs || []).find(run => run?.source === "distributed" && run.suite === suite) || null;
}

export function distributedProgress(run) {
  const progress = run?.distributed_progress || {};
  const total = progress.total ?? run?.task_count;
  if (!total) return null;
  const completed = progress.completed ?? 0;
  const errored = progress.errored ?? 0;
  return {
    count: completed + errored,
    total,
    ratio: Math.min(1, (completed + errored) / total),
    completed,
    errored,
    status: run.distributed_status,
    updatedAt: run.distributed_updated_at,
    fresh: ["pending", "running", "scoring"].includes(run.distributed_status),
    distributed: true,
  };
}
