export const POLL_MS = 4000;

export const DATA_ENDPOINTS = [
  "./data/dashboard.json",
];
export const STATE_ENDPOINTS = [
  "./data/state.json",
];


export const BENCHMARK_ENDPOINTS = [
  "./data/benchmarks.json",
];

export const PULLED_SUITES = [
  {
    suite: "model_score",
    key: "verified",
    scoreEndpoints: ["./data/model-scores.json"],
    predsEndpoints: ["./research/bench_results_verified"],
    reportEndpoints: ["./research/bench_results/eval/verified"],
    totalFallback: 500,
    fromKing: 0,
  },
  {
    suite: "swe_rebench_2026_03",
    key: "rebench",
    scoreEndpoints: ["./data/rebench-scores.json"],
    predsEndpoints: ["./research/bench_results_rebench"],
    reportEndpoints: ["./research/bench_results/eval/rebench"],
    totalFallback: 110,
    fromKing: 123,
  },
];

export const PREDS_POLL_MS = 15000;
export const PREDS_STALE_MS = 15 * 60 * 1000;

export const MANIFEST_ENDPOINTS = [
  "./datasets/manifest.meta.json",
];

export const LLMS_URLS = [
  "./llms.txt",
];

export const REGISTRATION_ENDPOINTS = [
  "./data/registrations_30d.json",
];

export const ENDPOINT_CACHE_KEY = "albedo.endpoint.v3";
export const SUBNET_NETUID = 97;

// Runs render only the entries their artifact map actually has, so the two legacy types below
// keep working for older runs while new runs simply show fewer links.
export const ARTIFACT_TYPES = [
  { key: "EVAL_VERDICT", label: "verdict.json", type: "json" },
  { key: "GENERATED_SAMPLES", label: "generated-samples.jsonl", type: "jsonl" },
  { key: "SCORING_RESULTS", label: "scoring-results.jsonl", type: "jsonl" },
  { key: "JUDGE_RESULTS", label: "judge-results.jsonl", type: "jsonl" },      // legacy
  { key: "EVAL_TRANSCRIPT", label: "duel-transcript.jsonl", type: "jsonl" },  // legacy
  { key: "REMOTE_PROGRESS", label: "progress.jsonl", type: "jsonl" },
  { key: "REMOTE_LOGS", label: "remote-logs.txt", type: "text" },
  { key: "SANITY_RESULT", label: "sanity-result.json", type: "json" },
];
