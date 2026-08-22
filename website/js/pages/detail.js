import { ARTIFACT_TYPES } from "../config.js";
import { fetchDashboard, fetchText } from "../fetch.js";
import { verdictInfo, faultCategory, faultCodeLabel } from "../data.js";
import { el, mount, link } from "../dom.js";
import { pct, fmtDateTime, shortHotkey, shortDigest } from "../format.js";
import { judgeMeta, judgeVersion, hubRepoUrl, modelRepo, modelName, taoMinerUrl, kingTitleName } from "../model.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const evalRunId = params.get("eval_run_id");
const submissionId = params.get("submission_id");

function setHead(name, sub, modelUri) {
  const repoUrl = modelUri && hubRepoUrl(modelUri);
  mount($("d-title"), repoUrl ? link(repoUrl, name, { title: modelRepo(modelUri) }) : name);
  $("d-sub").textContent = sub || "";
}

const kv = (k, v, cls) => el("div", { class: "kv" }, el("span", { class: "k" }, k), el("span", { class: cls ? "v " + cls : "v" }, v));

// Judge label with a visible version tag (GLM 5.2, QWEN 3.5-397b-a17b, DEEPSEEK v3.2) —
// so the detail page always says which version judged this run.
const judgeLabel = m => {
  const v = judgeVersion(m);
  return v ? [judgeMeta(m).label, el("span", { class: "judge-ver" }, " " + v)] : judgeMeta(m).label;
};
const scoreCls = n => n == null ? "" : Number(n) >= 0.5 ? "ok" : "bad";


function parseJsonl(text) {
  return text.split(/\n+/).map(line => {
    try { return line.trim() ? JSON.parse(line) : null; } catch { return null; }
  }).filter(Boolean);
}

function mean(values) {
  const nums = values.filter(v => v != null).map(Number);
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
}

function categoryScore(record, id) {
  return mean((record.judge_results || []).map(j => j.parse_ok ? j.metric_scores?.[id] : null));
}

function metricText(scores = {}, categories = []) {
  const names = categories.length ? categories.map(c => c.id) : Object.keys(scores);
  return names.map(id => `${id} ${pct(scores[id])}`).join(" · ") || "—";
}

// binary scoring mode: judges in stable order; per side, judge_model -> judge_result record
function pivotBinary(record) {
  const judges = [...new Set((record.judge_results || []).map(j => j.judge_model))];
  const bySide = {};
  for (const j of record.judge_results || []) (bySide[j.side] ||= {})[j.judge_model] = j;
  return { judges, king: bySide.previous_king || {}, chal: bySide.challenger || {} };
}


const LEGACY_TAG_WEIGHTS = {
  "reference:explore": 2.0,
  "reference:action": 2.0,
  "reference:claims": 1.5,
  "reference:economy": 1.0,
  "reference:grounding": 0.5,
  "reference:verification": 0.5,
  "behavior:anchored_edit": 0.25,
};
const legacyWeightOf = tag =>
  LEGACY_TAG_WEIGHTS[tag] ?? (tag && tag.startsWith("reference:") ? 0.5 : 0);
// pick the weight rule that provably applied to this record: stamped weights when present,
// otherwise whichever of {v3 legacy map, uniform (pre-v3)} reproduces the stored score
function weightFnFor(record) {
  const qs = record.questions || [];
  if (qs.some(q => typeof q.weight === "number")) {
    return q => (typeof q.weight === "number" ? q.weight : 0);
  }
  const legacy = q => legacyWeightOf(q.tag);
  const uniform = () => 1;
  const jr = (record.judge_results || []).find(j => j.side === "challenger" && j.parse_ok);
  const stored = record.challenger_score;
  if (stored == null || !jr) return legacy;
  const recompute = wf => {
    let num = 0, den = 0;
    for (const q of qs) {
      const w = wf(q);
      if (!w) continue;
      const a = jr.answers?.[q.id];
      if (a === "0" || a === "1") { den += w; if (a === "1") num += w; }
    }
    let s = den ? num / den : null;
    if (s != null && record.chal_amputated_thinking) s *= 0.5;
    return s;
  };
  for (const wf of [legacy, uniform]) {
    const s = recompute(wf);
    if (s != null && Math.abs(s - stored) < 1e-4) return wf;
  }
  return legacy;
}

function scoreShares(records, keyFn) {
  const acc = {};
  let n = 0;
  for (const record of records.filter(x => x.scored)) {
    const p = pivotBinary(record);
    const wf = weightFnFor(record);
    const kMul = record.king_amputated_thinking ? 0.5 : 1;
    const cMul = record.chal_amputated_thinking ? 0.5 : 1;
    for (const m of p.judges) {
      const kj = p.king[m], cj = p.chal[m];
      if (!kj?.parse_ok || !cj?.parse_ok) continue;
      n++;
      let kden = 0, cden = 0;
      const nums = {};
      for (const q of record.questions || []) {
        const w = wf(q);
        if (!w) continue;
        const ka = kj.answers?.[q.id], ca = cj.answers?.[q.id];
        if (ka === "0" || ka === "1") kden += w;
        if (ca === "0" || ca === "1") cden += w;
        const t = (nums[keyFn(q)] ||= { k: 0, c: 0, wk: 0, wc: 0, cells: 0, w });
        t.cells++;
        if (ka === "0" || ka === "1") t.wk += w;
        if (ca === "0" || ca === "1") t.wc += w;
        if (ka === "1") t.k += w;
        if (ca === "1") t.c += w;
      }
      for (const [key, t] of Object.entries(nums)) {
        const b = (acc[key] ||= { king: 0, chal: 0, max: 0, cells: 0, weight: t.w });
        b.king += kden ? kMul * t.k / kden : 0;
        b.chal += cden ? cMul * t.c / cden : 0;
        // ceiling: what a perfect (unpenalized) model would earn from this tag
        b.max += ((kden ? t.wk / kden : 0) + (cden ? t.wc / cden : 0)) / 2;
        b.cells += t.cells;
      }
    }
  }
  return Object.entries(acc)
    .map(([key, b]) => ({
      key,
      n: b.cells,
      weight: b.weight,
      max: n ? b.max / n : null,
      king: n ? b.king / n : null,
      chal: n ? b.chal / n : null,
      diff: n ? (b.chal - b.king) / n : null,
    }))
    .sort((a, b) => Math.abs(b.diff || 0) - Math.abs(a.diff || 0));
}

function shareTable(rows, keyHeader, showWeight) {
  const signed = v => v == null ? "\u2014" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}`;
  const signedCls = v => v > 0 ? "ok" : v < 0 ? "bad" : "";
  const share = v => v == null ? "\u2014" : (v * 100).toFixed(1);
  if (!rows.length) return el("div", { class: "empty" }, "no tagged questions.");
  const tot = rows.reduce(
    (s, r) => ({ max: s.max + (r.max || 0), king: s.king + (r.king || 0), chal: s.chal + (r.chal || 0), diff: s.diff + (r.diff || 0) }),
    { max: 0, king: 0, chal: 0, diff: 0 });
  return el("table", { class: "data-table sample-judge-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, keyHeader), el("th", { class: "r" }, "questions"),
      showWeight ? el("th", { class: "r" }, "weight") : false,
      el("th", { class: "r", title: "score available from this tag for a perfect model" }, "max"),
      el("th", { class: "r", title: "this tag's share of the king final score" }, "king"),
      el("th", { class: "r", title: "this tag's share of the challenger final score" }, "challenger"),
      el("th", { class: "r", title: "challenger share minus king share" }, "\u0394"))),
    el("tbody", {},
      rows.map(row => el("tr", {},
        el("td", {}, row.key),
        el("td", { class: "r" }, String(row.n)),
        showWeight ? el("td", { class: "r muted-dash" }, row.weight ? `\u00d7${row.weight}` : "0") : false,
        el("td", { class: "r muted-dash" }, share(row.max)),
        el("td", { class: "r" }, share(row.king)),
        el("td", { class: "r" }, share(row.chal)),
        el("td", { class: `r ${signedCls(row.diff)}` }, signed(row.diff)))),
      el("tr", { class: "total-row" },
        el("td", { colspan: showWeight ? "3" : "2" }, el("b", {}, "final score")),
        el("td", { class: "r muted-dash" }, el("b", {}, share(tot.max))),
        el("td", { class: "r" }, el("b", {}, share(tot.king))),
        el("td", { class: "r" }, el("b", {}, share(tot.chal))),
        el("td", { class: `r ${signedCls(tot.diff)}` }, el("b", {}, signed(tot.diff))))));
}

const tagMarginTable = records => shareTable(scoreShares(records, q => q.tag || "untagged"), "tag", true);

function lazyDetails(props, summaryChildren, buildBody, bodyClass) {
  const body = el("div", { class: bodyClass });
  const d = el("details", props, el("summary", {}, ...summaryChildren), body);
  let built = false;
  d.addEventListener("toggle", () => {
    if (d.open && !built) { built = true; mount(body, buildBody()); }
  });
  return d;
}

async function downloadZip(btn, entries, map, zipName) {
  if (typeof JSZip === "undefined") return;
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "zipping…";
  try {
    const zip = new JSZip();
    await Promise.all(entries.map(async a => {
      try { const r = await fetch(map[a.key]); if (r.ok) zip.file(a.label, await r.blob()); } catch {}
    }));
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${zipName || "artifacts"}.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function renderArtifacts(map, zipName) {
  const section = $("d-artifacts");
  const entries = ARTIFACT_TYPES.filter(a => map && map[a.key]);
  if (!entries.length) {
    mount(section, el("h2", {}, "artifacts"), el("div", { class: "note" }, "No artifacts recorded for this eval."));
    return;
  }
  const zipBtn = el("button", { class: "btn-zip" }, "↓ download all");
  zipBtn.addEventListener("click", () => downloadZip(zipBtn, entries, map, zipName));

  const rows = entries.map(a => el("div", { class: "artifact-row" },
    el("span", { class: "a-name" }, a.label),
    el("span", { class: "a-type" }, a.type),
    el("a", { href: map[a.key], download: a.label, class: "a-open" }, "download ↓")));

  mount(section,
    el("h2", {}, "artifacts"),
    el("div", { class: "artifacts-actions" }, zipBtn),
    el("div", { class: "artifact-list" }, rows));
}

async function renderSampleScores(r, section, recordsP) {
  mount(section, el("h2", {}, "samples"), el("div", { class: "note" }, "Loading sample scores..."));
  if (!r.artifacts?.SCORING_RESULTS) return mount(section, el("h2", {}, "samples"), el("div", { class: "note" }, "No scoring artifact recorded."));

  const records = await recordsP;
  if (!records.length) {
    return mount(section, el("h2", {}, "samples"), el("div", { class: "note" }, "No sample scores found."));
  }

  const source = records.find(x => x.category_source)?.category_source;
  const sourceLine = source
    ? `${source.provider || "category"} · ${source.model || "model"} · ${source.prompt_version || "prompt"}`
    : records.some(x => x.scoring_mode === "binary")
      ? `${records.filter(x => x.scored).length}/${records.length} samples scored · binary questions`
      : `${records.length} scored samples`;

  mount(section,
    el("h2", {}, "samples"),
    el("div", { class: "sample-source" }, sourceLine),
    el("div", { class: "sample-list" }, records.map((record, i) =>
      record.scoring_mode === "binary" ? binarySampleCard(record, i) : legacySampleCard(record, i))));
}

function legacySampleCard(record, i) {
  const cats = record.categories || [];
  const judges = record.judge_results || [];
  return el("details", { class: "sample-card" },
    el("summary", {},
      el("span", { class: "sample-id" }, record.sample_id || `sample ${i + 1}`),
      el("span", { class: `sample-score ${scoreCls(record.sample_score)}` }, pct(record.sample_score)),
      el("span", { class: "sample-meta" }, `${judges.filter(j => j.parse_ok).length}/${judges.length} judges · ${(record.order || []).join(" -> ")}`)),
    el("div", { class: "sample-body" },
      cats.length ? el("table", { class: "data-table sample-cat-table" },
        el("thead", {}, el("tr", {}, el("th", {}, "category"), el("th", { class: "r" }, "score"), el("th", {}, "description"))),
        el("tbody", {}, cats.map(cat => el("tr", {},
          el("td", { class: "sample-cat" }, el("b", {}, cat.id), el("span", {}, cat.name)),
          el("td", { class: `r ${scoreCls(categoryScore(record, cat.id))}` }, pct(categoryScore(record, cat.id))),
          el("td", {}, cat.description || cat.scoring_guidance || "—"))))) : false,
      el("table", { class: "data-table sample-judge-table" },
        el("thead", {}, el("tr", {}, el("th", {}, "judge"), el("th", {}, "provider"), el("th", { class: "r" }, "mean"), el("th", {}, "metrics"))),
        el("tbody", {}, judges.map(j => el("tr", {},
          el("td", { class: "judge", title: j.judge_model }, judgeLabel(j.judge_model)),
          el("td", {}, j.provider || "—"),
          el("td", { class: `r ${j.parse_ok ? scoreCls(j.judge_mean) : "bad"}` }, j.parse_ok ? pct(j.judge_mean) : "error"),
          el("td", { class: "metric-line" }, metricText(j.metric_scores, cats))))))));
}

function binarySampleCard(record, i) {
  const p = pivotBinary(record);
  const jr = record.judge_results || [];
  const okN = jr.filter(j => j.parse_ok).length;
  const delta = record.challenger_score != null && record.king_score != null
    ? record.challenger_score - record.king_score : null;
  const flag = record.scored ? false
    : jr.length
      ? el("span", { class: "sample-flag warn", title: record.error || "partial judge failure" }, "partial")
      : el("span", { class: "sample-flag bad", title: record.error || "" }, "unscored");
  const zeroed = jr.filter(j => j.looped || j.corrupted).map(j =>
    el("span", {
      class: "sample-flag warn",
      title: j.corruption_reason || (j.loop_reasons || []).join("; ") || "",
    }, `${j.side === "previous_king" ? "king" : "chal"} ${j.looped ? "looped" : "truncated"}`));

  return lazyDetails({ class: "sample-card binary" },
    [
      el("span", { class: "sample-id" }, record.sample_id || `sample ${i + 1}`),
      el("span", { class: "sample-duel" },
        el("span", { class: "chal" }, `chal ${pct(record.challenger_score)}`),
        el("span", { class: "sep" }, " · "),
        `king ${pct(record.king_score)}`,
        delta != null ? el("span", { class: "delta " + (delta > 0 ? "ok" : delta < 0 ? "bad" : "") },
          ` ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}`) : false),
      el("span", { class: "sample-meta" },
        `${(record.questions || []).length} q · ${okN}/${jr.length} judges `, flag, ...zeroed),
    ],
    () => binarySampleBody(record, p),
    "sample-body");
}

function binarySampleBody(record, p) {
  const err = record.error ? el("div", { class: "note bad-note" }, record.error) : false;
  if (!(record.questions || []).length) {
    return [err || el("div", { class: "note" }, "no questions recorded.")];
  }
  return [err, binaryJudgeTable(p), questionList(record, p)];
}

function binaryJudgeTable(p) {
  const sideCell = j => el("td", { class: `r ${j ? (j.parse_ok ? scoreCls(j.yes_rate) : "bad") : ""}` },
    j ? (j.parse_ok ? pct(j.yes_rate) : "error") : "—");
  return el("table", { class: "data-table sample-judge-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, "judge"), el("th", {}, "provider"),
      el("th", { class: "r" }, "king"), el("th", { class: "r" }, "challenger"), el("th", {}, "status"))),
    el("tbody", {}, p.judges.map(m => {
      const king = p.king[m], chal = p.chal[m];
      const failed = [king, chal].filter(j => j && !j.parse_ok);
      return el("tr", {},
        el("td", { class: "judge", title: m }, judgeLabel(m)),
        el("td", {}, king?.provider || chal?.provider || "—"),
        sideCell(king),
        sideCell(chal),
        failed.length
          ? el("td", { class: "bad" }, failed[0].error || "parse error")
          : el("td", {}, king && chal ? "ok" : "—"));
    })));
}

function questionList(record, p) {
  return el("div", { class: "q-list" },
    el("div", { class: "q-head" },
      el("span", {}, "#"), el("span", {}, "question"),
      el("span", { class: "c" }, "king"), el("span", { class: "c" }, "chal")),
    (record.questions || []).map(q => questionRow(q, p)));
}

function answerGlyphs(sideMap, judges, qid) {
  return judges.map(m => {
    const a = sideMap[m]?.answers?.[qid];
    const [glyph, cls] = a === "1" ? ["✓", "ok"] : a === "0" ? ["✗", "bad"] : ["·", "muted"];
    const v = judgeVersion(m);
    const tag = v ? `${judgeMeta(m).label} ${v}` : judgeMeta(m).label;
    return el("span", { class: "q-ans " + cls, title: `${tag}: ${a ?? "no answer"}` }, glyph);
  });
}

function questionRow(q, p) {
  return lazyDetails({ class: "q-row" },
    [
      el("span", { class: "q-id" }, q.id),
      el("span", { class: "q-text" }, q.text || "—"),
      el("span", { class: "c" }, answerGlyphs(p.king, p.judges, q.id)),
      el("span", { class: "c" }, answerGlyphs(p.chal, p.judges, q.id)),
    ],
    () => [
      el("table", { class: "data-table q-expl-table" },
        el("thead", {}, el("tr", {}, el("th", {}, "judge"), el("th", {}, "king"), el("th", {}, "challenger"))),
        el("tbody", {}, p.judges.map(m => el("tr", {},
          el("td", { class: "judge", title: m }, judgeLabel(m)),
          explCell(p.king[m], q.id),
          explCell(p.chal[m], q.id))))),
      q.example_bad ? el("div", { class: "q-bad" }, el("b", {}, "example bad: "), q.example_bad) : false,
    ],
    "q-expl");
}

function explCell(j, qid) {
  const a = j?.answers?.[qid];
  const glyph = a === "1" ? "✓ " : a === "0" ? "✗ " : "· ";
  const text = j?.explanations?.[qid] || (j && !j.parse_ok ? (j.error || "judge parse error") : "—");
  return el("td", { class: "q-expl-cell " + (a === "1" ? "ok" : a === "0" ? "bad" : "muted") },
    el("b", {}, glyph), el("span", {}, text));
}

const ROMAN = ["I", "II", "III", "IV"];
const stopClick = e => e.stopPropagation();

// win-on-both: a submission has two scored passes, so each gets its own head — same
// model, its own eval run id — and the head doubles as the selector for the body below.
function passHeads(r, netuid, passes) {
  const repoUrl = hubRepoUrl(r.model_uri);
  return passes.map((p, i) => {
    const active = p.eval_run_id === r.eval_run_id;
    const v = verdictInfo(p);
    const name = modelName(p);
    return el("div", {
      class: active ? "detail-head pass-head active" : "detail-head pass-head",
      title: `${v.badge} · ${fmtDateTime(p.finished_at)}`,
      onClick: () => {
        if (active) return;
        history.replaceState(null, "", `?eval_run_id=${encodeURIComponent(p.eval_run_id)}`);
        renderEval(p, netuid, passes);
      },
    },
      el("div", { class: "eyebrow" }, `eval result ${ROMAN[i] || i + 1}`),
      el("h1", {}, repoUrl ? link(repoUrl, name, { title: modelRepo(p.model_uri), onClick: stopClick }) : name),
      el("div", { class: "sub" }, `${p.eval_run_id} · uid ${p.uid ?? "—"} · ${shortHotkey(p.hotkey)}`));
  });
}

function renderEval(r, netuid, passes = [r]) {
  const multi = passes.length >= 2;
  $("d-head").style.display = multi ? "none" : "";
  mount($("d-heads"), multi ? passHeads(r, netuid, passes) : []);
  $("d-eyebrow").textContent = "eval result";
  setHead(modelName(r), `${r.eval_run_id} · uid ${r.uid ?? "—"} · ${shortHotkey(r.hotkey)}`, r.model_uri);

  const v = verdictInfo(r);
  const grid = el("div", { class: "kv-grid" },
    kv("result", v.badge, v.won ? "gold" : "bad"),
    kv("challenger", pct(v.chalMean), "gold"),
    kv("king", pct(v.kingMean)),
    kv("margin",
       v.winMargin != null
         ? `${Number(v.winMargin) * 100 >= 0 ? "+" : ""}${(Number(v.winMargin) * 100).toFixed(5)}%`
         : "—",
       v.winMargin != null ? (v.won ? "ok" : "bad") : undefined),
    r.required_win_margin != null
      ? kv("required margin", `≥ +${(Number(r.required_win_margin) * 100).toFixed(2)}%`)
      : false,
    kv("turns", r.total_turns != null ? `${r.valid_turns ?? r.total_turns}/${r.total_turns}` : "—"),
    kv("vllm errors", `${r.chal_vllm_errors ?? 0}c / ${r.king_vllm_errors ?? 0}k`),
    kv("scoring", r.scoring_mode || "fixed_metrics"),
    kv("judge errors", r.judge_errors ?? "—"),
    kv("finished", fmtDateTime(r.finished_at)));

  const recordsP = r.artifacts?.SCORING_RESULTS
    ? fetchText(r.artifacts.SCORING_RESULTS).then(t => t ? parseJsonl(t) : [])
    : Promise.resolve([]);

  const binary = r.scoring_mode === "binary";

  const byMetric = Object.keys(r.score_breakdown?.by_category || {}).length
    ? r.score_breakdown.by_category
    : (r.score_breakdown?.by_metric || {});
  const metrics = Object.keys(byMetric).length
    ? el("div", { class: "kv-grid" }, Object.entries(byMetric).map(([k, val]) => kv(k, pct(val))))
    : null;
  const metricTitle = r.scoring_mode === "glm_categories" ? "category slots" : "metrics";

  const king = r.king || {};
  const tao = taoMinerUrl(netuid, king.hotkey);
  const tagSection = binary ? el("div", { class: "detail-section" }) : null;
  const samplesSection = el("div", { class: "detail-section" });

  mount($("d-body"),
    grid,
    tagSection,
    metrics ? el("div", { class: "detail-section" }, el("h2", {}, metricTitle), metrics) : false,
    samplesSection,
    el("div", { class: "detail-section" }, el("h2", {}, "king it faced"),
      el("div", { class: "kv-grid" },
        kv("king era", kingTitleName(king.king_version)),
        kv("king model", modelName(king)),
        kv("king uid", tao ? link(tao, String(king.uid ?? "—")) : (king.uid ?? "—")))));

  if (tagSection) {
    mount(tagSection, el("h2", {}, "score by tag"), el("div", { class: "note" }, "Loading…"));
    recordsP.then(records => mount(tagSection, el("h2", {}, "score by tag"),
      tagMarginTable(records)));
  }
  renderSampleScores(r, samplesSection, recordsP);
  renderArtifacts(r.artifacts, `eval-${r.eval_run_id}`);
}

function renderFail(f) {
  $("d-eyebrow").textContent = "failed submission";
  setHead(modelName(f), `${f.submission_id} · uid ${f.uid ?? "—"} · ${shortHotkey(f.hotkey)}`, f.model_uri);

  const cat = faultCategory(f);
  const raw = (f.fault_message || f.fault_code || "").toString();
  const grid = el("div", { class: "kv-grid" },
    kv("category", cat.label, "bad"),
    kv("state", f.state || "—"),
    kv("fault class", f.fault_class || "—"),
    kv("fault code", f.fault_code || "—"),
    kv("model digest", shortDigest(f.model_hash)),
    kv("when", fmtDateTime(f.updated_at)));

  mount($("d-body"),
    grid,
    el("div", { class: "detail-section" }, el("h2", {}, "failure detail"),
      el("div", { class: "fail-panel" },
        el("div", { class: "fp-code" }, faultCodeLabel(f)),
        el("div", { class: "fp-detail" }, raw || "(no detail)"))));

  renderArtifacts(f.artifacts, `submission-${f.submission_id}`);
}

function passesOf(raw, r) {
  if (!r.submission_id) return [r];
  const runs = (raw.eval_runs || [])
    .filter(e => e.submission_id === r.submission_id)
    .sort((a, b) => String(a.finished_at || "").localeCompare(String(b.finished_at || "")));
  return runs.length >= 2 ? runs : [r];
}

async function load() {
  const raw = await fetchDashboard();
  if (!raw) { setHead("unavailable", "could not load dashboard data"); return; }
  const netuid = raw.chain?.netuid;

  if (evalRunId) {
    const r = (raw.eval_runs || []).find(e => e.eval_run_id === evalRunId)
      || (raw.current_eval?.eval_run_id === evalRunId ? raw.current_eval : null);
    if (r) return renderEval(r, netuid, passesOf(raw, r));
    const f = (raw.fails || []).find(e => e.eval_run_id === evalRunId);
    if (f) return renderFail(f);
    setHead(evalRunId, "eval run not found");
    return;
  }

  if (submissionId) {
    const f = (raw.fails || []).find(e => e.submission_id === submissionId)
      || (raw.queue || []).find(e => e.submission_id === submissionId);
    if (f) return renderFail(f);
    setHead(submissionId, "submission not found");
    return;
  }

  setHead("missing id", "pass ?eval_run_id=… or ?submission_id=…");
}

load();
