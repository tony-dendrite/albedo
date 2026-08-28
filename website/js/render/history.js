import { el, mount, link } from "../dom.js";
import { pct, fmtRelative, fmtDateTime } from "../format.js";
import { hubRepoUrl, modelRepo, modelName, modelCellText, taoMinerUrl, kingTitleName } from "../model.js";
import { verdictInfo, faultCategory, faultCodeLabel } from "../data.js";

const stop = e => e.stopPropagation();

// win-on-both: the two passes of one submission are one duel, so they collapse into
// a single row (newest pass represents it) and the margin cell carries both margins.
export function collapsePasses(rows) {
  const seen = new Map();
  const out = [];
  for (const r of rows) {
    const group = r.submission_id && seen.get(r.submission_id);
    if (group) { group.passes.push(r); continue; }
    const entry = { ...r, passes: [r] };
    if (r.submission_id) seen.set(r.submission_id, entry);
    out.push(entry);
  }
  return out;
}

const passScores = r => (r.passes || [r]).slice().reverse()
  .map((p, i) => `pass ${i + 1}: ${pct(p.score_challenger)} / ${pct(p.score_king)}`).join(" · ");

// final score aggregated across all judges (challenger / king); the per-judge
// breakdown stays on the detail page.
function scoreCell(r) {
  if (r.score_challenger == null) return el("span", { class: "muted-dash" }, "—");
  const title = (r.passes || []).length >= 2 ? passScores(r) : "challenger / king";
  if (r.score_king == null) return el("span", { class: "judge-scores", title: "challenger" }, pct(r.score_challenger));
  return el("span", { class: "judge-scores", title },
    pct(r.score_challenger), el("span", { class: "sep" }, " / "),
    el("span", { class: "king-score" }, pct(r.score_king)));
}

function marginSpan(m, required, title) {
  const cls = m >= (required ?? 0) ? "ok" : "bad";
  return el("span", { class: `margin-pct ${cls}`.trim(), title }, `${m > 0 ? "+" : ""}${pct(m)}%`);
}

function marginCell(r) {
  const required = r.required_win_margin;
  const reqTitle = required != null ? `required ≥ +${pct(required)}%` : null;
  const margins = Array.isArray(r.pass_margins) && r.pass_margins.length >= 2
    ? r.pass_margins : null;
  if (margins) {
    return el("span", { class: "judge-scores", title: "pass 1 / pass 2 (win-on-both)" },
      marginSpan(margins[0], required, reqTitle),
      el("span", { class: "sep" }, " / "),
      marginSpan(margins[1], required, reqTitle));
  }
  const m = r.win_margin;
  if (m == null) return el("span", { class: "muted-dash" }, "—");
  return marginSpan(m, required, reqTitle);
}

const evalHref = r => `detail.html?eval_run_id=${encodeURIComponent(r.eval_run_id || "")}`;
const failHref = f => f.eval_run_id
  ? `detail.html?eval_run_id=${encodeURIComponent(f.eval_run_id)}`
  : `detail.html?submission_id=${encodeURIComponent(f.submission_id || "")}`;

export function renderHistory(container, rows, netuid, currentKingEvalRunId) {
  if (!rows.length) {
    mount(container, el("div", { class: "empty" }, "no completed duels match."));
    return;
  }

  const head = el("tr", {},
    el("th", {}, "when"),
    el("th", {}, "uid"),
    el("th", {}, "model"),
    el("th", {}, "vs king"),
    el("th", { class: "center", title: "final score across all judges: challenger / king" }, "score"),
    el("th", { class: "center", title: "challenger score − king score" }, "margin"),
    el("th", { class: "r" }, "result"));

  const body = rows.map(r => {
    const v = verdictInfo(r);
    const isCurrentKing = currentKingEvalRunId != null && r.eval_run_id === currentKingEvalRunId;
    const repo = modelRepo(r.model_uri);
    const repoUrl = hubRepoUrl(r.model_uri);
    const tao = taoMinerUrl(netuid, r.hotkey);
    const king = r.king || {};
    const kingName = kingTitleName(king.king_version);
    const kingUrl = hubRepoUrl(king.model_uri);
    const kingTitle = modelRepo(king.model_uri);
    return el("tr", { class: isCurrentKing ? "clickable crowned-now" : "clickable", onClick: () => { location.href = evalHref(r); } },
      el("td", { class: "when", title: fmtDateTime(r.finished_at) }, fmtRelative(r.finished_at)),
      el("td", { class: "uid" }, tao ? link(tao, String(r.uid ?? "—"), { onClick: stop }) : String(r.uid ?? "—")),
      el("td", { class: "model" }, repoUrl ? link(repoUrl, modelCellText(r), { class: "model-cell", title: repo, onClick: stop }) : el("span", { class: "model-cell", title: repo }, modelCellText(r))),
      el("td", { class: "model vs" }, kingUrl ? link(kingUrl, kingName, { class: "model-cell", title: kingTitle, onClick: stop }) : el("span", { class: "model-cell", title: kingTitle }, kingName)),
      el("td", { class: "center" }, scoreCell(r)),
      el("td", { class: "center" }, marginCell(r)),
      el("td", { class: "r" }, el("span", { class: `verdict-badge ${v.badge}` }, v.badge)));
  });

  mount(container, el("table", { class: "data-table" }, el("thead", {}, head), el("tbody", {}, body)));
}

function failReasonCell(f) {
  const text = (f.fault_message || f.fault_code || faultCategory(f).label).toString();
  return el("td", { class: "fail-reason-cell" },
    el("span", { class: "fail-code", title: f.fault_class || "" }, faultCodeLabel(f)),
    el("span", { class: "fail-reason", title: text }, text));
}

export function renderFails(container, rows, netuid) {
  if (!rows.length) {
    mount(container, el("div", { class: "empty" }, "no failures match."));
    return;
  }
  const body = rows.map(f => {
    const repo = modelRepo(f.model_uri);
    const repoUrl = hubRepoUrl(f.model_uri);
    const tao = taoMinerUrl(netuid, f.hotkey);
    return el("tr", { class: "clickable", onClick: () => { location.href = failHref(f); } },
      el("td", { class: "when", title: fmtDateTime(f.updated_at) }, fmtRelative(f.updated_at)),
      el("td", { class: "uid" }, tao ? link(tao, String(f.uid ?? "—"), { onClick: stop }) : String(f.uid ?? "—")),
      el("td", { class: "model" }, repoUrl ? link(repoUrl, modelCellText(f), { class: "model-cell", title: repo, onClick: stop }) : el("span", { class: "model-cell", title: repo }, modelCellText(f))),
      failReasonCell(f));
  });
  mount(container,
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "model"), el("th", {}, "reason"))),
      el("tbody", {}, body)));
}
