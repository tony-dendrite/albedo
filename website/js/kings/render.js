import { kingTitle, shortDigest, shortHotkey, fmtDate, fmtRelative, escHtml } from "./format.js";
import { hubUrl, taoUrl, judgeLetter } from "./model.js";
import { buildKingsList, applyDisplayStartBlock } from "./data.js";

function renderRow(k, isFirst) {
  const isCurrent = isFirst;
  const title   = kingTitle(k.reign_number);
  const repo    = k.model_repo || "";
  const digest  = k.king_digest || k.model_digest || "";
  const hkUrl   = taoUrl(k.hotkey);
  const mdlUrl  = hubUrl(repo);

  const badgeCls   = isCurrent ? "current" : "past";
  const badgeLabel = isCurrent ? "current"  : "past";
  const inferredNote = k._inferred
    ? `<span class="era-inferred" title="reign number inferred from position">~</span>` : "";
  const eraHtml = `
    <td class="col-era">
      <div class="era-name">${inferredNote}${escHtml(title)}</div>
      <span class="era-badge ${badgeCls}">${badgeLabel}</span>
    </td>`;

  const uid = k.uid != null ? k.uid : "—";
  const uidInner = hkUrl
    ? `<a href="${escHtml(hkUrl)}" target="_blank" rel="noopener" title="${escHtml(k.hotkey||"")}">${uid}</a>`
    : uid;
  const hotkeyInner = k.hotkey
    ? (hkUrl
        ? `<a href="${escHtml(hkUrl)}" target="_blank" rel="noopener" title="${escHtml(k.hotkey)}">${shortHotkey(k.hotkey)}</a>`
        : shortHotkey(k.hotkey))
    : "—";
  const uidHtml = `
    <td class="col-uid">
      <div class="uid-val">${uidInner}</div>
      <div class="hotkey-val">${hotkeyInner}</div>
    </td>`;

  const repoHtml = mdlUrl
    ? `<a href="${escHtml(mdlUrl)}" target="_blank" rel="noopener">${escHtml(repo || "—")}</a>`
    : escHtml(repo || "—");
  const digestHtml = mdlUrl
    ? `<a href="${escHtml(mdlUrl)}" target="_blank" rel="noopener" title="${escHtml(digest)}">${shortDigest(digest)}</a>`
    : shortDigest(digest);
  const modelHtml = `
    <td class="col-model">
      <div class="model-repo">${repoHtml}</div>
      <div class="model-digest">${digestHtml}</div>
    </td>`;

  const judgesHtml = `
    <td class="col-judges">
      <div class="judges-row">
        ${(k.judges || []).map(j => {
          const score = j.king_mean != null ? (j.king_mean * 100).toFixed(1) + "%" : "—";
          return `<span class="judge-pill" title="${escHtml(j.model)}">
            <span class="jl">${judgeLetter(j.model)}</span>${score}
          </span>`;
        }).join("")}
      </div>
    </td>`;

  const dateHtml = `
    <td class="col-date">
      <div class="date-abs">${fmtDate(k.crowned_at)}</div>
      <div class="date-rel">${fmtRelative(k.crowned_at)}</div>
    </td>`;

  const rowCls = isCurrent ? " class=\"is-current\"" : "";
  return `<tr${rowCls}>${eraHtml}${uidHtml}${modelHtml}${judgesHtml}${dateHtml}</tr>`;
}

export function render(d) {
  const fd     = applyDisplayStartBlock(d);
  const kings  = buildKingsList(fd);
  const empty  = document.getElementById("kings-empty");
  const table  = document.getElementById("kings-table");
  const tbody  = document.getElementById("kings-tbody");
  const meta   = document.getElementById("kings-meta");
  const notice = document.getElementById("kings-notice");

  if (!kings.length) {
    empty.textContent = "no kings yet.";
    empty.hidden = false;
    table.hidden = true;
    return;
  }

  meta.textContent = `${kings.length} king${kings.length !== 1 ? "s" : ""}`;
  tbody.innerHTML = kings.map((k, i) => renderRow(k, i === 0)).join("");
  empty.hidden = true;
  table.hidden = false;

  const inferredCount = kings.filter(k => k._inferred).length;
  notice.textContent = inferredCount
    ? `~ reign numbers on ${inferredCount} older ${inferredCount === 1 ? "entry" : "entries"} are inferred from position — block and UID unavailable for those`
    : "";
}
