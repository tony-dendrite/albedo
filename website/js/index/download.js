export const DL_ICON = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
  <path d="M6 1.5v8M2.5 6L6 9.5 9.5 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M2 11h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
</svg>`;

const ZIP_FILES = [
  "responses_champion.jsonl",
  "responses_challenger.jsonl",
  "judge_raw.jsonl",
  "scores.json",
];

// ZIP download button for eval directory artifacts.
// Uses JSZip (must be loaded globally via <script> in the HTML).
export function dlButton(dirUrlOrLegacyUrl) {
  if (!dirUrlOrLegacyUrl) {
    return `<button type="button" class="data-dl" disabled aria-label="download">${DL_ICON}</button>`;
  }
  // Heuristic: if the URL ends with a known file extension it's a legacy single-file URL.
  const isDir = !dirUrlOrLegacyUrl.match(/\.(jsonl\.gz|jsonl|json|zip)(\?.*)?$/i);
  if (isDir) {
    return `<button type="button" class="data-dl" data-zip-dir="${escAttr(dirUrlOrLegacyUrl)}" aria-label="download eval ZIP">${DL_ICON}</button>`;
  }
  // Legacy single-file link.
  return `<a class="data-dl" href="${escAttr(dirUrlOrLegacyUrl)}" target="_blank" rel="noopener" aria-label="download rollouts">${DL_ICON}</a>`;
}

function escAttr(s) {
  return String(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

async function _downloadZip(dirUrl) {
  if (typeof JSZip === "undefined") {
    // Fallback: open the dir URL in a new tab.
    window.open(dirUrl, "_blank");
    return;
  }
  const base = dirUrl.replace(/\/$/, "");
  const zip  = new JSZip();
  const fetches = ZIP_FILES.map(async name => {
    const url  = `${base}/${name}`;
    const resp = await fetch(url);
    if (!resp.ok) return; // skip missing files silently
    const blob = await resp.blob();
    zip.file(name, blob);
  });
  await Promise.all(fetches);
  const zipBlob = await zip.generateAsync({ type: "blob" });
  const dirParts = base.split("/");
  const evalNum  = dirParts[dirParts.length - 1] || "eval";
  const date     = dirParts[dirParts.length - 2] || "";
  const filename = `eval-${date}-${evalNum}.zip`.replace(/^eval--/, "eval-");
  const url  = URL.createObjectURL(zipBlob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Fail-record JSON download.
const FAIL_DL_STORE = {};
let _failDlSeq = 0;

export function failDlButton(h) {
  const key = "fdl" + (_failDlSeq++);
  FAIL_DL_STORE[key] = h;
  return `<button type="button" class="data-dl" data-fdl="${key}" aria-label="download fail JSON">${DL_ICON}</button>`;
}

document.addEventListener("click", async e => {
  // ZIP download handler.
  const zipBtn = e.target.closest("[data-zip-dir]");
  if (zipBtn) {
    zipBtn.disabled = true;
    try {
      await _downloadZip(zipBtn.dataset.zipDir);
    } catch (err) {
      console.error("ZIP download failed:", err);
      window.open(zipBtn.dataset.zipDir, "_blank");
    } finally {
      zipBtn.disabled = false;
    }
    return;
  }

  // Fail JSON download handler.
  const failBtn = e.target.closest("[data-fdl]");
  if (!failBtn) return;
  const h = FAIL_DL_STORE[failBtn.dataset.fdl];
  if (!h) return;
  const blob = new Blob([JSON.stringify(h, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  const name = (h.hotkey || "fail").slice(0, 8);
  const ts   = (h.completed_at || "").replace(/[:.]/g, "-").slice(0, 19);
  a.href     = url;
  a.download = `fail-${name}-${ts}.json`;
  a.click();
  URL.revokeObjectURL(url);
});
