import { SUBNET_ALPHA_DAY } from "./config.js";

export const fmt = (n, d = 3) => n == null ? "—" : Number(n).toFixed(d);
export const fmtSigned = (n, d = 3) => n == null ? "—" : (n >= 0 ? "+" : "") + Number(n).toFixed(d);
export const shortHotkey = hk => !hk ? "—" : hk.slice(0, 6) + "…" + hk.slice(-4);
export const shortDigest = d => {
  if (!d) return "—";
  if (d.startsWith("sha256:")) return "sha256:" + d.slice(7, 19) + "…";
  if (d.startsWith("hf:"))     return "hf:"     + d.slice(3, 15) + "…";
  return d.slice(0, 19) + "…";
};
export const fmtTime = iso => {
  if (!iso) return "—";
  try { return new Date(iso).toISOString().slice(0, 16).replace("T", " ") + "Z"; }
  catch { return iso; }
};
export const fmtScore3 = score => score == null ? "—" : Number(score).toFixed(3);

export function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function toRoman(n) {
  const vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
  const syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"];
  let out = "";
  for (let i = 0; i < vals.length; i++) {
    while (n >= vals[i]) { out += syms[i]; n -= vals[i]; }
  }
  return out;
}

export function kingDateShort(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(d.getUTCDate()).padStart(2, "0");
    return `${mm}/${dd}`;
  } catch { return "—"; }
}

export function fmtRelative(iso) {
  if (!iso) return "—";
  try {
    const ms = Date.now() - new Date(iso).getTime();
    if (ms < 0) return "just now";
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 48) return `${h}h ago`;
    const d = Math.floor(h / 24);
    return `${d}d ago`;
  } catch { return "—"; }
}

export function fmtWhenCell(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(d.getUTCDate()).padStart(2, "0");
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mi = String(d.getUTCMinutes()).padStart(2, "0");
    return `<span class="rel">${fmtRelative(iso)}</span><span class="sep"> · </span><span class="dt">${mm}/${dd} ${hh}:${mi}</span>`;
  } catch { return fmtRelative(iso); }
}

export function fmtAlphaDay(weight) {
  if (weight == null || weight <= 0) return "—";
  return (weight * SUBNET_ALPHA_DAY).toFixed(1);
}

export function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
