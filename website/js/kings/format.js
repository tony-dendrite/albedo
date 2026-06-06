export function toRoman(n) {
  const vals = [1000,900,500,400,100,90,50,40,10,9,5,4,1];
  const syms = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"];
  let out = "";
  for (let i = 0; i < vals.length; i++) while (n >= vals[i]) { out += syms[i]; n -= vals[i]; }
  return out;
}

export function kingTitle(reignNumber) {
  const n = Number(reignNumber);
  if (reignNumber == null || !Number.isFinite(n) || n <= 0) return "base model";
  const r = toRoman(n);
  return r ? `ALBEDO-${r}` : "base model";
}

export function shortDigest(d) {
  if (!d) return "—";
  if (d.startsWith("sha256:")) return "sha256:" + d.slice(7, 19) + "…";
  if (d.startsWith("hf:"))     return "hf:" + d.slice(3, 15) + "…";
  return d.slice(0, 16) + "…";
}

export function shortHotkey(hk) {
  if (!hk) return "—";
  return hk.slice(0, 8) + "…" + hk.slice(-4);
}

export function fmtDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const pad = n => String(n).padStart(2, "0");
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
  } catch { return iso; }
}

export function fmtRelative(iso) {
  if (!iso) return "";
  try {
    const ms = Date.now() - new Date(iso).getTime();
    if (ms < 0) return "just now";
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 48) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  } catch { return ""; }
}

export function escHtml(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
