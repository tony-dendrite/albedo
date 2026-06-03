import { NETUID, JUDGE_META } from "./config.js";

export function hubUrl(repo) {
  if (!repo) return null;
  const p = repo.split("/");
  return p.length >= 2 ? `https://hub.hippius.com/models/${p[0]}/${p.slice(1).join("/")}` : null;
}

export function taoUrl(hotkey) {
  if (!hotkey) return null;
  return `https://taomarketcap.com/subnets/${NETUID}/miners?query=${encodeURIComponent(hotkey)}`;
}

export function judgeLetter(model) {
  return (JUDGE_META[model] || {}).letter || (model || "?").charAt(0).toUpperCase();
}
