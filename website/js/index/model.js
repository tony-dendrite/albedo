import { toRoman } from "./format.js";
import { EVALS_BASE } from "./config.js";

export const JUDGE_META = {
  "deepseek-ai/DeepSeek-V3.2-TEE": { letter: "D", label: "DEEPSEEK" },
  "Qwen/Qwen3-235B-A22B-Thinking-2507": { letter: "Q", label: "QWEN" },
  "moonshotai/Kimi-K2.6-TEE": { letter: "K", label: "KIMI" },
};

export function judgeShortName(model) {
  if (!model) return "—";
  const parts = model.split("/");
  let name = parts[parts.length - 1];
  if (name.length > 18) name = name.slice(0, 16) + "…";
  return name;
}

export function judgeMeta(model) {
  if (JUDGE_META[model]) return JUDGE_META[model];
  const short = judgeShortName(model);
  return { letter: short.charAt(0).toUpperCase(), label: short.toUpperCase().slice(0, 12) };
}

export function kingTitleName(reignNumber) {
  if (reignNumber == null) return "ALBEDO";
  const roman = toRoman(Number(reignNumber) + 1);
  return roman ? `ALBEDO-${roman}` : "ALBEDO";
}

export function challengerDisplayName(hotkey) {
  if (!hotkey) return "—";
  return `ALBEDO-${hotkey.slice(0, 5).toUpperCase()}`;
}

export function hubRepoUrl(repo) {
  if (!repo) return null;
  const parts = repo.split("/");
  if (parts.length < 2) return "https://hub.hippius.com/models";
  return `https://hub.hippius.com/models/${parts[0]}/${parts.slice(1).join("/")}`;
}

export function modelLinkHtml(repo, digest, label) {
  const text = label || "—";
  const url = hubRepoUrl(repo);
  if (!url || text === "—") return text;
  const title = digest ? `${repo}@${digest}` : (repo || text);
  return `<a href="${url}" target="_blank" rel="noopener" title="${title}">${text}</a>`;
}

export function taoMinerUrl(netuid, hotkey) {
  if (netuid == null || !hotkey) return null;
  return `https://taomarketcap.com/subnets/${netuid}/miners?query=${encodeURIComponent(hotkey)}`;
}

export function evalsUrlForEntry(entry, history) {
  const cid = entry.challenge_id;
  if (cid && cid !== "seed") {
    const h = (history || []).find(x => x.challenge_id === cid);
    if (h?.evals_url) return h.evals_url;
  }
  return EVALS_BASE;
}
