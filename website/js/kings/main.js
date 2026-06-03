import { POLL_MS, DATA_ENDPOINTS } from "./config.js";
import { render } from "./render.js";

let DATA_URL = null;

async function fetchDashboard() {
  const urls = DATA_URL ? [DATA_URL, ...DATA_ENDPOINTS.filter(u => u !== DATA_URL)] : [...DATA_ENDPOINTS];
  for (const url of urls) {
    try {
      const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) continue;
      const data = await r.json();
      if (url !== "../dashboard.json") DATA_URL = url;
      return data;
    } catch {}
  }
  return null;
}

async function poll() {
  const data = await fetchDashboard();
  if (data) render(data);
}

poll();
setInterval(poll, POLL_MS);
