// Sanity GPU host: the stateless pre-eval worker (no DB, no judge keys).
const fs = require("fs");
const path = require("path");

function loadEnv() {
  // Process env first (Doppler-injected secrets), then repo-root .env overrides when present.
  const file = process.env.ALBEDO_ENV_FILE || path.join(__dirname, "..", ".env");
  const env = { ...process.env };
  if (fs.existsSync(file)) {
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (m && !line.trim().startsWith("#")) env[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
  }
  return env;
}

module.exports = {
  apps: [
    {
      name: "albedo-gpu-sanity",
      cwd: path.join(__dirname, ".."),
      script: "uv",
      args: "run albedo gpu-sanity",
      env: loadEnv(),
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,
    },
  ],
};
