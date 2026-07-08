// Backend host: one orchestrator process + the SSH tunnels to both GPU workers.
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

const env = loadEnv();
const repo = path.join(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "albedo-backend",
      cwd: repo,
      script: "uv",
      args: "run albedo backend",
      env,
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,
    },
    {
      name: "albedo-eval-tunnel",
      script: "ssh",
      args: [
        "-N",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-L", `${env.ALBEDO_TUNNEL_BACKEND_LOCAL_GPU_PORT || 18090}:127.0.0.1:${env.ALBEDO_REMOTE_EVAL_API_PORT || 8090}`,
        `${env.ALBEDO_GPU_HOST_USER || "root"}@${env.ALBEDO_GPU_HOST_SSH_HOST}`,
      ],
      autorestart: true,
      restart_delay: 5000,
    },
    {
      name: "albedo-sanity-tunnel",
      script: "ssh",
      args: [
        "-N",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-L", `${env.ALBEDO_SANITY_TUNNEL_LOCAL_PORT || 19100}:127.0.0.1:${env.SANITY_REMOTE_API_PORT || 9100}`,
        `${env.ALBEDO_SANITY_GPU_HOST_USER || "root"}@${env.ALBEDO_SANITY_GPU_HOST_SSH_HOST}`,
      ],
      autorestart: true,
      restart_delay: 5000,
    },
  ],
};
