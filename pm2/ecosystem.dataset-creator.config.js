const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, "..", ".env");
  const env = {};
  if (!fs.existsSync(envPath)) return env;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    env[trimmed.slice(0, index)] = trimmed.slice(index + 1);
  }
  return env;
}

module.exports = {
  apps: [
    {
      name: "albedo-dataset-creator",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/python",
      args: "scripts/dataset_creator/pipeline.py --watch",
      env: loadEnv(),
      autorestart: true,
      max_restarts: 50,
      restart_delay: 30000,
    },
  ],
};
