// Reverse tunnel: exposes templar's albedo-eval-postgres (127.0.0.1:65432) on the
// EVAL GPU pod's localhost:65432, so model_validation there reaches the core DB.
// Target host comes from the albedo-eval-pro6000 alias in ~/.ssh/config.
module.exports = {
  apps: [
    {
      name: "albedo-eval-db-tunnel",
      script: "ssh",
      args: "-N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:65432:127.0.0.1:65432 albedo-eval-pro6000",
      autorestart: true,
    },
  ],
};
