#!/bin/bash
# Harbor verifier entrypoint: runs in the agent's container after the rollout (shared verifier mode).
# Writes /logs/verifier/reward.json (see verify.py for the reward definition).
set -uo pipefail
mkdir -p /logs/verifier
if ! python3 /tests/verify.py; then
  echo '{"reward": 0.0}' > /logs/verifier/reward.json
fi
