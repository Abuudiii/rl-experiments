#!/usr/bin/env bash
# Train, evaluate and record Space Invaders (HF Deep RL course, unit 3).
# Run from this directory with the venv activated.
set -euo pipefail
cd "$(dirname "$0")"
python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml
python -m rl_zoo3.enjoy --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder logs/
python -m rl_zoo3.record_video --algo dqn --env SpaceInvadersNoFrameskip-v4 -n 1000 -f logs/
