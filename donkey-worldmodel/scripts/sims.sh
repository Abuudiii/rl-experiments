#!/usr/bin/env bash
# Start N headless Donkey simulators (one Docker container each) and stay in the foreground.
#   scripts/sims.sh <port_base> <n>
# Run it as its own long-lived Slurm step; stopping the step removes the containers.
# Random cones on and beside the road are off; set CONES=1 to get the simulator's default back.
set -euo pipefail
BASE=$1
N=$2
SIM=/tmp/absheikh_donkey
EXE=$(cd $SIM && find . -name donkey_sim.x86_64 | head -1)
EXE=/sim/${EXE#./}
mkdir -p $SIM/logs
# Unity reads these settings from a prefs file in the (per-container) home directory.
cat > $SIM/prefs <<EOF
<unity_prefs version_major="1" version_minor="1">
	<pref name="generateRandomCones" type="int">${CONES:-0}</pref>
</unity_prefs>
EOF
cleanup() { docker rm -f $(docker ps -aq --filter label=absheikh_donkey_$BASE) >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
cleanup
for i in $(seq 0 $((N - 1))); do
  P=$((BASE + i))
  D=$((P % 1000 + 200))  # X display; unique per sim because the containers share the host network
  # xvfb-run hangs as a container's first process (it waits for a signal Xvfb never delivers), so start Xvfb directly.
  docker run --rm --init --name absheikh_donkey_$P --label absheikh_donkey --label absheikh_donkey_$BASE \
    --network host --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e LP_NUM_THREADS=${LP_NUM_THREADS:-8} \
    -e DISPLAY=:$D -v $SIM:/sim -w /tmp absheikh/donkey-sim:1 \
    sh -c "mkdir -p /tmp/home/.config/unity3d/unknown/unknown; cp /sim/prefs /tmp/home/.config/unity3d/unknown/unknown/prefs; \
      Xvfb :$D -screen 0 640x480x24 -nolisten tcp & sleep 3; exec $EXE \
      --port $P --host 0.0.0.0 ${SIM_GL_FLAG--force-glcore} -screen-fullscreen 0 -screen-width 160 -screen-height 120 \
      -screen-quality 'Very Low' \
      -logFile /sim/logs/sim_$P.log" &
done
wait
