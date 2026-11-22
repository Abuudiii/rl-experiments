#!/usr/bin/env bash
# One-time node setup; run inside an srun step with cwd = the project dir on the node. Safe to re-run.
set -euo pipefail
ROOT=$(pwd)
SIM=/tmp/absheikh_donkey
DREAMER_SHA=e01491fad6434b2245a3b8ca201dd7faedcc458c
DONKEY_SHA=a1f4ca6961e17a6929c1f2e883358a1edafb479d
SIM_URL=https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimLinux.zip
SIM_SHA256=fd4c637e41d11832724fba4e9aaf2b3f00b1bde9bdefa2753903a046ee3790ad
UV=/home/AMD/absheikh/.local/bin/uv

if [ ! -d dreamerv3/.git ]; then
  git clone -q https://github.com/danijar/dreamerv3 dreamerv3
fi
# Always rebuild the tree from the pinned commit plus our patch, so an updated patch takes effect.
git -C dreamerv3 checkout -q -f $DREAMER_SHA
git -C dreamerv3 apply "$ROOT/patches/dreamerv3.patch"
cp envs/donkey.py dreamerv3/embodied/envs/donkey.py

if [ ! -x .venv/bin/python ]; then
  $UV venv -p /usr/bin/python3.12 .venv
  $UV pip install -p .venv/bin/python \
    jax==0.12.0.1 jaxlib==0.12.0.1 jax-rocm10-plugin==0.12.0 jax-rocm10-pjrt==0.12.0 \
    numpy pillow gymnasium av chex colored_traceback einops elements granular ipdb jaxtyping \
    ninjax optax portal scope tqdm matplotlib \
    "gym-donkeycar @ git+https://github.com/tawnkramer/gym-donkeycar@$DONKEY_SHA"
fi

mkdir -p $SIM/logs
cd $SIM
if [ -z "$(find . -name donkey_sim.x86_64 | head -1)" ]; then
  curl -sSL -o DonkeySimLinux.zip $SIM_URL
  echo "$SIM_SHA256  DonkeySimLinux.zip" | sha256sum -c -
  unzip -q -o DonkeySimLinux.zip
fi
EXE=$(find . -name donkey_sim.x86_64 | head -1)
chmod +x "$EXE"
echo "sim executable: $SIM/${EXE#./}"

if [ -z "$(docker images -q absheikh/donkey-sim:1)" ]; then
  cat > Dockerfile <<'EOF'
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    xvfb xauth libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglu1-mesa libx11-6 libxext6 libxcursor1 \
    libxrandr2 libxinerama1 libxi6 libxrender1 libasound2t64 ca-certificates && rm -rf /var/lib/apt/lists/*
EOF
  mkdir -p ctx && cp Dockerfile ctx/ && docker build -q -t absheikh/donkey-sim:1 ctx
fi
echo "setup done"
