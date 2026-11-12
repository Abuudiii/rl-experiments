#!/usr/bin/env bash
# Recreate what fpv/render_airsim.py needs on this machine: the packaged Blocks environment of
# Project AirSim v1.0.1 and a Python venv with the `projectairsim` client.
#
#   bash scripts/setup_airsim.sh            # run ON THE NODE, from the project directory
#
# Result (node-local, may disappear with /tmp):
#   $FPV_AIRSIM_ROOT/blocks/   unzipped Blocks-Linux-1.0.1 (Blocks/Binaries/Linux/Blocks-Linux-Shipping)
#   $FPV_AIRSIM_ROOT/venv/     python venv: projectairsim 1.0.1 client, numpy, opencv-python
# Downloads are kept on the network home so they survive: third_party/Blocks-Linux-1.0.1.zip and
# third_party/ProjectAirSim (shallow clone of tag v1.0.1; the client is installed from it).
# Idempotent: finished parts are skipped. Takes about 3 minutes from a cold cache.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${FPV_AIRSIM_ROOT:-/tmp/fpv_airsim}
KEEP=$PWD/third_party
ZIP=$KEEP/Blocks-Linux-1.0.1.zip
ZIP_URL=https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1/Blocks-Linux-1.0.1.zip
ZIP_SHA256=4a0c7b6de27ed32bdf4de1c31c222c19b796500108b654382b5d76b4ba000040
REPO=$KEEP/ProjectAirSim
EXE=$ROOT/blocks/Blocks/Binaries/Linux/Blocks-Linux-Shipping
mkdir -p "$ROOT" "$KEEP"

if ! echo "$ZIP_SHA256  $ZIP" | sha256sum -c --status 2>/dev/null; then
  if [ -f "$ROOT/Blocks-Linux-1.0.1.zip" ] && echo "$ZIP_SHA256  $ROOT/Blocks-Linux-1.0.1.zip" | sha256sum -c --status; then
    cp "$ROOT/Blocks-Linux-1.0.1.zip" "$ZIP"
  else
    curl -fL --retry 3 -o "$ZIP.part" "$ZIP_URL" && mv "$ZIP.part" "$ZIP"
  fi
  echo "$ZIP_SHA256  $ZIP" | sha256sum -c
fi

if [ ! -x "$EXE" ]; then
  rm -rf "$ROOT/blocks.tmp" && mkdir -p "$ROOT/blocks.tmp"
  unzip -q "$ZIP" -d "$ROOT/blocks.tmp"
  # the archive may or may not have a single top-level directory
  SRC=$(dirname "$(find "$ROOT/blocks.tmp" -maxdepth 3 -name Blocks.sh | head -1)")
  rm -rf "$ROOT/blocks" && mv "$SRC" "$ROOT/blocks" && rm -rf "$ROOT/blocks.tmp"
  chmod +x "$ROOT/blocks/Blocks.sh" "$EXE"
fi

if [ ! -d "$REPO/client/python/projectairsim" ]; then
  rm -rf "$REPO"
  git clone --quiet --depth 1 --branch v1.0.1 https://github.com/iamaisim/ProjectAirSim.git "$REPO"
fi

if ! "$ROOT/venv/bin/python" -c "import projectairsim, numpy, cv2" 2>/dev/null; then
  rm -rf "$ROOT/venv"
  python3 -m venv "$ROOT/venv"
  "$ROOT/venv/bin/pip" install --quiet --upgrade pip
  # not editable: the venv must not depend on the clone staying in place
  "$ROOT/venv/bin/pip" install --quiet "$REPO/client/python/projectairsim" opencv-python numpy
fi

"$ROOT/venv/bin/python" -c "import projectairsim, numpy, cv2; print('client ok: numpy', numpy.__version__, 'cv2', cv2.__version__)"
test -x "$EXE" && echo "blocks ok: $EXE"
