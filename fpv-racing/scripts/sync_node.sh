#!/usr/bin/env bash
# Mirror fpv/ and scripts/ from the local project dir to the node (through the Slurm login host).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGIN=${LOGIN:?set LOGIN to the ssh host of the Slurm login node}
JOB=${JOB:?set JOB to the id of the running Slurm job that holds the GPUs}
NODE_DIR=${NODE_DIR:?set NODE_DIR to the project directory on the node}
tar cz --exclude='__pycache__' fpv scripts |
  ssh -o BatchMode=yes $LOGIN "srun --jobid=$JOB --overlap -n1 --chdir=$NODE_DIR bash -c 'tar xz && find fpv scripts -type f -not -path \"*__pycache__*\" | xargs md5sum | sort -k2 | md5sum'"
find fpv scripts -type f -not -path '*__pycache__*' | xargs md5sum | sort -k2 | md5sum
