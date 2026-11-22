#!/usr/bin/env bash
# Mirror envs/ patches/ scripts/ from the local project dir to the node (through the Slurm login host).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGIN=${LOGIN:?set LOGIN to the ssh host of the Slurm login node}
JOB=${JOB:?set JOB to the id of the running Slurm job that holds the GPUs}
NODE_DIR=${NODE_DIR:?set NODE_DIR to the project directory on the node}
tar cz --exclude='__pycache__' envs patches scripts |
  ssh -o BatchMode=yes $LOGIN "srun --jobid=$JOB --overlap -n1 bash -c 'mkdir -p $NODE_DIR && cd $NODE_DIR && tar xz && if [ -d dreamerv3/embodied/envs ]; then cp envs/donkey.py dreamerv3/embodied/envs/donkey.py; fi && find envs patches scripts -type f -not -path \"*__pycache__*\" | xargs md5sum | sort -k2 | md5sum'"
find envs patches scripts -type f -not -path '*__pycache__*' | xargs md5sum | sort -k2 | md5sum
