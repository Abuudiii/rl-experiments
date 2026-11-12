#!/usr/bin/env bash
# Copy the student pipeline's files to the node (extract over the top, nothing is deleted, other
# workers' files are not touched). Prints the md5 of the set on the node and locally.
set -euo pipefail
cd "$(dirname "$0")/.."
LOGIN=${LOGIN:?set LOGIN to the ssh host of the Slurm login node}
JOB=${JOB:?set JOB to the id of the running Slurm job that holds the GPUs}
NODE_DIR=${NODE_DIR:?set NODE_DIR to the project directory on the node}
FILES=$(ls fpv/render_synth.py fpv/student.py fpv/dagger.py scripts/*student* scripts/*dagger* | sort)
tar cz $FILES |
  ssh -o BatchMode=yes $LOGIN "srun --jobid=$JOB --overlap -n1 --chdir=$NODE_DIR bash -c 'tar xz && md5sum $(echo $FILES) | md5sum'"
md5sum $FILES | md5sum
