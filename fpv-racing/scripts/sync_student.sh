#!/usr/bin/env bash
# Copy the student pipeline's files to the node (extract over the top, nothing is deleted, other
# workers' files are not touched). Prints the md5 of each file on the node and locally.
set -euo pipefail
cd "$(dirname "$0")/.."
LOGIN=${LOGIN:?set LOGIN to the ssh host of the Slurm login node}
JOB=${JOB:?set JOB to the id of the running Slurm job that holds the GPUs}
NODE_DIR=${NODE_DIR:?set NODE_DIR to the project directory on the node}
FILES="fpv/render_synth.py fpv/student.py fpv/dagger.py scripts/run_dagger.py scripts/train_student.py
  scripts/eval_student.py scripts/rerender_dagger_shards.py scripts/student_rounds_table.py
  scripts/student_failure_sheet.py scripts/check_student_synth_render.py scripts/sync_student.sh
  scripts/student_scenario_table.py scripts/student_ambiguity.py scripts/extra_collect.py
  scripts/student_flight_video.py scripts/check_highlight_frames.py scripts/student_failure_diag.py
  scripts/student_final_table.py scripts/record_closed_loop.py fpv/closed_loop.py scripts/check_hairpin_tracks.py"
FILES=$(ls $FILES 2>/dev/null | sort)
tar cz $FILES |
  ssh -o BatchMode=yes $LOGIN "srun --jobid=$JOB --overlap -n1 --chdir=$NODE_DIR bash -c 'tar xz && md5sum $(echo $FILES)'" > /tmp/fpv_sync_node.md5
md5sum $FILES | diff - /tmp/fpv_sync_node.md5 && echo "in sync: $(echo $FILES | wc -w) files"
