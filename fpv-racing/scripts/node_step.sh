#!/usr/bin/env bash
# Run a bash script (read from stdin) as its own long-lived Slurm step on the node.
#   scripts/node_step.sh <step-name> <n-cpus> <<'EOF' ... EOF
# Log: runs/steps/<step-name>.log on the node. Stop it by killing its srun client on the login host:
#   ssh $LOGIN 'pkill -f "fpv_step_<step-name>[.]sh"'
set -euo pipefail
NAME=$1
CPUS=${2:-4}
LOGIN=${LOGIN:?set LOGIN to the ssh host of the Slurm login node}
JOB=${JOB:?set JOB to the id of the running Slurm job that holds the GPUs}
NODE_DIR=${NODE_DIR:?set NODE_DIR to the project directory on the node}
ssh -o BatchMode=yes $LOGIN "srun --jobid=$JOB --overlap -n1 --chdir=$NODE_DIR bash -c 'mkdir -p runs/steps runs/teacher && cat > runs/steps/fpv_step_$NAME.sh'"
timeout 20 ssh -o BatchMode=yes $LOGIN "cd /tmp && setsid nohup srun --jobid=$JOB --overlap -n1 -c$CPUS --chdir=$NODE_DIR bash -c 'exec bash runs/steps/fpv_step_$NAME.sh > runs/steps/$NAME.log 2>&1' > /tmp/fpv_step_${NAME}_srun.out 2>&1 < /dev/null & sleep 2; echo launched $NAME" </dev/null || true
