#!/usr/bin/env python3
"""Drive the TRL HarborEnv path (no model) on the logslice Harbor task: no-op, code-only and oracle rollouts.

    .venv-rl/bin/python rl/probe_env.py [--modes noop codeonly oracle]

For each mode: start a fresh sandbox via PrDescBashEnv.reset (timed), run the mode's commands through the env's
`bash` tool, then read env.reward (runs tests/test.sh in the sandbox, timed). Prints one JSON line per mode with the
reward, the verifier's reward.json components, and timings.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prdesc_env import PrDescBashEnv  # noqa: E402

TASK = HERE / "harbor_tasks" / "logslice"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["noop", "codeonly", "oracle"])
    args = ap.parse_args()
    solve = (TASK / "solution" / "solve.sh").read_text()
    for mode in args.modes:
        env = PrDescBashEnv(environment_type="docker")
        t0 = time.time()
        instruction = env.reset(task_dir=str(TASK))
        t_reset = time.time() - t0
        out = ""
        if mode != "noop":
            env._run(env._env.exec(f"cat > /tmp/solve.sh <<'SOLVE_EOF'\n{solve}\nSOLVE_EOF"))
            out = env.bash("bash /tmp/solve.sh" + (" --code-only" if mode == "codeonly" else ""))
        t1 = time.time()
        reward = env.reward
        t_verify = time.time() - t1
        details = env._run(env._env.exec("cat /logs/verifier/reward.json /logs/verifier/details.json")).stdout
        t2 = time.time()
        env._run(env._stop())
        t_stop = time.time() - t2
        print(json.dumps({"mode": mode, "reward": reward, "reset_s": round(t_reset, 1),
                          "verify_s": round(t_verify, 1), "stop_s": round(t_stop, 1),
                          "instruction_tail": instruction[-120:], "solve_tail": out[-300:],
                          "verifier_files": details}), flush=True)


if __name__ == "__main__":
    main()
