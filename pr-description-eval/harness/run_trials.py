#!/usr/bin/env python3
"""Run mini-swe-agent (Qwen3.8-27B via vLLM) on the PR-description task in fresh Docker containers.

Runs on the GPU node, from the project root, with the harness venv:
  .venv/bin/python harness/run_trials.py --variant base --n 10

For each trial i in 1..N (trials that already have grades.json are skipped, so reruns resume):
  1. start a fresh container from pr-desc-task:<variant> (no network, pinned CPUs)
  2. run mini-swe-agent's DefaultAgent + LitellmModel (bash tool calls) with the stock `mini.yaml`
     prompt templates and task/instruction.md as the task
  3. copy out final.diff, pr-12.json, calls.log, repo.tar (task/collect.sh) + trajectory.json into trials/<variant>/<i>/
  4. run task/tests/grade.py on the trial dir (-> grades.json) and write run.json (exit status, steps, tokens)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

import yaml  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASK = ROOT / "task"
VARIANTS = {"base": "0", "contributing": "1"}
# Job cpuset on d03-08 is 28-31,60-63,92-95,124-127; vLLM is pinned to 28-31,60-63, containers get the rest.
# Containers land in the calling Slurm step's cgroup on this node; `--memory` fails there
# ("cannot enter cgroupv2 ... invalid state"), so no memory limit is set.
CONTAINER_CPUS = os.environ.get("CONTAINER_CPUS", "92-95,124-127")

print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def build_image(variant: str) -> str:
    tag = f"pr-desc-task:{variant}"
    log(f"building {tag} (CONTRIBUTING={VARIANTS[variant]})")
    sh(["docker", "build", "-q", "--build-arg", f"CONTRIBUTING={VARIANTS[variant]}", "-t", tag,
        str(TASK / "environment")], stdout=subprocess.DEVNULL)
    if variant == "base":
        sh(["docker", "tag", tag, "pr-desc-task:latest"])
    return tag


def image_id(tag: str) -> str:
    return sh(["docker", "image", "inspect", "-f", "{{.Id}}", tag], capture_output=True, text=True).stdout.strip()


def collect(cid: str, out: Path) -> list[str]:
    """Copy the contract artifacts out of the (still running) container via the task's collect.sh."""
    r = subprocess.run(["bash", str(TASK / "collect.sh"), cid, str(out)], capture_output=True, text=True,
                       timeout=300)
    return [] if r.returncode == 0 else [f"collect.sh exit {r.returncode}: {(r.stdout + r.stderr)[-500:]}"]


def traj_stats(traj_path: Path) -> dict:
    if not traj_path.exists():
        return {}
    data = json.loads(traj_path.read_text())
    msgs = data.get("messages", [])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    n_actions = 0
    max_prompt = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        n_actions += len(m.get("extra", {}).get("actions", []))
        u = (m.get("extra", {}).get("response") or {}).get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens") or 0
        usage["completion_tokens"] += u.get("completion_tokens") or 0
        usage["reasoning_tokens"] += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        max_prompt = max(max_prompt, u.get("prompt_tokens") or 0)
    return {"exit_status": data.get("info", {}).get("exit_status", ""),
            "api_calls": data.get("info", {}).get("model_stats", {}).get("api_calls"),
            "n_actions": n_actions, "max_prompt_tokens": max_prompt, **usage}


def run_trial(variant: str, idx: int, tag: str, cfg: dict, args) -> dict:
    out = args.out_dir / str(idx)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():  # stale partial results from an interrupted run
        f.unlink()
    instruction = (TASK / "instruction.md").read_text()
    t0 = time.time()
    run = {"variant": variant, "trial": idx, "image": tag, "image_id": image_id(tag), "model": args.model,
           "temperature": args.temperature, "top_p": args.top_p, "step_limit": args.step_limit,
           "wall_limit_s": args.wall, "mini_swe_agent_config": "builtin mini.yaml templates", "errors": []}
    env = None
    try:
        env = DockerEnvironment(
            image=tag, cwd="/repo", timeout=args.cmd_timeout,
            env=cfg["environment"]["env"],
            run_args=["--network", "none", "--cpuset-cpus", CONTAINER_CPUS,
                      "--label", f"pr-desc-eval={args.label}-{idx}"],
            container_timeout=f"{args.wall + 1800}s",
        )
        model = LitellmModel(
            model_name=args.model,
            model_kwargs={"api_base": args.api_base, "api_key": "EMPTY", "temperature": args.temperature,
                          "top_p": args.top_p, "max_tokens": args.max_tokens, "timeout": 900,
                          "drop_params": True},
            cost_tracking="ignore_errors",
            observation_template=cfg["model"]["observation_template"],
            format_error_template=cfg["model"]["format_error_template"],
        )
        agent = DefaultAgent(
            model, env,
            system_template=cfg["agent"]["system_template"],
            instance_template=cfg["agent"]["instance_template"],
            step_limit=args.step_limit, cost_limit=0, wall_time_limit_seconds=args.wall,
            output_path=out / "trajectory.json",
        )
        log(f"{variant}/{idx}: started container {env.container_id[:12]}")
        try:
            result = agent.run(task=instruction)
            run["agent_exit_status"] = result.get("exit_status", "")
        except Exception as e:  # agent already saved trajectory with an exit message
            run["agent_exit_status"] = type(e).__name__
            run["errors"].append(f"agent: {e!r}")
        run["agent_seconds"] = round(time.time() - t0, 1)
        run["errors"] += collect(env.container_id, out)
    except Exception:
        run.setdefault("agent_exit_status", "HarnessError")
        run["errors"].append(f"harness: {traceback.format_exc()}")
    finally:
        # Remove by label: also catches a container that was created but failed to start (env is None then).
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=pr-desc-eval={args.label}-{idx}"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
        if env is not None:
            env.container_id = None  # already removed; stop mini-swe-agent's __del__ cleanup
    run.update(traj_stats(out / "trajectory.json"))
    if (out / "final.diff").exists():
        if err := grade(out, tag):
            run["errors"].append(err)
    run["total_seconds"] = round(time.time() - t0, 1)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    grades = json.loads((out / "grades.json").read_text()) if (out / "grades.json").exists() else {}
    log(f"{variant}/{idx}: exit={run.get('agent_exit_status')} calls={run.get('api_calls')} "
        f"tok={run.get('prompt_tokens')}/{run.get('completion_tokens')} {run['total_seconds']}s "
        f"grades={ {k: v for k, v in grades.items() if isinstance(v, bool)} } errors={len(run['errors'])}")
    return run


def grade(out: Path, tag: str) -> str | None:
    """Run the hidden graders on a collected trial dir (-> grades.json). Returns an error string on failure."""
    g = subprocess.run([sys.executable, str(TASK / "tests" / "grade.py"), str(out), "--runner", "docker",
                        "--image", tag], capture_output=True, text=True, timeout=900)
    (out / "grade.log").write_text(g.stdout + g.stderr)
    if g.returncode != 0 or not (out / "grades.json").exists():
        return f"grade.py exit {g.returncode}: {(g.stdout + g.stderr)[-500:]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--n", type=int, required=True, help="ensure trials 1..N exist")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="hosted_vllm/qwen3.8-27b")
    ap.add_argument("--api-base", default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:18431/v1"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192, help="per model response (thinking + answer)")
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument("--wall", type=int, default=1800, help="agent wall-clock limit, seconds")
    ap.add_argument("--cmd-timeout", type=int, default=120, help="per bash command, seconds")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun trials that already have grades.json")
    ap.add_argument("--regrade", action="store_true",
                    help="no agent runs: re-run grade.py on every collected trial 1..N of this variant")
    ap.add_argument("--out", help="trial root relative to the project (default: trials/<variant>), "
                                  "e.g. trials/qwen35-4b/base")
    args = ap.parse_args()
    args.out_dir = ROOT / (args.out or f"trials/{args.variant}")
    args.label = args.out_dir.relative_to(ROOT).as_posix().replace("/", "_")

    logging.basicConfig(level=logging.WARNING)
    if not (TASK / "READY").exists():
        sys.exit("task/READY missing: task is not validated yet")
    cfg = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())
    tag = f"pr-desc-task:{args.variant}" if (args.no_build or args.regrade) else build_image(args.variant)
    if args.regrade:
        for i in range(1, args.n + 1):
            out = args.out_dir / str(i)
            if not (out / "final.diff").exists():
                log(f"{args.variant}/{i}: no collected files, skipped")
                continue
            err = grade(out, tag)
            g = json.loads((out / "grades.json").read_text()) if (out / "grades.json").exists() else {}
            log(f"{args.variant}/{i}: regraded {err or ''} { {k: v for k, v in g.items() if isinstance(v, bool)} }")
        return
    todo = [i for i in range(1, args.n + 1)
            if args.force or not (args.out_dir / str(i) / "grades.json").exists()]
    log(f"{args.variant}: running trials {todo} with {args.workers} workers")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda i: run_trial(args.variant, i, tag, cfg, args), todo))


if __name__ == "__main__":
    main()
