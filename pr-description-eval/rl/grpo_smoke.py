#!/usr/bin/env python3
"""GRPO smoke test: TRL GRPOTrainer + HarborSpec on the single logslice Harbor task, vLLM colocated, one GPU.

Not a training run: proves rollouts execute in docker sandboxes, rewards come back and optimizer steps happen.

    .venv-rl/bin/python rl/grpo_smoke.py --model Qwen/Qwen3.5-2B --max-steps 2

Writes rl/runs/<name>/: trainer logs (log_history.json), env_timings.jsonl (per-rollout sandbox start / verify
seconds and reward), gpu.json (peak torch memory), and the trainer output dir.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.harbor import HarborSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prdesc_env import PrDescBashEnv  # noqa: E402

TIMINGS: Path | None = None


class TimedPrDescBashEnv(PrDescBashEnv):
    """PrDescBashEnv that appends sandbox start/verify timings and the reward to env_timings.jsonl."""

    def reset(self, task_dir=None, **kwargs):
        t0 = time.time()
        out = super().reset(task_dir=task_dir, **kwargs)
        self._reset_s = time.time() - t0
        self._n_bash = 0
        return out

    def bash(self, command: str) -> str:
        """
        Run a shell command in the sandbox and return its combined stdout and stderr.

        Args:
            command: The shell command to run (working directory /repo).

        Returns:
            The command's combined stdout and stderr (truncated to 8000 characters).
        """
        self._n_bash += 1
        return super().bash(command)

    @property
    def reward(self) -> float:
        fresh = self._env is not None and not isinstance(self._reward, float)
        t0 = time.time()
        r = super().reward
        if fresh:
            parts = self._run(self._env.exec("cat /logs/verifier/reward.json")).stdout
            with TIMINGS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "reset_s": round(self._reset_s, 2),
                                    "verify_s": round(time.time() - t0, 2), "bash_calls": self._n_bash,
                                    "reward": r, "reward_json": parts.strip()}) + "\n")
        return r


class StepTimer(TrainerCallback):
    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        rec = {"step": state.global_step, "step_s": round(time.time() - self.t0, 1),
               "torch_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        print("STEP", json.dumps(rec), flush=True)
        with (Path(args.output_dir) / "steps.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    global TIMINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=12288)
    ap.add_argument("--vllm-max-model-length", type=int, default=16384)
    ap.add_argument("--max-tool-calling-iterations", type=int, default=30)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True in the chat template")
    ap.add_argument("--sleep-mode", action="store_true", help="vllm_enable_sleep_mode (offload vLLM while training)")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--save", action="store_true", help="save the trained model to rl/runs/<name>/final")
    args = ap.parse_args()

    name = args.name or f"{args.model.split('/')[-1]}-smoke"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    TIMINGS = out / "env_timings.jsonl"

    spec = HarborSpec(str(HERE / "harbor_tasks"), agent=TimedPrDescBashEnv, environment_type="docker")
    # One task: each step samples it once and runs num_generations rollouts of it (the trainer loops over epochs).
    config = GRPOConfig(
        output_dir=str(out / "trainer"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,  # generation batch = 1 prompt x num_generations
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        chat_template_kwargs={"enable_thinking": args.thinking},
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_enable_sleep_mode=args.sleep_mode,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": args.dtype},
        logging_steps=1,
        log_completions=True,
        save_strategy="no",
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=spec.train_dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=spec.reward_funcs,
        callbacks=[StepTimer()],
    )
    t0 = time.time()
    trainer.train()
    total = time.time() - t0
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")
    (out / "gpu.json").write_text(json.dumps({
        "train_seconds": round(total, 1),
        "torch_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "torch_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print("DONE", json.dumps({"train_seconds": round(total, 1)}), flush=True)
    if args.save:
        trainer.save_model(str(out / "final"))
        trainer.processing_class.save_pretrained(str(out / "final"))
        print("SAVED", out / "final", flush=True)


if __name__ == "__main__":
    main()
