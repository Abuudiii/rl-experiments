"""TRL/Harbor harness for the logslice PR-description task: one `bash` tool, no answer file.

TRL's built-in `HarborBashEnv` tells the model that data is in /home/user/input and that it must submit by writing
/workdir/answer.txt (its data-analysis task suite). Neither is true here: the task is "address the review and push",
graded from the final /repo and the fake-GitHub PR state. So this harness only exposes `bash` and says how to finish.

Used as `HarborSpec(..., agent="rl/prdesc_env.py:PrDescBashEnv")`.
"""

from trl.experimental.harbor import HarborEnv

PROMPT_SUFFIX = (
    "\n\nYou have a single `bash` tool: it runs one shell command in the sandbox (working directory /repo, "
    "shell state is not kept between calls) and returns its stdout+stderr. When you have finished the task, "
    "reply with a short summary and no tool call."
)


class PrDescBashEnv(HarborEnv):
    PROMPT_SUFFIX = PROMPT_SUFFIX

    def bash(self, command: str) -> str:
        """
        Run a shell command in the sandbox and return its combined stdout and stderr.

        Args:
            command: The shell command to run (working directory /repo).

        Returns:
            The command's combined stdout and stderr (truncated to 8000 characters).
        """
        return self._exec(command, timeout=120)

    async def _setup(self) -> None:
        # HarborEnv._start re-uploads every file COPY'd by the Dockerfile (a workaround for E2B builds). On docker that
        # overwrites the fake `gh` with a copy whose mode comes from the host file and re-creates /opt/task-setup,
        # which the image build deleted. Restore the image state; also drop the dirs HarborEnv creates for its
        # data-analysis suite.
        await self._env.exec(
            "chmod 755 /usr/local/bin/gh && rm -rf /opt/task-setup /workdir /home/user/input", user="root"
        )
