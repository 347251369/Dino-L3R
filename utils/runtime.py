from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


def command_text(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def run_command(
    command: list[str],
    *,
    cwd: str | Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[command] {command_text(command)}", flush=True)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=str(cwd), env=merged_env, check=True)


def require_count(directory: str | Path, pattern: str, expected: int) -> list[Path]:
    paths = sorted(Path(directory).glob(pattern))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} files matching {pattern} in {directory}, found {len(paths)}"
        )
    return paths
