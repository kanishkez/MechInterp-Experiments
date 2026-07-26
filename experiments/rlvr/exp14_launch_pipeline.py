"""
Launch exp14 v5 as a two-stage pipeline:
1. rebuild the GSM8K quadrants with the corrected HF decode pipeline
2. run the corrected trajectory patching sweep against those quadrants

This script is intended to be started from a live marimo notebook session.
It runs as a background process and writes durable logs and checkpoints into
/marimo so progress can be verified later.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

MARIMO_DIR = Path("/marimo")
PIPELINE_LOG = MARIMO_DIR / "exp14_pipeline.log"
REBUILD_LOG = MARIMO_DIR / "exp14_rebuild.log"
V5_LOG = MARIMO_DIR / "exp14_v5.log"
REBUILD_PID = MARIMO_DIR / "exp14_rebuild.pid"
V5_PID = MARIMO_DIR / "exp14_v5.pid"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a") as f:
        f.write(line + "\n")


def launch(cmd: list[str], log_path: Path, cwd: str = "/marimo") -> subprocess.Popen:
    log(f"Launching: {' '.join(cmd)}")
    log_file = log_path.open("w")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )


def write_pid(path: Path, pid: int) -> None:
    path.write_text(f"{pid}\n")


def main() -> int:
    log("Starting exp14 pipeline")

    rebuild = launch(["python3", "/marimo/exp14_rebuild_v2.py"], REBUILD_LOG)
    write_pid(REBUILD_PID, rebuild.pid)
    log(f"Rebuild PID: {rebuild.pid}")

    rebuild_rc = rebuild.wait()
    log(f"Rebuild exit code: {rebuild_rc}")
    if rebuild_rc != 0:
        log("Rebuild failed; stopping pipeline")
        return rebuild_rc

    v5 = launch(["python3", "/marimo/run_exp14_v5_corrected.py"], V5_LOG)
    write_pid(V5_PID, v5.pid)
    log(f"v5 PID: {v5.pid}")

    v5_rc = v5.wait()
    log(f"v5 exit code: {v5_rc}")
    return v5_rc


if __name__ == "__main__":
    raise SystemExit(main())
