"""
Shared job state — tracks running subprocesses so tools can poll them.

Training jobs are long-running (10-60 min). We launch them as background
subprocesses, store their state here, and let Claude poll via
get_training_status(job_id) rather than blocking the MCP tool call.
"""

import json
import time
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

STATE_FILE = Path("./outputs/.job_state.json")


@dataclass
class JobState:
    job_id: str
    status: str                  # queued | running | completed | failed
    script: str
    args: list[str]
    pid: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    log_path: Optional[str] = None
    output_dir: Optional[str] = None
    error_msg: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class JobRegistry:
    """Persist job state to disk so it survives server restarts."""

    def __init__(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = self._load()
        self._procs: dict[str, subprocess.Popen] = {}

    def _load(self) -> dict[str, JobState]:
        if STATE_FILE.exists():
            try:
                raw = json.loads(STATE_FILE.read_text())
                return {k: JobState(**v) for k, v in raw.items()}
            except Exception:
                return {}
        return {}

    def _save(self):
        STATE_FILE.write_text(
            json.dumps({k: asdict(v) for k, v in self._jobs.items()}, indent=2)
        )

    def create(self, job_id: str, script: str, args: list[str],
               log_path: str, output_dir: str) -> JobState:
        job = JobState(
            job_id=job_id,
            status="queued",
            script=script,
            args=args,
            log_path=log_path,
            output_dir=output_dir,
        )
        self._jobs[job_id] = job
        self._save()
        return job

    def start(self, job_id: str, proc: subprocess.Popen):
        job = self._jobs[job_id]
        job.pid = proc.pid
        job.status = "running"
        job.started_at = time.time()
        self._procs[job_id] = proc
        self._save()

    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def refresh(self, job_id: str) -> Optional[JobState]:
        """Poll the subprocess and update status if it has finished."""
        job = self._jobs.get(job_id)
        if not job or job.status in ("completed", "failed"):
            return job

        proc = self._procs.get(job_id)
        if proc is not None:
            ret = proc.poll()
            if ret is not None:
                job.finished_at = time.time()
                job.status = "completed" if ret == 0 else "failed"
                if ret != 0:
                    job.error_msg = f"Process exited with code {ret}"
                self._save()
        return job

    def list_all(self) -> list[JobState]:
        return list(self._jobs.values())


# Module-level singleton — all tools share this registry
registry = JobRegistry()
