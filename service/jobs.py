"""
In-memory job store backing the FastAPI service. A submitted task runs a
loop pattern in a background thread so the HTTP request that submitted it
returns immediately with a job id -- a multi-iteration research loop can
take much longer than a single request should block on -- and the caller
polls GET /tasks/{job_id} for status/result.

This is intentionally a plain in-memory dict, not a queue/worker system:
the project's scope is demonstrating the service shape (submit, poll,
async execution), not building production job infrastructure. Swapping in
a real queue (Celery, RQ, ...) would replace this module without touching
the loop or harness layers.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

from harness.contracts import AgentLoop, Task, TaskResult


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    result: TaskResult | None = None
    error: str | None = None


class JobStore:
    """Thread-safe in-memory job registry. One instance is shared across
    all requests in the FastAPI app's lifetime."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)

    def run_in_background(self, job_id: str, loop_factory: Callable[[], AgentLoop], task: Task) -> None:
        def target() -> None:
            self._update(job_id, status=JobStatus.RUNNING)
            try:
                loop = loop_factory()
                result = loop.run(task)
                self._update(job_id, status=JobStatus.DONE, result=result)
            except Exception as e:  # noqa: BLE001 — surface as a job error, never crash the server
                self._update(job_id, status=JobStatus.ERROR, error=f"{type(e).__name__}: {e}")

        threading.Thread(target=target, daemon=True).start()


def task_result_to_dict(result: TaskResult) -> dict[str, Any]:
    return asdict(result)
