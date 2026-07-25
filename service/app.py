"""
FastAPI service: submit a task to a loop pattern, poll for its result.

POST /tasks   -> {job_id, status: "pending"}, execution happens in a
                 background thread (service/jobs.py) so the request never
                 blocks on a multi-iteration loop run.
GET  /tasks/{job_id} -> current status, and the TaskResult once done.

Uses the same LOOP_FACTORIES registry and mock reasoning policy as
eval/run_comparison.py -- see eval/mock_agent.py's docstring for why
accuracy through this service isn't a meaningful benchmark signal until a
real LLM is wired into harness/llm.py.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from eval.run_comparison import LOOP_FACTORIES
from harness.contracts import Task
from service.jobs import JobStatus, JobStore, task_result_to_dict
from service.schemas import JobCreatedResponse, JobStatusResponse, TaskSubmitRequest

app = FastAPI(title="recur", description="Compare ReAct/Reflection/Plan-Execute agent loops")
job_store = JobStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks", response_model=JobCreatedResponse, status_code=202)
def submit_task(request: TaskSubmitRequest) -> JobCreatedResponse:
    loop_factory = LOOP_FACTORIES.get(request.loop)
    if loop_factory is None:
        raise HTTPException(status_code=400, detail=f"Unknown loop: {request.loop}")

    task = Task(
        id="api-" + request.question[:40],
        question=request.question,
        gold_answer=request.gold_answer,
        expected_hops=request.expected_hops,
    )
    job = job_store.create()
    submitted_status = job.status.value  # capture before starting the background
    # thread -- it may flip to RUNNING before this function returns
    job_store.run_in_background(job.id, loop_factory, task)
    return JobCreatedResponse(job_id=job.id, status=submitted_status)


@app.get("/tasks/{job_id}", response_model=JobStatusResponse)
def get_task(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        result=task_result_to_dict(job.result) if job.status == JobStatus.DONE else None,
        error=job.error,
    )
