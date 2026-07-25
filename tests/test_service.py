from __future__ import annotations

import time

from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_submit_unknown_loop_returns_400():
    response = client.post("/tasks", json={"question": "q", "loop": "not_a_real_loop"})
    assert response.status_code == 422  # pydantic Literal rejects it before it reaches the handler


def test_get_unknown_job_returns_404():
    response = client.get("/tasks/does-not-exist")
    assert response.status_code == 404


def _poll_until_done(job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        response = client.get(f"/tasks/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_submit_and_poll_react_task_to_completion():
    submit_response = client.post(
        "/tasks",
        json={"question": "What is 2+2?", "gold_answer": "4", "loop": "react"},
    )
    assert submit_response.status_code == 202
    body = submit_response.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]

    final = _poll_until_done(job_id)
    assert final["status"] == "done"
    assert final["result"]["loop_name"] == "react"
    assert "predicted_answer" in final["result"]


def test_submit_and_poll_each_loop_pattern():
    for loop_name in ("react", "reflection", "plan_execute"):
        submit_response = client.post(
            "/tasks",
            json={"question": "What is 2+2?", "gold_answer": "4", "loop": loop_name},
        )
        assert submit_response.status_code == 202
        job_id = submit_response.json()["job_id"]

        final = _poll_until_done(job_id)
        assert final["status"] == "done", final
        assert final["result"]["loop_name"] == loop_name
