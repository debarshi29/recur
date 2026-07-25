from __future__ import annotations

import pytest

from eval.run_comparison import run_all
from harness.contracts import AgentLoop, ScoringFn, Task, TaskResult, exact_match_scorer


class _StubLoop(AgentLoop):
    """Answers every task correctly without touching tools/LLMs -- exists
    purely to exercise run_comparison's plumbing (loop x task iteration,
    tracker logging) independent of any real reasoning loop."""

    name = "stub"

    def run(self, task: Task, scorer: ScoringFn = exact_match_scorer) -> TaskResult:
        correct, score = scorer(task.gold_answer, task.gold_answer)
        return TaskResult(
            task_id=task.id,
            loop_name=self.name,
            predicted_answer=task.gold_answer,
            correct=correct,
            score=score,
            iterations=1,
            tool_calls=0,
            total_tokens=0,
            wall_clock_ms=0.0,
        )


@pytest.fixture(autouse=True)
def _mlflow_local_tracking(tmp_path, monkeypatch):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-run-comparison")


def test_run_all_raises_without_registered_loops():
    with pytest.raises(RuntimeError):
        run_all(loop_factories={})


def test_run_all_runs_every_loop_over_every_task():
    tasks = [
        Task(id="t1", question="q1", gold_answer="a1"),
        Task(id="t2", question="q2", gold_answer="a2"),
    ]
    results = run_all(loop_factories={"stub": lambda: _StubLoop(tools=[])}, tasks=tasks)

    assert len(results) == 2
    assert {r.task_id for r in results} == {"t1", "t2"}
    assert all(r.loop_name == "stub" and r.correct for r in results)
