from __future__ import annotations

import pytest

import eval.check_regression as check_regression
from harness.contracts import AgentLoop, ScoringFn, Task, TaskResult, exact_match_scorer


class _StubLoop(AgentLoop):
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
    mlflow.set_experiment("test-check-regression")
    monkeypatch.setattr(check_regression.tracker, "configure", lambda: None)


def test_main_passes_when_every_loop_completes_every_task(monkeypatch):
    monkeypatch.setattr(check_regression, "LOOP_FACTORIES", {"stub": lambda: _StubLoop(tools=[])})
    monkeypatch.setattr(check_regression, "TASKS", [Task(id="t1", question="q", gold_answer="a")])

    assert check_regression.main() == 0
