"""
MLflow logging for loop runs. This is what turns individual TaskResults
into a comparable eval history across loop patterns over time — the
through-line for observability across the project.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlflow

from harness.contracts import TaskResult


def configure(tracking_uri: str = "file:./mlruns", experiment: str = "loop-engineering"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


@contextmanager
def track_run(loop_name: str, task_id: str, run_config: dict):
    with mlflow.start_run(run_name=f"{loop_name}-{task_id}") as run:
        mlflow.set_tags({"loop": loop_name, "task_id": task_id})
        mlflow.log_params(run_config)
        yield run


def log_result(result: TaskResult) -> None:
    mlflow.log_metrics(
        {
            "correct": float(result.correct),
            "score": result.score,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "total_tokens": result.total_tokens,
            "wall_clock_ms": result.wall_clock_ms,
        }
    )
    mlflow.log_dict({"trace": result.trace}, f"traces/{result.task_id}.json")


def log_comparison_summary(results: list[TaskResult]) -> None:
    """Call once after running all loops over all tasks; logs an
    aggregate comparison table as a run-level artifact."""
    by_loop: dict[str, list[TaskResult]] = {}
    for r in results:
        by_loop.setdefault(r.loop_name, []).append(r)

    summary = {}
    for loop_name, rs in by_loop.items():
        n = len(rs)
        summary[loop_name] = {
            "accuracy": sum(r.correct for r in rs) / n,
            "avg_score": sum(r.score for r in rs) / n,
            "avg_iterations": sum(r.iterations for r in rs) / n,
            "avg_tool_calls": sum(r.tool_calls for r in rs) / n,
            "avg_tokens": sum(r.total_tokens for r in rs) / n,
            "avg_wall_clock_ms": sum(r.wall_clock_ms for r in rs) / n,
            "n_tasks": n,
        }

    with mlflow.start_run(run_name="comparison-summary"):
        mlflow.log_dict(summary, "comparison_summary.json")
