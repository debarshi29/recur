from __future__ import annotations

from eval.qa_dataset import TASKS


def test_dataset_has_15_to_20_questions():
    assert 15 <= len(TASKS) <= 20


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_every_task_has_nonempty_question_and_gold_answer():
    for task in TASKS:
        assert task.question.strip()
        assert task.gold_answer.strip()
        assert task.expected_hops >= 1


def test_at_least_three_questions_require_multi_hop_reasoning():
    multi_hop = [t for t in TASKS if t.expected_hops >= 2]
    assert len(multi_hop) >= 3
