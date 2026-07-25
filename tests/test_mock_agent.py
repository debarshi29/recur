from __future__ import annotations

from eval.mock_agent import policy


def test_policy_accepts_reflection_critique_unconditionally():
    prompt = "Question: q\nCandidate answer: 4\n\nCritique the candidate answer..."
    assert policy(prompt) == "GOOD"


def test_policy_emits_fixed_plan_for_plan_stage():
    prompt = "Question: q\n\nProduce a short plan as 'PLAN: step1 | step2 | ...'"
    assert policy(prompt).startswith("PLAN:")


def test_policy_searches_on_first_execute_call():
    prompt = "Question: q\n\nCurrent step: search for it\n\nHistory for this step:\n(none yet)"
    assert policy(prompt).startswith("ACTION: web_search")


def test_policy_marks_step_done_on_non_final_step_with_observation():
    prompt = (
        "Question: q\n\nCurrent step: search for it\n\n"
        "History for this step:\nThought/Action: ACTION: web_search | {}\nObservation: some text"
    )
    assert policy(prompt).startswith("STEP_DONE:")


def test_policy_answers_on_final_step_with_observation():
    prompt = (
        "Question: q\n\nCurrent step (final step): answer it\n\n"
        "History for this step:\nThought/Action: ACTION: web_search | {}\nObservation: some text"
    )
    assert policy(prompt).startswith("FINAL_ANSWER:")


def test_policy_falls_back_on_replan_stage():
    prompt = "Question: q\n\nThe current step failed: bad step\nReason given: broke"
    assert policy(prompt) == "FINAL_ANSWER: unresolved"
