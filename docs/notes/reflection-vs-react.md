# Reflection vs. ReAct — structural comparison note

Against the mock reasoning policy (`eval/mock_agent.py`), Reflection averages
3.0 LLM calls/task vs. ReAct's 2.0, and ~403 tokens/task vs. ~343 — the
expected cost of the extra critique step on every round. Both loops used an
identical single search-then-answer strategy under the mock policy, so this
run validates the *mechanism* (draft → critique → accept/revise, at higher
cost per task) rather than a real accuracy difference; a real accuracy
comparison needs a real LLM wired into `harness/llm.py` (see Sprint 7).
