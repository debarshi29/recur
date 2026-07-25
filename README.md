# recur

A production-grade, head-to-head comparison of three agentic control-loop
architectures — **ReAct**, **Reflection**, and **Plan-Execute** — built on
LangGraph and evaluated against the same multi-hop research/QA benchmark.

All three patterns share one harness: one `Task` definition, one `Tool`
interface, one retry/backoff policy, one scoring function. The only
variable under test is the control-loop architecture itself.

## Status

All 8 sprints complete: shared harness, an 18-question multi-hop
benchmark, all three loop patterns, a reliability layer (checkpointing,
circuit breaker, timeout), a FastAPI service, and a full comparison
writeup with per-pattern ADRs.

Loop reasoning currently runs against a deterministic mock LLM
(`harness/llm.py`, `eval/mock_agent.py`) since no live provider key is
configured -- see those modules' docstrings for what that does and
doesn't validate. Wiring a real provider is a single new `LLM` subclass.

## Results

See [`docs/writeup.md`](docs/writeup.md) for the full cross-pattern
comparison and per-task-shape recommendation, and `docs/adrs/` for the
per-pattern design decisions. Headline structural numbers from a real
measured run (mock-LLM policy -- see the writeup's caveat on why accuracy
isn't yet a meaningful signal):

| pattern | avg iterations/task | avg tool calls/task | avg tokens/task |
|---|---|---|---|
| ReAct | 2.00 | 1.00 | 343.4 |
| Reflection | 3.00 | 1.00 | 403.3 |
| Plan-Execute | 5.00 | 2.00 | 921.4 |

## Layout

```
harness/       shared contracts: Task, Tool, AgentLoop, ToolResult, TaskResult, reliability
loops/         one file per pattern: react_loop.py, reflection_loop.py, plan_execute_loop.py
eval/          QA dataset, comparison runner, CI regression gate
service/       FastAPI service: submit a task, poll for its result
tests/         unit + integration tests, run in CI on every PR
docs/adrs/     one ADR per loop pattern documenting observed trade-offs
```

## Development

```bash
uv venv .venv
uv pip install --python .venv -r requirements-dev.txt
.venv/Scripts/python -m pytest -q   # .venv/bin/python on macOS/Linux
```

## Running the service

```bash
docker build -t recur .
docker run -p 8000:8000 recur
```

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2+2?", "gold_answer": "4", "loop": "react"}'
# -> {"job_id": "...", "status": "pending"}

curl http://localhost:8000/tasks/<job_id>
```
