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

Loop reasoning can run against either a deterministic mock LLM
(`eval/mock_agent.py`, no API key needed -- validates control-flow
mechanism, not accuracy) or a real provider, `GroqLLM`
(`harness/llm.py`, requires `GROQ_API_KEY` -- see `.env.example`).
`python -m eval.run_comparison --llm mock|groq` selects between them.

## Results

See [`docs/writeup.md`](docs/writeup.md) for the full cross-pattern
comparison, the real-LLM run's caveats (scorer strictness, Reflection's
convergence behavior), and per-task-shape recommendation; `docs/adrs/`
has the per-pattern design decisions.

Real-LLM run (Groq `llama-3.3-70b-versatile`, `exact_match_scorer` --
see the writeup for why this is a lower bound on accuracy):

| pattern | accuracy | avg iterations/task | avg tool calls/task | avg tokens/task |
|---|---|---|---|---|
| ReAct | 27.8% | 2.06 | 1.06 | 560.4 |
| Reflection | 0.0% | 5.22 | 1.44 | 1661.0 |
| Plan-Execute | 11.1% | 4.39 | 2.33 | 1771.3 |

Mock-policy run (mechanism validation only -- accuracy isn't meaningful
here, see the writeup):

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
