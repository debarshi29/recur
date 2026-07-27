# recur

A head-to-head comparison of three agentic control-loop architectures —
**ReAct**, **Reflection**, and **Plan-Execute** — built on LangGraph and
evaluated against the same multi-hop research/QA benchmark.

All three patterns share one harness: one `Task` definition, one `Tool`
interface, one retry/circuit-breaker/timeout policy, one scoring
function. The only variable under test is the control-loop architecture
itself — if a loop needed its own tools or its own reliability logic to
work, that would be a bug in the comparison, not a feature of the
pattern.

## Status

All 8 sprints complete: shared harness, an 18-question multi-hop
benchmark, all three loop patterns, a reliability layer (checkpointing,
circuit breaker, timeout), a FastAPI service, and a full comparison
writeup with per-pattern ADRs. 70 tests passing; CI runs the suite plus a
crash-free regression gate on every push/PR to `main`.

Loop reasoning can run against either a deterministic mock LLM
(`eval/mock_agent.py`, no API key needed — validates control-flow
mechanism, not accuracy) or a real provider, `GroqLLM`
(`harness/llm.py`, requires `GROQ_API_KEY`). `python -m
eval.run_comparison --llm mock|groq` selects between them; CI always uses
`mock` so it stays deterministic and API-key-free.

**Open follow-ups** (see `docs/writeup.md` for the findings behind them):
`exact_match_scorer` under-counts substantively-correct free-text answers
and needs a paraphrase-tolerant replacement, and Reflection's critique
step needs a convergence bound so it stops nitpicking correct drafts
indefinitely.

## Quickstart

```bash
uv venv .venv
uv pip install --python .venv -r requirements-dev.txt
.venv/Scripts/python -m pytest -q   # .venv/bin/python on macOS/Linux

# structural comparison sweep, no API key needed
.venv/Scripts/python -m eval.run_comparison --llm mock

# real accuracy comparison -- requires GROQ_API_KEY
cp .env.example .env   # then fill in GROQ_API_KEY
.venv/Scripts/python -m eval.run_comparison --llm groq

# inspect either run
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Results

See [`docs/writeup.md`](docs/writeup.md) for the full cross-pattern
comparison, the real-LLM run's caveats (scorer strictness, Reflection's
convergence behavior), and per-task-shape recommendations; `docs/adrs/`
has the per-pattern design decisions and trade-offs.

**Real-LLM run** (Groq `llama-3.3-70b-versatile`, `exact_match_scorer` —
see the writeup for why this is a lower bound on accuracy):

| pattern | accuracy | avg iterations/task | avg tool calls/task | avg tokens/task |
|---|---|---|---|---|
| ReAct | 27.8% | 2.06 | 1.06 | 560.4 |
| Reflection | 0.0% | 5.22 | 1.44 | 1661.0 |
| Plan-Execute | 11.1% | 4.39 | 2.33 | 1771.3 |

**Mock-policy run** (mechanism validation only — accuracy isn't
meaningful here, see the writeup):

| pattern | avg iterations/task | avg tool calls/task | avg tokens/task |
|---|---|---|---|
| ReAct | 2.00 | 1.00 | 343.4 |
| Reflection | 3.00 | 1.00 | 403.3 |
| Plan-Execute | 5.00 | 2.00 | 921.4 |

## Layout

```
harness/       shared contracts: Task, Tool, AgentLoop, ToolResult, TaskResult, LLM, reliability
loops/         one file per pattern: react_loop.py, reflection_loop.py, plan_execute_loop.py, common.py
eval/          QA dataset, local corpus, comparison runner, mock LLM policy, CI regression gate
service/       FastAPI service: submit a task, poll for its result
tests/         unit + integration tests, run in CI on every push/PR
docs/adrs/     one ADR per loop pattern documenting observed trade-offs
docs/writeup.md   full cross-pattern comparison, mock-run and real-LLM-run results
```

## Development

```bash
uv venv .venv
uv pip install --python .venv -r requirements-dev.txt
.venv/Scripts/python -m pytest -q                 # .venv/bin/python on macOS/Linux
.venv/Scripts/python -m ruff check .
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
