# GridWise — LLM-Assisted Campus Energy Optimizer

BUP CSE Fest 2026 · Hackathon · Online Preliminary

An HTTP service that reads natural-language campus operator notes with a language
model, converts them into machine-checkable directives, validates those directives
deterministically, and returns a cost-minimal, fully valid 24-hour energy schedule.

```
operator notes (English)
        │
        ▼
┌───────────────────┐   untrusted structured output
│  LLM interpreter  │──────────────────────┐
│  Groq → Gemini    │                      ▼
└───────────────────┘        ┌──────────────────────────┐
        │ provider failure   │  deterministic guardrails │  invalid → no_op
        ▼                    │  (app/directives.py)      │
┌───────────────────┐        └──────────────────────────┘
│ rule-based backup │──────────────────────┤
│  (safe failure)   │                      ▼
└───────────────────┘        ┌──────────────────────────┐
                             │  LP optimizer (HiGHS)     │
                             │  (app/optimizer.py)       │
                             └──────────────────────────┘
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │  self-replay verifier     │
                             │  (app/replay.py)          │
                             └──────────────────────────┘
                                          │
                                          ▼
                                 validated JSON response
```

---

## 1. Quickstart (clean environment)

Requires **Python 3.10+**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your keys into .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service is ready when `GET /health` answers. No build step, no training, no
database, no migrations.

### Verify health

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Verify the main endpoint

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

A successful call returns `200` with `scenario_id`, `directive_interpretation`
(one entry per note, in `note_index` order), `hourly_plan` (24 entries),
`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` and `plan_summary`.

---

## 2. Configuration

All configuration is environment variables. **No secret values are committed to
this repository**; `.env` is git-ignored and `.env.example` contains names only.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | recommended | — | Primary interpretation provider. Free tier at console.groq.com. |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` | Groq model id. |
| `GEMINI_API_KEY` | recommended | — | Fallback interpretation provider. Free tier at aistudio.google.com. |
| `GEMINI_MODEL` | no | `gemini-2.0-flash` | Gemini model id. |
| `LLM_TIMEOUT_SECONDS` | no | `8` | Per-provider HTTP timeout, kept well under the 30 s judging limit. |
| `LOG_LEVEL` | no | `INFO` | Standard Python log level. |
| `PORT` | no | `8000` | Port the container binds on `0.0.0.0`. |

`GET /` reports which providers are currently configured, without revealing any
key material.

### Model / provider disclosure

| Role | Provider · model |
|---|---|
| Primary operator-note interpretation | **Groq**, `llama-3.3-70b-versatile`, forced tool call, `temperature=0` |
| Fallback operator-note interpretation | **Google Gemini**, `gemini-2.0-flash`, `responseSchema` JSON mode, `temperature=0` |
| Optimization | **HiGHS** linear solver via `scipy.optimize.linprog` — no model involved |

Both providers are called over plain HTTPS with `httpx`, so no vendor SDK is
required at runtime.

---

## 3. Running the public sample cases

The repository ships a scoring harness that mirrors what the judge does: it
compares the structured interpretation against the published ground truth **and**
independently replays the returned schedule against that ground truth.

```bash
# in-process, no server needed
python -m tests.run_samples

# against a running or deployed service
python -m tests.run_samples --url http://localhost:8000
python -m tests.run_samples --url https://<your-deployment>
```

Expected result — all ten public cases pass both checks and match the organizer
reference cost exactly:

```
PASS SAMPLE-01  Solar cleaning + distractor        interp=ok plan=ok cost=38365.00 (+0.00 vs reference)
PASS SAMPLE-02  Battery charging maintenance       interp=ok plan=ok cost=42885.00 (+0.00 vs reference)
...
interpretation exact-match : 10/10
schedule valid vs truth    : 10/10
optimization quality       : 1.0000  -> 10.00/10 rubric points
```

Paraphrase robustness is tested separately, since hidden notes reword the same
directives:

```bash
python -m tests.test_robustness            # deterministic interpreter
LIVE_LLM=1 python -m tests.test_robustness # the configured model provider
```

---

## 4. How the pipeline works

### Stage 1 — LLM interpretation (`app/llm.py`) — *mandatory stage*

The model receives the operator notes plus the scenario's battery parameters
(needed to resolve phrasings such as "50% of battery capacity") and returns
structured candidate directives through a forced tool call / JSON schema. It is
asked for structure only; it never computes a schedule, a cost, or any number
that is not stated in the note.

The system prompt pins the three rules that hidden paraphrases hinge on:

* Windows are **start-inclusive, end-exclusive** — "6 PM until 9 PM" → `[18,19,20]`.
* `factor` is the fraction that **remains** — "an 80% reduction" → `0.2`.
* Battery percentages resolve against the scenario's `capacity_kwh`.

### Stage 2 — Deterministic guardrails (`app/directives.py`)

Model output is untrusted until every Problem Statement §08 check passes:
allowed directive type, one entry per note in `note_index` order, hours unique
integers 0–23 ascending, `0 ≤ factor ≤ 1`, reserve finite and within capacity,
grid cap finite and non-negative, and the exact `structured_adjustment` shape for
the chosen type. **A candidate that fails any check is downgraded to `no_op`**,
never repaired by guesswork — so a malformed model response can never invent a
constraint or crash the service.

### Stage 3 — Optimization (`app/optimizer.py`)

The validated directives become bounds and constraints in a linear program over
96 variables (grid, solar used, charge and discharge for each of 24 hours),
solved with HiGHS:

```
minimise   Σ tariff[h] · grid[h]
s.t.       grid[h] + solar[h] + discharge[h] − charge[h] = demand[h]
           min_level[h] ≤ E₀ + Σ_{k≤h}(charge[k] − discharge[k]) ≤ capacity
           Σ(charge[h] − discharge[h]) = 0                    (end-of-day neutrality)
           0 ≤ solar[h] ≤ effective_solar[h]                  (after solar_reduction)
           0 ≤ grid[h] ≤ max_grid_kwh[h]                      (max_grid_window)
           charge[h] = 0 in a no_charge_window
           discharge[h] = 0 in a no_discharge_window
```

An LP guarantees the true optimum, which is why the harness reports a cost
quality ratio of exactly 1.0000 against every public reference.

Because the challenge models no round-trip loss, the LP is degenerate: many
optima exist and a solver may split one hour across both charge and discharge. A
tiny throughput penalty (`1e-6`) selects the cleanest optimum, and the result is
netted to exactly one action per hour, well inside the 0.01 BDT tolerance.

### Stage 4 — Self-replay (`app/replay.py`)

Before responding, the service replays its own schedule the way the judge will:
24 unique hours, hourly energy balance, solar never above effective solar,
battery transitions, capacity and reserve bounds, charge/discharge rate limits,
`battery_kwh = 0` when idle, every directive obeyed, end-of-day neutrality, and
totals recomputed from `hourly_plan`. If the replay rejects a plan, the service
falls back to a schedule that is valid by construction rather than shipping one
the judge would reject.

### Safe failure

If **every** provider errors, times out, or returns unusable output, a
rule-based interpreter (`app/fallback.py`) takes over so the service returns a
controlled, valid answer instead of a 5xx. This is a backup path only — the
language model is the primary and required interpreter, and the fallback is never
the sole interpretation route.

---

## 5. API contract

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `200` · `{"status":"ok"}` |
| `POST /optimize-energy` | `200` with the interpretation + 24-hour plan |
| — | `400` malformed JSON or structurally invalid request |
| — | `500` controlled internal error, no stack trace, no secrets |
| `GET /` | service metadata and configured providers (diagnostic) |
| `GET /docs` | OpenAPI UI (diagnostic) |

---

## 6. Docker fallback

```bash
docker build -t gridwise-llm:1.0.0 .

docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  gridwise-llm:1.0.0

curl http://localhost:8000/health
# {"status":"ok"}
```

The image binds `0.0.0.0` on `$PORT` (default `8000`), ships a `HEALTHCHECK`, and
contains **no baked-in credentials** — keys are supplied at runtime with `-e`.

---

## 7. Dependencies and credits

| Package | Role |
|---|---|
| `fastapi` + `uvicorn` | HTTP service and ASGI server |
| `pydantic` v2 | Request/response schema validation |
| `scipy` (HiGHS) + `numpy` | Linear programming solver |
| `httpx` | Async HTTP client for both model providers |
| `python-dotenv` | Local `.env` loading (optional at runtime) |

External services: **Groq API** (Llama 3.3 70B) and **Google Gemini API**
(Gemini 2.0 Flash), both on free tiers. AI coding assistance was used during
development; the architecture, LP formulation, guardrail design and test harness
are the team's own work.

---

## 8. Known limitations

* **Provider dependency.** Interpretation quality depends on Groq/Gemini
  availability and free-tier quota. The deterministic fallback keeps the service
  answering, but with narrower paraphrase coverage than the model.
* **Fallback coverage.** `app/fallback.py` handles the phrasings in
  `tests/test_robustness.py`; unusual wording outside that range resolves to
  `no_op` rather than risking an invented constraint.
* **Ambiguous bare-hour ranges.** A note such as "from one until three" with no
  meridiem is resolved by context heuristics in the fallback path; the model
  handles these more reliably.
* **Infeasible inputs.** Organizer scoring scenarios are promised feasible. For
  contradictory input the optimizer relaxes constraints in a fixed order (grid
  caps → raised reserves → no-discharge → no-charge → neutrality) and, in the
  last resort, returns an idle-battery baseline, reporting what was relaxed in
  `plan_summary` rather than failing the request.
* **No grid export.** Surplus solar is curtailed, per the Problem Statement.

## 9. Secret handling

* `.env` is git-ignored; only `.env.example` (names, no values) is committed.
* No key, token, prompt containing a key, or raw stack trace is written to logs
  or returned in any API response; provider errors are logged by exception type
  only.
* The Docker image contains no credentials; they are injected at runtime.
