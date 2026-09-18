# GridWise — LLM-Assisted Campus Energy Optimizer

BUP CSE Fest 2026 · Hackathon · Online Preliminary

An HTTP service that reads natural-language campus operator notes with a language
model, converts them into machine-checkable directives, validates those directives
deterministically, and returns a cost-minimal, fully valid 24-hour energy schedule.

**Deployed judge endpoint**: `https://gridwise-llm-production-6051.up.railway.app`
(`GET /health`, `POST /optimize-energy`) — verified live against both LLM providers
and all 10 public sample cases; see §4.

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
  -d @sample_request_small.json
```

A successful call returns `200` with `scenario_id`, `directive_interpretation`
(one entry per note, in `note_index` order), `hourly_plan` (24 entries),
`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` and `plan_summary`. A worked
example is in the next section; a second request built from the public sample
pack is at [`sample_request.json`](sample_request.json).

---

## 2. Sample request / response

The full request below is [`sample_request_small.json`](sample_request_small.json) —
send it exactly as shown with `curl -X POST .../optimize-energy -d @sample_request_small.json`.
The response is the service's actual live output (`hourly_plan` is 24 entries;
representative hours are shown here, with `...` marking the rest).

**Request**

```json
{
  "scenario_id": "README-DEMO",
  "operator_notes": [
    "Solar output will drop to about 20% from 1 PM to 3 PM.",
    "Keep at least 50 kWh in the battery from 6 PM until 9 PM.",
    "The cafeteria menu changes tomorrow."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
    "...": "21 more hourly entries",
    {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
  ],
  "battery": {
    "capacity_kwh": 220,
    "initial_energy_kwh": 110,
    "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50,
    "max_discharge_kwh_per_hour": 50
  }
}
```

**Response** (`200`, produced by the live service — `openai/gpt-oss-120b` via Groq)

```json
{
  "scenario_id": "README-DEMO",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
      "explanation": "Solar output reduced to 20% of normal between 1 PM and 3 PM."
    },
    {
      "note_index": 1,
      "applies": true,
      "directive_type": "minimum_battery_reserve",
      "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 50.0},
      "explanation": "Maintain at least 50 kWh in the battery from 6 PM to 9 PM."
    },
    {
      "note_index": 2,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "The note concerns a cafeteria menu change and does not affect today's electricity schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0},
    {"hour": 1, "grid_kwh": 45.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0},
    "...": "20 more hourly entries",
    {"hour": 18, "grid_kwh": 155.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 50.0, "battery_energy_after_kwh": 150.0},
    {"hour": 19, "grid_kwh": 165.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 50.0, "battery_energy_after_kwh": 100.0},
    {"hour": 23, "grid_kwh": 155.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 50.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2678.0,
  "total_cost_bdt": 38000.0,
  "peak_grid_kwh": 192.0,
  "plan_summary": "Charged the battery in 7 cheap hours and discharged it in 9 expensive hours, using solar first and returning the battery to its starting energy by the end of hour 23. Applied reduced solar availability, a raised battery reserve from the operator notes. Total grid import 2678.00 kWh at a cost of 38000.00 BDT, peaking at 192.00 kWh."
}
```

Note hour 23's `battery_energy_after_kwh` (110.0) exactly equals the request's
`initial_energy_kwh` — end-of-day neutrality holds — and hour 18-20 stay at or
above the 50 kWh reserve the second note requested.

To regenerate this example against your own running service:
`python -m tests.make_readme_sample` (writes `tests/_readme_response.json`).

---

## 3. Configuration

All configuration is environment variables. **No secret values are committed to
this repository**; `.env` is git-ignored and `.env.example` contains names only.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | recommended | — | Primary interpretation provider. Free tier at console.groq.com. |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b` | Groq model id. |
| `GEMINI_API_KEY` | recommended | — | Fallback interpretation provider. Free tier at aistudio.google.com. |
| `GEMINI_MODEL` | no | `gemini-3.5-flash-lite` | Gemini model id. |
| `LLM_TIMEOUT_SECONDS` | no | `8` | Per-provider HTTP timeout, kept well under the 30 s judging limit. |
| `LOG_LEVEL` | no | `INFO` | Standard Python log level. |
| `PORT` | no | `8000` | Port the container binds on `0.0.0.0`. |

`GET /` reports which providers are currently configured, without revealing any
key material.

### Model / provider disclosure

| Role | Provider · model |
|---|---|
| Primary operator-note interpretation | **Groq**, `openai/gpt-oss-120b`, forced tool call, `temperature=0` |
| Fallback operator-note interpretation | **Google Gemini**, `gemini-3.5-flash-lite`, `responseSchema` JSON mode, `temperature=0` |
| Optimization | **HiGHS** linear solver via `scipy.optimize.linprog` — no model involved |

Both providers are called over plain HTTPS with `httpx`, so no vendor SDK is
required at runtime.

---

## 4. Running the public sample cases

The repository ships a scoring harness that mirrors what the judge does: it
compares the structured interpretation against the published ground truth **and**
independently replays the returned schedule against that ground truth.

```bash
# in-process, no server needed
python -m tests.run_samples

# against a running or deployed service
python -m tests.run_samples --url http://localhost:8000
python -m tests.run_samples --url https://gridwise-llm-production-6051.up.railway.app
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

## 5. How the pipeline works

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

### Stage 1b — Window-end normalisation (`app/service.py`)

Time windows are start-inclusive, end-exclusive, but a model occasionally
returns one hour too many on phrasing like "7 PM through 10 PM" (English often
reads "through" as inclusive). A deterministic parser — the same one used by
the safe-failure fallback — re-derives the window from the note text; where it
agrees with the model on the window's start but disagrees on the end, the
parser's boundary wins. If it disagrees on the start too, that is a different
reading of the note, not a boundary slip, and the model's answer is left alone.
This is normalisation of a time convention, not reinterpretation of the note.

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

## 6. API contract

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `200` · `{"status":"ok"}` |
| `POST /optimize-energy` | `200` with the interpretation + 24-hour plan |
| — | `400` malformed JSON or structurally invalid request |
| — | `500` controlled internal error, no stack trace, no secrets |
| `GET /` | service metadata and configured providers (diagnostic) |
| `GET /docs` | OpenAPI UI (diagnostic) |

---

## 7. Docker fallback

The image is built and published automatically by
[`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml)
on every push to `main` — no local Docker install is required to reproduce it.
It is public on GHCR and pullable without any credentials.

**Registry image reference**

```
ghcr.io/faiyadhoque/elite-ball-knowledge-bup-quali:1.0.0
ghcr.io/faiyadhoque/elite-ball-knowledge-bup-quali@sha256:4a457d9517d64431ca28b2b86416f15f6a6b86f242d9ef7c2e5e87cf617ffbe9
```

**Verified `docker run` command**

```bash
docker pull ghcr.io/faiyadhoque/elite-ball-knowledge-bup-quali:1.0.0

docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  ghcr.io/faiyadhoque/elite-ball-knowledge-bup-quali:1.0.0

curl http://localhost:8000/health
# {"status":"ok"}
```

The image binds `0.0.0.0` on `$PORT` (default `8000`), ships a `HEALTHCHECK`,
and contains **no baked-in credentials** — keys are supplied at runtime with
`-e`. The CI workflow itself runs this exact `docker run` + `/health` sequence
against the freshly pushed image before ever reporting success, so a broken
image cannot pass silently.

To build locally instead (once Docker is available on your machine):

```bash
docker build -t gridwise-llm:1.0.0 .
docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  gridwise-llm:1.0.0
```

---

## 8. Dependencies and credits

| Package | Role |
|---|---|
| `fastapi` + `uvicorn` | HTTP service and ASGI server |
| `pydantic` v2 | Request/response schema validation |
| `scipy` (HiGHS) + `numpy` | Linear programming solver |
| `httpx` | Async HTTP client for both model providers |
| `python-dotenv` | Local `.env` loading (optional at runtime) |

External services: **Groq API** (`openai/gpt-oss-120b`) and **Google Gemini API**
(Gemini 3.5 Flash Lite), both on free tiers. AI coding assistance was used during
development; the architecture, LP formulation, guardrail design and test harness
are the team's own work.

---

## 9. Known limitations

* **Provider dependency and free-tier quota.** Groq's free tier allows roughly
  8000 tokens per minute, and one request costs about 1400, so sustained bursts
  beyond ~5 requests/minute spill over to Gemini. That is by design: the
  provider chain absorbs it and the schedule is unaffected. The deterministic
  fallback keeps the service answering if both providers are exhausted, with
  narrower paraphrase coverage than either model.
* **Model availability shifts.** `llama-3.3-70b-versatile` and
  `gemini-2.5-flash` were both withdrawn during development. The pinned ids
  below are verified working; `tests/check_providers.py --models` lists what an
  account can currently reach if one is retired again.
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

## 10. Secret handling

* `.env` is git-ignored; only `.env.example` (names, no values) is committed.
* No key, token, prompt containing a key, or raw stack trace is written to logs
  or returned in any API response; provider errors are logged by exception type
  only.
* The Docker image contains no credentials; they are injected at runtime.
