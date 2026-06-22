# Causal LLM Evaluator

A FastAPI microservice that answers one question: **does changing my prompt actually cause better outputs, or did I just get lucky?**

Most LLM evaluation reports averages. You run prompt A ten times, prompt B ten times, and conclude "B is better" because it scored higher. But that's correlation — you can't tell if the prompt *caused* the improvement or if you got lucky with the sample.

This evaluator treats prompt A/B testing as a proper randomised experiment:

- **p-value** — is the difference statistically real, or noise?
- **Cohen's d** — is the effect size practically meaningful?
- **95% bootstrap CI** — what's the plausible range of true effects?

---

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Pydantic v2 |
| LLM calls | Anthropic SDK / Groq SDK |
| Statistics | SciPy (Welch's t-test) + NumPy (bootstrap CI) |
| Scoring | LLM-as-judge (Haiku / Llama) or ROUGE-L |
| Persistence | SQLite + JSONL experiment log |
| Dashboard | Streamlit (forest plot) |
| Packaging | uv |

---

## Setup

```bash
git clone <repo>
cd causal-llm-evaluator

# install dependencies
uv sync --no-install-project

# configure keys
cp .env.example .env
# edit .env — add ANTHROPIC_API_KEY and/or GROQ_API_KEY
```

---

## Running

**Terminal 1 — API server:**
```bash
uv run uvicorn app.main:app --reload
```

**Terminal 2 — Dashboard:**
```bash
uv run streamlit run frontend/dashboard.py
```

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

---

## API

### `POST /experiment`

Run a randomised prompt A/B test.

```json
{
  "variants": [
    {
      "id": "control",
      "system_prompt": "You are a helpful assistant.",
      "user_template": "Summarise: {input}"
    },
    {
      "id": "chain_of_thought",
      "system_prompt": "Think step by step before answering.",
      "user_template": "Summarise: {input}"
    }
  ],
  "test_inputs": ["...text 1...", "...text 2..."],
  "n_samples": 30,
  "scorer": "llm_judge",
  "provider": "anthropic"
}
```

**Response:**
```json
{
  "experiment_id": "a3f7c2b1",
  "control_id": "control",
  "effects": [
    {
      "variant_id": "chain_of_thought",
      "ate": 0.087,
      "ci_lower": 0.041,
      "ci_upper": 0.134,
      "p_value": 0.003,
      "cohens_d": 0.61,
      "significant": true
    }
  ],
  "winner": "chain_of_thought",
  "interpretation": "'chain_of_thought' best variant (ATE=+0.087, Cohen's d=0.61, p=0.003). 95% CI: [0.041, 0.134]. Medium practical effect."
}
```

### `GET /results/{exp_id}`

Fetch a past experiment by ID.

### `GET /results`

List all past experiments.

### `GET /health`

Liveness check.

---

## Providers

| Field | Default | Options |
|---|---|---|
| `provider` | `"anthropic"` | `"anthropic"`, `"groq"` |
| `model` | provider default | any model string |
| `judge_provider` | same as `provider` | `"anthropic"`, `"groq"` |
| `judge_model` | provider default | any model string |

**Provider defaults:**

| Provider | Generation model | Judge model |
|---|---|---|
| `anthropic` | `claude-opus-4-8` | `claude-haiku-4-5` |
| `groq` | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` |

**Mix providers** — Groq for generation (fast/cheap), Anthropic for judging (higher quality):
```json
{
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "judge_provider": "anthropic",
  "judge_model": "claude-haiku-4-5"
}
```

---

## Scorers

| `scorer` value | Method | Notes |
|---|---|---|
| `"llm_judge"` | LLM rates output 1–10, normalised to 0–1 | Default. Domain-aware rubric. |
| `"rouge"` | ROUGE-L F1 vs input as reference | Fast, no API cost. Best for summarisation. |

---

## Demo — Primetag Creator Analysis

`demo/creator_experiment.json` runs four prompt variants against three real creator profiles:

| Variant | Strategy |
|---|---|
| `control_minimal` | Bare instruction — no structure, no persona |
| `structured_json` | Explicit 4-dimension rubric (audience fit, content quality, brand safety, growth signal) |
| `chain_of_thought` | Step-by-step reasoning before recommendation |
| `persona_talent_buyer` | Luxury brand buyer persona — €2M budget, 90% pass rate |

**Run it:**
```bash
# via API docs at /docs — paste demo/creator_experiment.json body
# or via dashboard — enter path in the input box and click Run
```

Each experiment is saved to `experiments.jsonl` — one JSON line per run, with timestamp, all effect estimates, and the interpretation. Share individual lines as portable result references.

---

## Experiment log

Every completed experiment appends to `experiments.jsonl`:

```jsonl
{"timestamp": "2026-06-22T10:14:03+00:00", "experiment_id": "a3f7c2b1", "winner": "structured_json", "interpretation": "...", "effects": [...]}
{"timestamp": "2026-06-22T11:02:44+00:00", "experiment_id": "b9e1d4f2", "winner": "chain_of_thought", ...}
```

Read a specific result:
```python
import json
results = [json.loads(l) for l in open("experiments.jsonl")]
```

---

## How causal inference is applied here

Classic A/B testing assigns users to variants. Here, each `(input, sample)` pair is randomly assigned to a prompt variant at runtime — the same clinical trial design, applied to LLM outputs.

**Why this matters:** without random assignment, harder inputs might accidentally cluster in one variant and bias the scores. Random assignment breaks that correlation.

**Effect estimation pipeline:**

```
Random assignment
    → LLM calls (asyncio.gather — all variants run concurrently)
    → LLM-as-judge scoring (0–1 normalised)
    → Average Treatment Effect: ATE = mean(treatment) − mean(control)
    → Bootstrap 95% CI (2000 resamples)
    → Welch's t-test (p-value — no equal-variance assumption)
    → Cohen's d (practical effect size)
```

**Interpreting Cohen's d:**

| d | Magnitude |
|---|---|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

---

## Docker

```bash
docker build -t causal-llm-evaluator .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e GROQ_API_KEY=gsk_... \
  causal-llm-evaluator
```
