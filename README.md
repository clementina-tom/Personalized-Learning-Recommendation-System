# PLRS — Personalized Learning Recommendation System

> Constraint-aware personalized learning recommendations.
> Plug in your curriculum, get intelligent recommendations out.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Val AUC](https://img.shields.io/badge/Val%20AUC-0.8613-blue.svg)]()

PLRS combines **Self-Attentive Knowledge Tracing with Ebbinghaus Forgetting Curve Decay (SAKTWithDecay)** with a **DAG-based prerequisite constraint layer** to generate recommendations that are both personalized *and* pedagogically sound.

Unlike standard recommenders that optimize for engagement, PLRS guarantees that students are never recommended topics they are not ready for — achieving **0% prerequisite violation rate** against 81%+ for collaborative filtering baselines.

---

## Benchmark Results (v0.2.0 — Real OULAD Training)

| Model | Val AUC | Test AUC | Test Acc |
|-------|---------|----------|----------|
| **SAKTWithDecay** (v0.2.0) | **0.8613** | **0.8152** | **0.7629** |
| SAKTModel (vanilla baseline) | 0.8194 | 0.7823 | 0.7318 |
| SAKT FYP baseline (synthetic data) | 0.7692 | — | — |

**Dataset:** OULAD (23,295 students, 173,739 assessment interactions, 20 skill buckets)
**Hardware:** Kaggle T4 GPU · **Training time:** ~17 epochs / 25s total

The **+0.042 AUC improvement** of SAKTWithDecay over vanilla SAKT demonstrates that Ebbinghaus forgetting curve decay in attention weights is a meaningful architectural contribution, not just a tuning difference.

| Metric | PLRS | Collaborative Filtering | Matrix Factorization |
|--------|------|------------------------|----------------------|
| Val AUC | **0.8613** | — | — |
| Prerequisite Violation Rate | **0.0%** | 81.3% | 83.7% |

---

## How it works

```
Student History → SAKTWithDecay → Mastery Vector
                                        │
                                DAG Constraint Layer
                                (approved / challenging / vetoed)
                                        │
                          Multi-Objective Ranker
                          (gap + readiness + downstream + spaced repetition)
                                        │
                              Ranked Recommendations
                                        │
                            LLM Explainability Layer  ← v0.3.0
                         (natural language explanations)
```

**Three-tier constraint system:**
- ✅ **Approved** — prerequisites met, topic is ready to learn
- ⚠️ **Challenging** — prerequisites partially met, proceed with awareness
- ❌ **Vetoed** — prerequisites not met, structurally blocked

---

## Quick start

> **Note:** PyPI package coming soon. For now, install directly from GitHub:

```bash
pip install git+https://github.com/clementina-tom/Personalized-Learning-Recommendation-System.git
```

```python
from plrs import PLRSPipeline
from plrs.curriculum import load_dag

curriculum = load_dag("data/knowledge_maps/math_dag.json")
pipeline   = PLRSPipeline(curriculum, model_path="sakt_decay_best.pt")

results = pipeline.recommend_from_mastery({
    "whole_numbers":         0.90,
    "algebraic_expressions": 0.75,
    "quadratic_equations":   0.40,
})

for rec in results["approved"]:
    print(f"✅ {rec['topic_label']} (score={rec['score']})")
    print(f"   {rec['reasoning']}")
```

---

## LLM Explainability (v0.3.0)

PLRS supports pluggable LLM providers for natural language explanations.
Use Claude, OpenAI, a local Ollama model, or any custom provider.

```python
from plrs.explain import Explainer
from plrs.llm import OllamaProvider   # free, local, no API key

explainer = Explainer(provider=OllamaProvider(model="llama3.2"))

explanation = explainer.explain_recommendation(
    topic_label="Algebraic Factorization",
    mastery=0.42,
    status="approved",
    reasoning="All prerequisites met.",
    prerequisites=["Algebraic Expressions"],
    unmet_prerequisites=[],
    downstream_count=4,
    score_breakdown={"gap": 0.23, "readiness": 0.40, "downstream": 0.14, "spaced_rep": 0.12},
)
print(explanation.text)
# → "Since you've mastered Algebraic Expressions, you're ready to tackle
#    Algebraic Factorization. Completing this unlocks 4 more topics..."
```

---

## LLM Curriculum Builder (v0.3.0)

Build a curriculum DAG from a topic list or syllabus text — no manual JSON authoring:

```python
from plrs.curriculum.llm_builder import CurriculumBuilder
from plrs.llm import OllamaProvider

builder = CurriculumBuilder(provider=OllamaProvider(model="llama3.2"))

result = builder.from_topics(
    topics=["Mechanics", "Kinematics", "Dynamics", "Energy", "Waves"],
    domain="A-Level Physics",
)
result.save("physics_dag.json")
# → Ready to use with PLRSPipeline
```

Or from a text file:

```bash
python scripts/build_curriculum.py \
    --text-file syllabus.txt \
    --domain "A-Level Physics" \
    --provider ollama --model llama3.2 \
    --output data/knowledge_maps/physics_dag.json
```

---

## REST API

```bash
python scripts/serve.py
# → http://127.0.0.1:8000/docs
```

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET`  | `/health` | Liveness check | Public |
| `GET`  | `/curriculum/{domain}` | Inspect curriculum | Key |
| `POST` | `/recommend` | Get recommendations | Key |
| `POST` | `/what-if` | Simulate mastering a topic | Key |
| `GET`  | `/usage` | Rate limit usage | Key |
| `POST` | `/explain` | Natural language explanation | Key |
| `POST` | `/explain/results` | Explain full result set | Key |
| `POST` | `/explain/what-if` | Explain what a topic unlocks | Key |
| `POST` | `/curriculum/generate` | Build DAG from topics/text | Key |
| `GET`  | `/curriculum/providers` | List LLM providers | Public |

---

## LLM Providers

| Provider | Free | Key Required | Best For |
|----------|------|-------------|----------|
| `ollama` | ✅ Yes | None | Privacy, local dev, no cost |
| `huggingface` | ✅ Free tier | `HF_TOKEN` | Open-source models |
| `openai` | ❌ | `OPENAI_API_KEY` | Production quality |
| `claude` | ❌ | `ANTHROPIC_API_KEY` | Best quality |

```python
# Any provider, same interface
from plrs.llm import ClaudeProvider, OpenAIProvider, OllamaProvider, HuggingFaceProvider

# Bring your own
from plrs.llm import LLMProvider

class MyProvider(LLMProvider):
    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.0):
        return my_model.generate(prompt)
```

---

## Bundled curricula

PLRS ships with two knowledge maps built from the **Nigerian NERDC secondary school curriculum (JSS3–SS2)**:

| Domain | Nodes | Edges |
|--------|-------|-------|
| Secondary School Mathematics | 38 | 45 |
| CS Fundamentals (Digital Technologies) | 31 | 39 |

---

## Architecture

```
plrs/
├── model/
│   ├── sakt.py          # SAKTModel — vanilla baseline
│   ├── sakt_decay.py    # SAKTWithDecay — Ebbinghaus forgetting curve ★
│   ├── trainer.py       # Training loop, early stopping, AUC tracking
│   └── evaluator.py     # Evaluation vs BKT / Popularity baselines
├── constraints/dag.py   # MasteryVector + 3-tier DAGConstraintLayer
├── ranking/
│   ├── ranker.py        # MultiObjectiveRanker (4 signals)
│   └── spaced_repetition.py  # SuperMemo-2 scoring
├── curriculum/
│   ├── loader.py        # load_dag() — any JSON curriculum
│   └── llm_builder.py   # LLM-powered DAG generation ★
├── llm/                 # Pluggable provider layer ★
│   └── providers/       # Claude, OpenAI, Ollama, HuggingFace
├── explain/             # Natural language explainability ★
│   └── explainer.py
├── api/
│   ├── app.py           # FastAPI — auth, rate limiting
│   └── llm_router.py    # LLM endpoints ★
└── pipeline.py          # PLRSPipeline — main entry point

★ = added in v0.3.0
```

---

## Training your own model

```bash
# Prepare your interaction CSV: student_id, skill_id, correct, [timestamp]
python scripts/train.py \
    --data studentVle_processed.csv \
    --num-skills 20 \
    --epochs 50 \
    --device cuda \
    --run-name sakt_decay_v2

# Evaluate against baselines
python scripts/evaluate.py \
    --data studentVle_processed.csv \
    --model checkpoints/sakt_decay_v2_best.pt \
    --domain math
```

---

## API Keys

```bash
# Create a key
python scripts/manage_keys.py create --name "My App" --tier standard

# List keys
python scripts/manage_keys.py list

# Tiers: free (10/min), standard (60/min), premium (300/min), internal
python scripts/manage_keys.py tiers
```

---

## Development

```bash
git clone https://github.com/clementina-tom/plrs
cd plrs

# Core only
pip install -e ".[dev]"

# With LLM providers
pip install -e ".[dev,claude,ollama]"

pytest tests/ -v
```

---

## Roadmap

- [x] v0.1.0 — Open core: SAKT + DAG constraints + FastAPI + auth
- [x] v0.2.0 — Real training: SAKTWithDecay, Val AUC 0.8613 on OULAD
- [x] v0.3.0 — LLM layer: pluggable providers, curriculum builder, explainability
- [ ] v0.4.0 — Hosted: PostgreSQL, Redis, React frontend, multi-tenant
- [ ] v1.0.0 — Production: A/B testing, EdBERTa transfer, mobile SDK

See [ROADMAP.md](ROADMAP.md) and [TASK.md](TASK.md) for full detail.

---

## License

MIT © [Clementina Tom](https://github.com/clementina-tom)
