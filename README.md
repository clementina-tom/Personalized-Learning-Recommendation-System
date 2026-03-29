# PLRS — Personalized Learning Recommendation System

> Constraint-aware personalized learning recommendations.  
> Plug in your curriculum, get intelligent recommendations out.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)]()

PLRS combines **Self-Attentive Knowledge Tracing (SAKT)** with a **DAG-based prerequisite constraint layer** to generate recommendations that are both personalized *and* pedagogically sound. Unlike standard recommenders that optimize for engagement, PLRS guarantees that students are never recommended topics they are not ready for — achieving **0% prerequisite violation rate** against 81%+ for collaborative filtering baselines.

---

## How it works

```
Student History → SAKT Model → Mastery Vector
                                      │
                              DAG Constraint Layer
                              (approved / challenging / vetoed)
                                      │
                           Multi-Objective Ranker
                           (gap + readiness + downstream value)
                                      │
                              Ranked Recommendations
```

**Three-tier constraint system:**
- ✅ **Approved** — prerequisites met, topic is ready to learn
- ⚠️ **Challenging** — prerequisites partially met, proceed with caution
- ❌ **Vetoed** — prerequisites not met, blocked until foundations are solid

---

## Quick start

```bash
pip install plrs
```

```python
from plrs import PLRSPipeline
from plrs.curriculum import load_dag

# Load your curriculum (or use the bundled Nigerian secondary school maps)
curriculum = load_dag("data/knowledge_maps/math_dag.json")

# Create pipeline
pipeline = PLRSPipeline(curriculum)

# Get recommendations from mastery scores
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

## REST API

Start the server with the bundled Nigerian curriculum DAGs:

```bash
python scripts/serve.py
# → http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check + loaded domains |
| `GET`  | `/curriculum/{domain}` | Inspect curriculum nodes & edges |
| `POST` | `/recommend` | Get personalized recommendations |
| `POST` | `/what-if` | Simulate mastering a topic |

### Example: `/recommend`

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "math",
    "mastery_scores": {
      "whole_numbers": 0.90,
      "algebraic_expressions": 0.75
    },
    "top_n": 5
  }'
```

```json
{
  "approved": [
    {
      "topic_id": "fractions",
      "topic_label": "Fractions",
      "status": "approved",
      "mastery": 0.0,
      "score": 0.9,
      "reasoning": "All 1 prerequisite(s) met.",
      "score_breakdown": {"gap": 0.36, "readiness": 0.4, "downstream": 0.14}
    }
  ],
  "challenging": [...],
  "vetoed": [...],
  "stats": {
    "approved_count": 7,
    "vetoed_count": 28,
    "prerequisite_violation_rate": 0.0
  }
}
```

### Example: `/what-if`

```bash
curl -X POST http://localhost:8000/what-if \
  -H "Content-Type: application/json" \
  -d '{"domain": "math", "topic_id": "algebraic_expressions"}'
```

```json
{
  "topic_label": "Algebraic Expressions",
  "direct_unlocks": [
    {"id": "algebraic_factorization", "label": "Algebraic Factorization"},
    {"id": "logical_reasoning", "label": "Logical Reasoning"}
  ],
  "total_unlocked": 9
}
```

---

## Bring your own curriculum

PLRS is curriculum-agnostic. Define your knowledge graph in a simple JSON format:

```json
{
  "domain": "Physics",
  "nodes": [
    {"id": "mechanics_basics", "label": "Mechanics Basics", "level": "Year 1"},
    {"id": "kinematics",       "label": "Kinematics",       "level": "Year 1"},
    {"id": "dynamics",         "label": "Dynamics",         "level": "Year 2"}
  ],
  "edges": [
    {"from": "mechanics_basics", "to": "kinematics"},
    {"from": "kinematics",       "to": "dynamics"}
  ]
}
```

Then:

```python
from plrs.curriculum import load_dag
from plrs.pipeline import PLRSPipeline

curriculum = load_dag("physics.json")
pipeline   = PLRSPipeline(curriculum)
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
├── model/          # SAKT knowledge tracing model
│   └── sakt.py     # SAKTModel — train, infer, save, load
├── constraints/    # DAG prerequisite constraint layer
│   └── dag.py      # MasteryVector, DAGConstraintLayer
├── ranking/        # Multi-objective recommendation ranker
│   └── ranker.py   # MultiObjectiveRanker
├── curriculum/     # Pluggable curriculum loading
│   └── loader.py   # load_dag(), CurriculumGraph
├── api/            # FastAPI REST interface
│   └── app.py      # /recommend, /what-if, /curriculum, /health
└── pipeline.py     # PLRSPipeline — main orchestrator
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.70` | Mastery probability to consider a topic learned |
| `soft_threshold` | `0.50` | Below threshold but above this → "challenging" |
| `top_n` | `5` | Number of top approved recommendations |
| `w_gap` | `0.4` | Weight: how close student is to mastery |
| `w_readiness` | `0.4` | Weight: prerequisite readiness score |
| `w_downstream` | `0.2` | Weight: how many topics this unlocks |

---

## Running with a trained SAKT model

```bash
python scripts/serve.py --model path/to/sakt_model.pt
```

Or without a model (mastery-dict mode, no SAKT inference):

```bash
python scripts/serve.py
```

---

## Development

```bash
git clone https://github.com/clementina-tom/plrs
cd plrs
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Results

Evaluated on the OULAD dataset with Nigerian secondary school curriculum knowledge maps:

| Metric | PLRS | Collaborative Filtering | Matrix Factorization |
|--------|------|------------------------|----------------------|
| SAKT Val AUC | **0.7692** | — | — |
| Prerequisite Violation Rate | **0.0%** | 81.3% | 83.7% |

---

## Roadmap

- [ ] Forgetting curve decay in SAKT attention (Ebbinghaus/ACT-R)
- [ ] Spaced repetition score in ranking function (SuperMemo-2)
- [ ] AKT / DTransformer model upgrade
- [ ] Dynamic prerequisite discovery via GNNs
- [ ] Hosted API at plrs.dev

---

## License

MIT © [Clementina Tom](https://github.com/clementina-tom)
