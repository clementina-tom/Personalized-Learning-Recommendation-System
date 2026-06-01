# PLRS Product Roadmap

> Living document. Updated as architecture decisions are made.
> Current version: **v0.3.0** (released)

---

## Version Status

| Version | Status | Target |
|---------|--------|--------|
| v0.1.0 | ✅ Released | Done |
| v0.2.0 | ✅ Released | Done |
| v0.3.0 | ✅ Released | Done |
| v0.4.0 | 📋 Planned | 3 months |
| v1.0.0 | 🔭 Vision | 12–18 months |

---

## v0.1.0 — Open Core Foundation ✅

**Theme:** Installable, testable, API-first open-source product.

### Delivered
- [x] SAKT knowledge tracing model (NaN-safe, save/load)
- [x] SAKTWithDecay — Ebbinghaus forgetting curve in attention weights
- [x] 3-tier DAG constraint layer (approved / challenging / vetoed)
- [x] Multi-objective ranker (gap + readiness + downstream + spaced repetition)
- [x] SuperMemo-2 spaced repetition scoring
- [x] FastAPI backend (`/recommend`, `/what-if`, `/curriculum`, `/health`, `/usage`)
- [x] API key auth (4 tiers: free / standard / premium / internal)
- [x] Sliding window rate limiter (per-minute + per-day)
- [x] Admin key management endpoints + CLI (`scripts/manage_keys.py`)
- [x] SAKTTrainer with early stopping, AUC tracking, NaN gradient hook
- [x] PLRSEvaluator with BKT + Popularity baselines
- [x] Streamlit demo (instrument panel aesthetic)
- [x] GitHub Actions CI (Python 3.10–3.12)
- [x] 109 passing tests
- [x] Nigerian SS Mathematics + CS Fundamentals knowledge maps
- [x] HuggingFace Space deployment
- [x] plrs.dev landing page (GitHub Pages ready)

---

## v0.2.0 — AUC Sprint + Real Training ✅

**Theme:** Replace synthetic weights with real trained model. Publish benchmarks.

### Knowledge Tracing
- [x] Train SAKTWithDecay on full OULAD dataset
  - Upload `studentVle.csv` + `vle.csv`
  - Run `scripts/train.py` with forgetting curve decay enabled
  - Target: val AUC ≥ 0.785 (from baseline 0.7692)
- [x] Replace `skill_encoder_v2.csv` (synthetic) with real OULAD `vle.csv` merge
- [x] Commit trained weights to HuggingFace Hub [`Clementio/PLRS`](http://huggingface.co/Clementio/PLRS)
- [x] Publish evaluation report: PLRS vs BKT vs Popularity baseline table

### Infrastructure
- [ ] Push clean repo to `github.com/clementina-tom/plrs`
- [ ] Enable GitHub Pages → `clementina-tom.github.io/plrs`
- [x] Push new HF Space (instrument panel UI)
- [x] Pin model version in `config.json` on HF Hub

### Documentation
- [x] Add training guide to README (`scripts/train.py` walkthrough)
- [x] Add evaluation results table with real numbers
- [ ] Write CONTRIBUTING.md

---

## v0.3.0 — LLM Integration + Architecture Upgrade ✅

**Theme:** Replace static expert-defined DAG with LLM-assisted curriculum understanding.
Add GPT/Claude as an explainability and curriculum authoring layer.

### LLM-Assisted Curriculum Authoring
- [x] `plrs/curriculum/llm_builder.py` — generate a curriculum DAG from:
  - A syllabus PDF or plain-text description
  - A list of topic names
  - Uses LLM (Claude API / OpenAI) to infer prerequisite relationships
  - Human review step before committing the DAG
- [x] `scripts/build_curriculum.py --input syllabus.pdf --output my_dag.json`
- [x] Validate LLM-generated DAGs against known curricula (Nigerian SS Math as ground truth)

### LLM-Powered Explainability
- [x] `plrs/api/explain.py` — new `/explain` endpoint
  - Input: topic_id + student mastery vector
  - Output: natural language explanation of the recommendation
  - Example: *"We recommend Algebraic Factorization next because you've mastered Algebraic Expressions (87%) and it unlocks 4 downstream topics including Quadratic Equations."*
  - Backed by Claude API (streamed response)
- [x] Explanation shown in Streamlit demo alongside recommendations
- [x] Caching layer — same mastery state → same explanation (avoid redundant API calls)

### Knowledge Tracing Architecture Upgrade 
> Coming soon 
- [ ] Implement AKT (Attentive Knowledge Tracing) as alternative model backend
  - `plrs/model/akt.py`
  - AKT explicitly models concept relationships in attention — closer to true knowledge structure
  - Benchmark AKT vs SAKTWithDecay on OULAD
- [ ] Model registry: switch between SAKT / SAKTWithDecay / AKT via config
  - `PLRSPipeline(curriculum, model_backend="akt")`
- [ ] Consider DTransformer if AKT benchmark justifies complexity

### Dynamic Prerequisite Discovery (Research Track)
- [ ] `plrs/curriculum/discovery.py`
  - Learn prerequisite graph from student performance data using Bayesian networks
  - Compare learned DAG vs expert DAG on violation rate and recommendation quality
  
- [ ] Hybrid mode: combine expert DAG edges with data-driven confidence scores

### New Endpoints
- [x] `POST /explain` — natural language recommendation explanation
- [ ] `POST /curriculum/generate` — LLM-assisted DAG generation
- [ ] `GET /curriculum/{domain}/validate` — check a DAG for cycles, orphans, quality

---

## v0.4.0 — Hosted Platform

**Theme:** Multi-tenant SaaS. Schools sign up, upload curricula, get recommendations.

### Multi-Tenancy
- [ ] PostgreSQL backend replacing in-memory KeyStore
  - Tenant table: school name, plan, API key
  - Curriculum table: domain configs per tenant
  - Usage table: request logs for billing
- [ ] Redis replacing in-memory rate limiter (multi-process safe)
- [ ] Tenant isolation: each school's curriculum and student data is scoped

### Auth & Billing
- [ ] OAuth2 / magic link signup (no password)
- [ ] Stripe integration: free → standard → premium upgrade flow
- [ ] Usage dashboard: API calls consumed, quota remaining, top recommended topics
- [ ] Webhook: alert when 80% of quota consumed

### Frontend
- [ ] Replace Streamlit demo with React frontend
  - Teacher view: class-wide mastery heatmap
  - Student view: personal recommendation feed
  - Curriculum editor: drag-and-drop DAG builder (LLM-assisted from v0.3)
- [ ] Deploy on Vercel or Cloudflare Pages

### Deployment
- [x] Docker Compose: FastAPI + PostgreSQL + Redis + Celery (async training jobs)
- [ ] Kubernetes Helm chart for self-hosted enterprise deployments
- [x] GitHub Actions: push-to-deploy to Railway or Render (managed hosting)

### Cold Start
- [ ] Diagnostic assessment module: 10–15 adaptive questions for new students
  - Before SAKT has enough history, use a short quiz to bootstrap mastery vector
  - Questions selected using IRT (Item Response Theory) difficulty calibration

---

## v1.0.0 — Production Platform

**Theme:** Enterprise-grade. Real student outcomes. Publishable results.

### Educational Validation
- [ ] A/B testing framework: PLRS recommendations vs random control
  - Measure: time-to-mastery, topic completion rate, prerequisite violation rate
  - Requires real users — partner with 2–3 Nigerian secondary schools
- [ ] Knowledge gain metric: pre/post test score improvement
- [ ] Retention metric: test scores 1 month after recommendation followed

### Advanced ML
- [ ] MAML (Model-Agnostic Meta-Learning) for cold start
  - Adapt quickly to new students with 3–5 interactions instead of 20+
- [ ] Counterfactual evaluation via Inverse Propensity Scoring
  - Estimate "what would have happened if we recommended B instead of A?"
- [ ] Engagement prediction: secondary model predicting likelihood of topic completion

### Cross-Domain Transfer
- [ ] EdBERTa-based automatic curriculum mapping
  - Map "Algebra" in Nigerian curriculum → "Algebra" in Kenyan / Indian curriculum
  - True plug-and-play across education systems without manual DAG authoring
- [ ] Zero-shot transfer: train on Math, deploy to Physics with concept graph alignment

### Mobile
- [ ] iOS + Android SDK
  - Lightweight Swift/Kotlin wrapper around the REST API
  - Local caching: recommendations available offline
  - Push notifications: "Algebraic Factorization is due for review today"

### Enterprise
- [ ] SSO / SAML integration (for school district IT systems)
- [ ] GDPR + FERPA compliance mode (student data stays on-premise)
- [ ] White-label SDK: EdTech companies embed PLRS under their own brand

---

## Features Explicitly Not Planned

These were evaluated and rejected:

| Feature | Reason |
|---------|--------|
| NSGA-II Pareto optimization | Research complexity exceeds user value |
| Time-of-day recommendation adaptation | Needs longitudinal real-user data not available |
| Multi-task learning (correctness + response time) | OULAD has no response-time data at question level |
| Diversity penalty (MMR) | Unprovable benefit without user study |
| PageRank downstream scoring | Marginal improvement over current descendant-count |

---

## Architecture Principles (All Versions)

1. **Curriculum-agnostic by default** — any JSON DAG works, Nigerian maps are bundled examples
2. **Model-agnostic by design** — SAKT / AKT / DTransformer are interchangeable backends
3. **Open core, hosted premium** — core library always free and open source
4. **Constraint-first** — DAG prerequisite validation is non-negotiable, not optional
5. **API-first** — every feature is a REST endpoint before it's a UI
6. **Tests before merge** — CI must pass on Python 3.10–3.12 for every PR

---

## Research Publications Target

| Paper | Target Venue | Timeline |
|-------|-------------|----------|
| Dynamic prerequisite discovery via Bayesian networks | EDM 2027 or LAK 2027 | v0.3.0 |
| Constraint-aware KT with forgetting curve decay | AIED 2027 | v0.2.0 |
| Cross-curriculum zero-shot transfer | NeurIPS Education Workshop 2027 | v1.0.0 |

---

*Last updated: April 2026*
*Maintainer: Clementina Tom — github.com/clementina-tom*
