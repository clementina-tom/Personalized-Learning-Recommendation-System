"""
plrs.api.llm_router
====================
FastAPI router for LLM-powered endpoints.

Mounts onto the main app in app.py:
    from plrs.api.llm_router import llm_router
    app.include_router(llm_router)

Endpoints:
    POST /explain               — natural language explanation for a recommendation
    POST /explain/results       — explain a full recommendation set
    POST /explain/what-if       — explain what mastering a topic unlocks
    POST /curriculum/generate   — build a curriculum DAG from topics or text
    GET  /curriculum/providers  — list available LLM providers

Authentication:
    All endpoints require the same X-API-Key header as the main API.
    LLM endpoints consume more resources — rate limits apply per-key.

LLM Provider selection:
    Pass "provider" in the request body: "claude", "openai", "ollama", "huggingface"
    Developers using the framework can wire their own provider at server startup
    via register_default_llm_provider().

Provider configuration:
    API keys come from environment variables:
        ANTHROPIC_API_KEY  — Claude
        OPENAI_API_KEY     — OpenAI
        HF_TOKEN           — HuggingFace
        OLLAMA_HOST        — Ollama server (default: http://localhost:11434)

    Or set a single default provider at startup:
        from plrs.api.llm_router import register_default_llm_provider
        from plrs.llm import OllamaProvider
        register_default_llm_provider(OllamaProvider(model="llama3.2"))
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plrs.llm.base import LLMProvider, resolve_provider

# ── Router ────────────────────────────────────────────────────────────────────

llm_router = APIRouter(tags=["LLM"])

# ── Default provider registry ─────────────────────────────────────────────────

_default_provider: LLMProvider | None = None


def register_default_llm_provider(provider: LLMProvider | str) -> None:
    """
    Register a default LLM provider for all LLM endpoints.

    Called at server startup — overrides per-request provider selection.

    Example:
        from plrs.api.llm_router import register_default_llm_provider
        from plrs.llm import OllamaProvider

        register_default_llm_provider(OllamaProvider(model="llama3.2"))
    """
    global _default_provider
    _default_provider = resolve_provider(provider)


def _get_provider(requested: str | None) -> LLMProvider:
    """
    Resolve provider from request or fall back to registered default.

    Priority:
      1. Per-request provider field (if set)
      2. Registered default (register_default_llm_provider)
      3. Claude (final fallback — requires ANTHROPIC_API_KEY)
    """
    if requested:
        try:
            return resolve_provider(requested)
        except (ValueError, ImportError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid LLM provider '{requested}': {e}. "
                       f"Valid options: claude, openai, ollama, huggingface.",
            )

    if _default_provider is not None:
        return _default_provider

    # Final fallback
    try:
        return resolve_provider("claude")
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM provider configured. "
                "Set a default via register_default_llm_provider(), "
                "or pass 'provider' in the request body."
            ),
        )


# ── Request / Response models ─────────────────────────────────────────────────

class ExplainRecommendationRequest(BaseModel):
    topic_label:        str   = Field(..., description="Human-readable topic name")
    mastery:            float = Field(..., ge=0.0, le=1.0, description="Current mastery [0, 1]")
    status:             str   = Field(..., description="approved | challenging | vetoed")
    reasoning:          str   = Field(..., description="Constraint layer reasoning string")
    prerequisites:      list[str] = Field(default_factory=list)
    unmet_prerequisites: list[str] = Field(default_factory=list)
    downstream_count:   int   = Field(0, ge=0)
    score_breakdown:    dict[str, float] = Field(default_factory=dict)
    topic_id:           str | None = None
    provider:           str | None = Field(
        None,
        description="LLM provider: claude | openai | ollama | huggingface"
    )


class ExplainResultsRequest(BaseModel):
    results:  dict[str, Any] = Field(..., description="Output from /recommend endpoint")
    top_n:    int = Field(3, ge=1, le=10)
    provider: str | None = None


class ExplainWhatIfRequest(BaseModel):
    topic_label:    str       = Field(..., description="Topic being simulated")
    direct_unlocks: list[str] = Field(default_factory=list)
    total_unlocked: int       = Field(0, ge=0)
    blocked_by:     list[str] = Field(default_factory=list)
    topic_id:       str | None = None
    provider:       str | None = None


class GenerateCurriculumRequest(BaseModel):
    domain:   str = Field(..., description="Curriculum domain name")
    topics:   list[str] | None = Field(
        None,
        description="List of topic names. Use this OR text_content.",
    )
    text_content: str | None = Field(
        None,
        description="Syllabus or course description text. Use this OR topics.",
    )
    provider: str | None = Field(
        None,
        description="LLM provider: claude | openai | ollama | huggingface"
    )
    run_validation: bool = Field(
        False,
        description="Run a validation pass (extra LLM call). Improves quality."
    )
    max_tokens:  int   = Field(2048, ge=256, le=8192)
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class ExplanationResponse(BaseModel):
    text:     str
    topic_id: str | None
    provider: str
    cached:   bool


class CurriculumGenerateResponse(BaseModel):
    domain:    str
    num_nodes: int
    num_edges: int
    provider:  str
    warnings:  list[str]
    dag:       dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@llm_router.get("/curriculum/providers")
def list_providers() -> dict:
    """
    List available LLM providers and their configuration requirements.
    No authentication required.
    """
    return {
        "providers": [
            {
                "name":        "claude",
                "description": "Anthropic Claude — best quality, hosted",
                "env_var":     "ANTHROPIC_API_KEY",
                "free":        False,
                "streaming":   True,
                "install":     "pip install anthropic",
            },
            {
                "name":        "openai",
                "description": "OpenAI GPT — also works with Groq, Together, Fireworks via base_url",
                "env_var":     "OPENAI_API_KEY",
                "free":        False,
                "streaming":   True,
                "install":     "pip install openai",
            },
            {
                "name":        "ollama",
                "description": "Local open-source models via Ollama — free, private, no API key",
                "env_var":     "OLLAMA_HOST (optional, default: http://localhost:11434)",
                "free":        True,
                "streaming":   True,
                "install":     "pip install ollama + https://ollama.com/download",
                "recommended_models": ["llama3.2", "mistral", "phi3", "gemma2"],
            },
            {
                "name":        "huggingface",
                "description": "HuggingFace Inference API — open-source models, free tier available",
                "env_var":     "HF_TOKEN",
                "free":        "free tier available",
                "streaming":   True,
                "install":     "pip install huggingface_hub",
                "recommended_models": [
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "microsoft/Phi-3-mini-4k-instruct",
                ],
            },
        ],
        "note": (
            "Pass 'provider' in any LLM endpoint request body. "
            "Or set a default at server startup via register_default_llm_provider()."
        ),
    }


@llm_router.post("/explain", response_model=ExplanationResponse)
def explain_recommendation(req: ExplainRecommendationRequest) -> ExplanationResponse:
    """
    Generate a natural language explanation for a single recommendation.

    Takes the same fields returned by /recommend and produces a 2-3 sentence
    explanation written for the student: why this topic is recommended,
    what to be aware of, and what it unlocks.

    Example — pipe output from /recommend into /explain:
        recommendation = results["approved"][0]
        POST /explain with {**recommendation, "provider": "ollama"}
    """
    from plrs.explain import Explainer

    provider = _get_provider(req.provider)
    explainer = Explainer(provider=provider, cache=True)

    try:
        result = explainer.explain_recommendation(
            topic_label=req.topic_label,
            mastery=req.mastery,
            status=req.status,
            reasoning=req.reasoning,
            prerequisites=req.prerequisites,
            unmet_prerequisites=req.unmet_prerequisites,
            downstream_count=req.downstream_count,
            score_breakdown=req.score_breakdown,
            topic_id=req.topic_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return ExplanationResponse(
        text=result.text,
        topic_id=result.topic_id,
        provider=result.provider,
        cached=result.cached,
    )


@llm_router.post("/explain/results", response_model=ExplanationResponse)
def explain_results(req: ExplainResultsRequest) -> ExplanationResponse:
    """
    Generate a natural language overview of a full recommendation set.

    Pass the entire output from /recommend. Returns a 3-4 sentence summary:
    what to focus on first, any challenging topics, and an encouraging note.
    """
    from plrs.explain import Explainer

    provider = _get_provider(req.provider)
    explainer = Explainer(provider=provider, cache=True)

    try:
        result = explainer.explain_results(req.results, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return ExplanationResponse(
        text=result.text,
        topic_id=None,
        provider=result.provider,
        cached=result.cached,
    )


@llm_router.post("/explain/what-if", response_model=ExplanationResponse)
def explain_what_if(req: ExplainWhatIfRequest) -> ExplanationResponse:
    """
    Generate a natural language explanation of what mastering a topic unlocks.

    Pair with /what-if: call /what-if to get the data, then /explain/what-if
    to turn it into a sentence a student can understand.
    """
    from plrs.explain import Explainer

    provider = _get_provider(req.provider)
    explainer = Explainer(provider=provider, cache=True)

    try:
        result = explainer.explain_what_if(
            topic_label=req.topic_label,
            direct_unlocks=req.direct_unlocks,
            total_unlocked=req.total_unlocked,
            blocked_by=req.blocked_by,
            topic_id=req.topic_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return ExplanationResponse(
        text=result.text,
        topic_id=result.topic_id,
        provider=result.provider,
        cached=result.cached,
    )


@llm_router.post("/curriculum/generate", response_model=CurriculumGenerateResponse)
def generate_curriculum(req: GenerateCurriculumRequest) -> CurriculumGenerateResponse:
    """
    Build a curriculum DAG from a list of topics or a text description.

    The LLM infers prerequisite relationships and returns a JSON DAG compatible
    with load_dag() and PLRSPipeline. The generated DAG is also saved to the
    pipeline registry if valid.

    Provide either:
    - topics: ["Whole Numbers", "Fractions", "Algebra", ...]
    - text_content: "Unit 1 covers whole numbers. Unit 2 builds on this..."

    The response DAG can be passed directly to /recommend after registering
    the domain via the serve script.
    """
    from plrs.curriculum.llm_builder import CurriculumBuilder

    if not req.topics and not req.text_content:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'topics' (list of strings) or 'text_content' (syllabus text).",
        )

    if req.topics and req.text_content:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'topics' or 'text_content', not both.",
        )

    provider = _get_provider(req.provider)

    builder = CurriculumBuilder(
        provider=provider,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        validate=req.run_validation,
    )

    try:
        if req.topics:
            result = builder.from_topics(topics=req.topics, domain=req.domain)
        else:
            result = builder.from_text(text=req.text_content, domain=req.domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    return CurriculumGenerateResponse(
        domain=result.domain,
        num_nodes=result.num_nodes,
        num_edges=result.num_edges,
        provider=result.provider,
        warnings=result.warnings,
        dag=result.dag,
    )
