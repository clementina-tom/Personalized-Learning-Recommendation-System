"""
plrs.assess.router
==================
FastAPI router for cold start diagnostic assessment.

Mount on the main app:
    from plrs.assess.router import assess_router
    app.include_router(assess_router)

Endpoints:
    POST /assess/start                 — begin assessment, returns first question
    POST /assess/{session_id}/answer   — submit answer, returns next question or completion
    GET  /assess/{session_id}          — get session state
    POST /assess/{session_id}/complete — force completion, return mastery vector
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plrs.assess.diagnostic import (
    get_engine,
    get_session_store,
)

assess_router = APIRouter(prefix="/assess", tags=["Assessment"])


# ── Request / Response models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    domain:               str   = Field(..., description="Curriculum domain: 'math' or 'cs'")
    max_questions:        int   = Field(12, ge=3, le=20)
    confidence_threshold: float = Field(0.80, ge=0.5, le=1.0)


class AnswerRequest(BaseModel):
    question_id: str = Field(..., description="ID of the question being answered")
    chosen_idx:  int = Field(..., ge=0, le=3, description="0-based index of chosen option")
    time_ms:     int | None = Field(None, description="Response time in milliseconds")


class QuestionResponse(BaseModel):
    question_id:   str
    topic_id:      str
    topic_label:   str
    text:          str
    options:       list[str]
    difficulty:    float
    question_num:  int
    total_allowed: int


class AnswerResponse(BaseModel):
    correct:         bool
    correct_answer:  str
    explanation:     str
    mastery_after:   float
    confidence:      float
    next_question:   QuestionResponse | None
    session_complete: bool
    session:         dict[str, Any]


class CompletionResponse(BaseModel):
    session_id:   str
    mastery:      dict[str, float]
    session_info: dict[str, Any]
    message:      str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _question_to_response(
    question,
    question_num: int,
    total_allowed: int,
) -> QuestionResponse:
    return QuestionResponse(
        question_id=question.question_id,
        topic_id=question.topic_id,
        topic_label=question.topic_label,
        text=question.text,
        options=question.options,
        difficulty=question.difficulty,
        question_num=question_num,
        total_allowed=total_allowed,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@assess_router.post("/start", response_model=dict)
def start_assessment(req: StartRequest) -> dict:
    """
    Start a new diagnostic assessment session.

    Returns the first question immediately.
    The student must answer via POST /assess/{session_id}/answer.
    """
    engine = get_engine(req.domain)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"No assessment engine for domain '{req.domain}'. "
                   f"Register one via plrs.assess.register_engine().",
        )

    session = engine.start_session(
        domain=req.domain,
        max_questions=req.max_questions,
        confidence_threshold=req.confidence_threshold,
    )

    question = engine.next_question(session)
    if question is None:
        raise HTTPException(status_code=500, detail="No questions available.")

    get_session_store().save(session)

    return {
        "session_id":   session.session_id,
        "domain":       req.domain,
        "max_questions": req.max_questions,
        "first_question": _question_to_response(
            question,
            question_num=1,
            total_allowed=req.max_questions,
        ).model_dump(),
    }


@assess_router.post("/{session_id}/answer", response_model=AnswerResponse)
def submit_answer(session_id: str, req: AnswerRequest) -> AnswerResponse:
    """
    Submit an answer to the current question.

    Returns:
    - Whether the answer was correct
    - The next question (or None if assessment is complete)
    - Updated session state
    """
    store   = get_session_store()
    session = store.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )
    if session.is_complete:
        raise HTTPException(
            status_code=400,
            detail="Assessment already complete. Use POST /assess/{session_id}/complete to get results.",
        )

    engine = get_engine(session.domain)
    if engine is None:
        raise HTTPException(status_code=500, detail="Assessment engine not available.")

    # Find the question
    question = engine.question_bank.get(req.question_id.replace("q_", ""))
    # Try direct lookup or with prefix
    if question is None:
        question = next(
            (q for q in engine.question_bank.values() if q.question_id == req.question_id),
            None,
        )
    if question is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question '{req.question_id}' not found.",
        )

    result = engine.record_answer(
        session=session,
        question=question,
        chosen_idx=req.chosen_idx,
        time_ms=req.time_ms,
    )

    store.save(session)

    next_q = None
    if not session.is_complete:
        next_question = engine.next_question(session)
        if next_question:
            next_q = _question_to_response(
                next_question,
                question_num=session.num_answered + 1,
                total_allowed=session.max_questions,
            )

    return AnswerResponse(
        correct=result["correct"],
        correct_answer=result["correct_answer"],
        explanation=result["explanation"],
        mastery_after=result["mastery_after"],
        confidence=result["confidence"],
        next_question=next_q,
        session_complete=session.is_complete,
        session=session.to_dict(),
    )


@assess_router.get("/{session_id}", response_model=dict)
def get_session(session_id: str) -> dict:
    """Get the current state of an assessment session."""
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )
    return session.to_dict()


@assess_router.post("/{session_id}/complete", response_model=CompletionResponse)
def complete_assessment(session_id: str) -> CompletionResponse:
    """
    Finalise the assessment and return the mastery vector.

    The returned mastery dict is ready to pass directly to:
        PLRSPipeline.recommend_from_mastery(mastery)

    Can be called at any time — does not require all questions to be answered.
    """
    store   = get_session_store()
    session = store.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )

    engine = get_engine(session.domain)
    if engine is None:
        raise HTTPException(status_code=500, detail="Assessment engine not available.")

    mastery = engine.finalise(session)
    store.save(session)

    return CompletionResponse(
        session_id=session_id,
        mastery=mastery,
        session_info=session.to_dict(),
        message=(
            f"Assessment complete. {session.num_answered} questions answered. "
            f"Mastery vector ready for PLRSPipeline."
        ),
    )
