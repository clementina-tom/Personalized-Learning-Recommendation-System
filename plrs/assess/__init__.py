"""
plrs.assess
===========
Cold start diagnostic assessment for new students.

Bootstraps the mastery vector via adaptive questioning before SAKT kicks in.

Usage:
    from plrs.assess.diagnostic import DiagnosticEngine, register_engine

    engine = DiagnosticEngine(curriculum)
    register_engine("math", engine)

    # Then mount plrs.assess.router on your FastAPI app
"""
from plrs.assess.diagnostic import (
    DiagnosticEngine,
    DiagnosticQuestion,
    AssessmentSession,
    SessionStore,
    register_engine,
    get_engine,
    get_session_store,
)

__all__ = [
    "DiagnosticEngine",
    "DiagnosticQuestion",
    "AssessmentSession",
    "SessionStore",
    "register_engine",
    "get_engine",
    "get_session_store",
]
