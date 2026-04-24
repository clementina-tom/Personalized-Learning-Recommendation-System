"""
plrs.explain
============
Natural language explanations for PLRS recommendations.

Usage:
    from plrs.explain import Explainer
    from plrs.llm import OllamaProvider

    explainer = Explainer(provider=OllamaProvider(model="llama3.2"))
    result = explainer.explain_recommendation(
        topic_label="Quadratic Equations",
        mastery=0.42,
        status="approved",
        ...
    )
    print(result.text)
"""

from plrs.explain.explainer import Explainer, ExplanationResult

__all__ = ["Explainer", "ExplanationResult"]
