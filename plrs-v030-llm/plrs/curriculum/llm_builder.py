"""
plrs.curriculum.llm_builder
============================
Build a PLRS-compatible curriculum DAG from plain text or topic lists
using any pluggable LLM provider.

Works with any LLMProvider — Claude, OpenAI, Ollama, HuggingFace, or custom.

Usage:
    from plrs.llm import ClaudeProvider, OllamaProvider
    from plrs.curriculum.llm_builder import CurriculumBuilder

    # From a list of topic names
    builder = CurriculumBuilder(provider=ClaudeProvider())
    dag = builder.from_topics(
        topics=["Whole Numbers", "Fractions", "Algebra", "Quadratic Equations"],
        domain="Secondary School Mathematics",
    )

    # From a syllabus text
    dag = builder.from_text(
        text="Unit 1: Whole Numbers. Unit 2: Fractions build on Unit 1...",
        domain="Mathematics",
    )

    # Save to file for use with load_dag()
    builder.save(dag, "my_curriculum.json")

    # With local model (no API key, free)
    builder = CurriculumBuilder(provider=OllamaProvider(model="llama3.2"))
    dag = builder.from_topics(topics=[...])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plrs.llm.base import LLMProvider, MockProvider, resolve_provider


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert curriculum designer and educator.
Your task is to analyze educational topics and determine prerequisite relationships between them.
A prerequisite relationship means: a student should learn topic A before topic B.

You must respond ONLY with valid JSON — no explanation, no markdown, no preamble."""

FROM_TOPICS_PROMPT = """Given these educational topics for the domain "{domain}":

{topics_list}

Determine the prerequisite relationships between them.
Return a JSON object with this exact structure:

{{
  "domain": "{domain}",
  "nodes": [
    {{"id": "snake_case_id", "label": "Human Readable Label", "level": "optional level/grade"}}
  ],
  "edges": [
    {{"from": "prerequisite_topic_id", "to": "dependent_topic_id"}}
  ]
}}

Rules:
- Use snake_case for all node IDs (e.g. "quadratic_equations" not "Quadratic Equations")
- Only add an edge if topic A is clearly a prerequisite for topic B
- Do not add circular dependencies
- "level" is optional — include it if the topic has a clear grade/year level
- Return ONLY valid JSON, nothing else"""

FROM_TEXT_PROMPT = """Analyze this educational content for the domain "{domain}":

---
{text}
---

Extract all distinct topics and their prerequisite relationships.
Return a JSON object with this exact structure:

{{
  "domain": "{domain}",
  "nodes": [
    {{"id": "snake_case_id", "label": "Human Readable Label", "level": "optional level"}}
  ],
  "edges": [
    {{"from": "prerequisite_topic_id", "to": "dependent_topic_id"}}
  ]
}}

Rules:
- Use snake_case for all node IDs
- Only add edges for clear prerequisite relationships
- Do not create circular dependencies
- Extract between 5 and 50 topics maximum
- Return ONLY valid JSON, nothing else"""

VALIDATE_PROMPT = """Review this curriculum DAG JSON and fix any issues:

{dag_json}

Issues to check and fix:
1. Circular dependencies (A requires B requires A)
2. Node IDs in edges that don't exist in nodes list
3. Invalid snake_case IDs (fix spaces, special chars)
4. Self-referencing edges (a topic requiring itself)

Return the corrected JSON with the same structure. Return ONLY valid JSON."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    """Result from a curriculum build operation."""
    dag:          dict[str, Any]              # raw DAG dict (compatible with load_dag)
    domain:       str
    num_nodes:    int
    num_edges:    int
    provider:     str
    warnings:     list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return self.dag

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.dag, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"BuildResult(domain={self.domain!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges}, "
            f"provider={self.provider!r}, warnings={len(self.warnings)})"
        )


# ── Curriculum Builder ────────────────────────────────────────────────────────

class CurriculumBuilder:
    """
    Build a PLRS-compatible curriculum DAG using an LLM.

    Provider-agnostic — works with Claude, OpenAI, Ollama, HuggingFace,
    or any custom LLMProvider implementation.

    Parameters
    ----------
    provider : str or LLMProvider
        LLM provider to use. String shorthand: "claude", "openai", "ollama", "huggingface".
    max_tokens : int
        Max tokens for LLM response (default 2048 — DAGs can be verbose).
    temperature : float
        Sampling temperature. 0.0 = deterministic (recommended for JSON output).
    validate : bool
        If True, runs a second LLM pass to validate and fix the generated DAG.
        Adds one extra API call but improves reliability.
    """

    def __init__(
        self,
        provider: str | LLMProvider = "claude",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        validate: bool = False,
    ) -> None:
        self.provider    = resolve_provider(provider)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.validate    = validate

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def from_topics(
        self,
        topics: list[str],
        domain: str = "Unknown Domain",
    ) -> BuildResult:
        """
        Build a curriculum DAG from a flat list of topic names.

        Parameters
        ----------
        topics : list[str]
            Topic names in any order (e.g. ["Fractions", "Algebra", "Calculus"]).
            The LLM infers prerequisite relationships.
        domain : str
            Human-readable domain name (e.g. "Secondary School Mathematics").

        Returns
        -------
        BuildResult
        """
        if not topics:
            raise ValueError("topics list cannot be empty")

        topics_list = "\n".join(f"- {t}" for t in topics)
        prompt = FROM_TOPICS_PROMPT.format(
            domain=domain,
            topics_list=topics_list,
        )

        return self._build(prompt=prompt, domain=domain)

    def from_text(
        self,
        text: str,
        domain: str = "Unknown Domain",
        max_text_chars: int = 8000,
    ) -> BuildResult:
        """
        Build a curriculum DAG from a syllabus or course description text.

        Parameters
        ----------
        text : str
            Syllabus, course outline, or any descriptive educational text.
        domain : str
            Human-readable domain name.
        max_text_chars : int
            Truncate text to this length before sending to LLM (default 8000).

        Returns
        -------
        BuildResult
        """
        if not text.strip():
            raise ValueError("text cannot be empty")

        truncated = text[:max_text_chars]
        if len(text) > max_text_chars:
            truncated += "\n[... text truncated ...]"

        prompt = FROM_TEXT_PROMPT.format(domain=domain, text=truncated)
        return self._build(prompt=prompt, domain=domain)

    def save(self, result: BuildResult, path: str | Path) -> None:
        """Save a BuildResult's DAG to a JSON file."""
        result.save(path)

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _build(self, prompt: str, domain: str) -> BuildResult:
        """Core build loop: prompt → parse → optionally validate → return."""
        raw = self.provider.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        dag, warnings = self._parse_dag(raw, domain)

        # Optional validation pass
        if self.validate and not isinstance(self.provider, MockProvider):
            dag, val_warnings = self._validate_dag(dag)
            warnings.extend(val_warnings)

        # Post-process: ensure IDs are consistent
        dag, fix_warnings = self._fix_dag(dag)
        warnings.extend(fix_warnings)

        return BuildResult(
            dag=dag,
            domain=dag.get("domain", domain),
            num_nodes=len(dag.get("nodes", [])),
            num_edges=len(dag.get("edges", [])),
            provider=self.provider.provider_name,
            warnings=warnings,
            raw_response=raw,
        )

    def _parse_dag(self, raw: str, domain: str) -> tuple[dict, list[str]]:
        """Extract and parse JSON from LLM response."""
        warnings = []

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # Find first { to last } (handle preamble/postamble)
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            warnings.append("LLM response did not contain JSON — using empty DAG")
            return self._empty_dag(domain), warnings

        json_str = cleaned[start:end]

        try:
            dag = json.loads(json_str)
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error: {e} — using empty DAG")
            return self._empty_dag(domain), warnings

        # Schema validation
        if "nodes" not in dag:
            dag["nodes"] = []
            warnings.append("LLM response missing 'nodes' — defaulting to empty")
        if "edges" not in dag:
            dag["edges"] = []
            warnings.append("LLM response missing 'edges' — defaulting to empty")
        if "domain" not in dag:
            dag["domain"] = domain

        return dag, warnings

    def _validate_dag(self, dag: dict) -> tuple[dict, list[str]]:
        """Second LLM pass to fix structural issues."""
        warnings = []
        try:
            prompt = VALIDATE_PROMPT.format(dag_json=json.dumps(dag, indent=2))
            raw = self.provider.complete(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            validated, parse_warnings = self._parse_dag(raw, dag.get("domain", ""))
            warnings.extend(parse_warnings)
            return validated, warnings
        except Exception as e:
            warnings.append(f"Validation pass failed: {e} — keeping original")
            return dag, warnings

    def _fix_dag(self, dag: dict) -> tuple[dict, list[str]]:
        """Post-process: sanitise IDs, remove orphan edges, remove cycles."""
        warnings: list[str] = []
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])

        # Build ID set
        node_ids = {n.get("id", "") for n in nodes if n.get("id")}

        # Sanitise node IDs
        fixed_nodes = []
        id_map: dict[str, str] = {}  # old_id → new_id
        for node in nodes:
            raw_id  = node.get("id", "")
            clean   = self._sanitise_id(raw_id)
            if clean != raw_id:
                warnings.append(f"Node ID sanitised: '{raw_id}' → '{clean}'")
                id_map[raw_id] = clean
            node["id"] = clean
            fixed_nodes.append(node)

        # Rebuild node_ids with clean IDs
        node_ids = {n["id"] for n in fixed_nodes}

        # Filter and remap edges
        fixed_edges = []
        seen_edges: set[tuple[str, str]] = set()

        for edge in edges:
            src = id_map.get(edge.get("from", ""), edge.get("from", ""))
            dst = id_map.get(edge.get("to", ""),   edge.get("to", ""))

            if src not in node_ids:
                warnings.append(f"Edge source '{src}' not in nodes — removed")
                continue
            if dst not in node_ids:
                warnings.append(f"Edge target '{dst}' not in nodes — removed")
                continue
            if src == dst:
                warnings.append(f"Self-loop on '{src}' — removed")
                continue
            if (src, dst) in seen_edges:
                warnings.append(f"Duplicate edge '{src}' → '{dst}' — removed")
                continue

            seen_edges.add((src, dst))
            fixed_edges.append({"from": src, "to": dst})

        # Remove cycles using DFS
        fixed_edges, cycle_warnings = self._remove_cycles(fixed_edges, node_ids)
        warnings.extend(cycle_warnings)

        dag["nodes"] = fixed_nodes
        dag["edges"] = fixed_edges
        return dag, warnings

    @staticmethod
    def _sanitise_id(raw: str) -> str:
        """Convert any string to a valid snake_case identifier."""
        s = raw.lower().strip()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = s.strip("_")
        return s or "unknown_topic"

    @staticmethod
    def _remove_cycles(
        edges: list[dict],
        node_ids: set[str],
    ) -> tuple[list[dict], list[str]]:
        """Remove edges that create cycles using greedy DFS."""
        warnings: list[str] = []
        adj: dict[str, set[str]] = {n: set() for n in node_ids}
        safe_edges: list[dict] = []

        for edge in edges:
            src, dst = edge["from"], edge["to"]
            # Would adding src→dst create a cycle?
            # Check if dst can already reach src
            visited: set[str] = set()
            stack = [dst]
            creates_cycle = False
            while stack:
                node = stack.pop()
                if node == src:
                    creates_cycle = True
                    break
                if node not in visited:
                    visited.add(node)
                    stack.extend(adj.get(node, set()))

            if creates_cycle:
                warnings.append(f"Cycle detected — removed edge '{src}' → '{dst}'")
            else:
                adj[src].add(dst)
                safe_edges.append(edge)

        return safe_edges, warnings

    @staticmethod
    def _empty_dag(domain: str) -> dict:
        return {"domain": domain, "nodes": [], "edges": []}
