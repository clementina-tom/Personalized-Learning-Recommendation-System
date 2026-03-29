"""
plrs.curriculum.loader
======================
Load and validate curriculum knowledge graphs from JSON.

The JSON schema is deliberately simple so educators can author their own:

    {
        "domain": "Mathematics",
        "nodes": [
            {"id": "algebra_basics", "label": "Algebra Basics", "level": "JSS3"},
            {"id": "quadratic_equations", "label": "Quadratic Equations", "level": "SS1"}
        ],
        "edges": [
            {"from": "algebra_basics", "to": "quadratic_equations"}
        ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class CurriculumGraph:
    """Thin wrapper around a NetworkX DiGraph with domain metadata."""

    domain: str
    graph: nx.DiGraph
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def label(self, node_id: str) -> str:
        return self.graph.nodes[node_id].get("label", node_id)

    def level(self, node_id: str) -> str:
        return self.graph.nodes[node_id].get("level", "")

    def prerequisites(self, node_id: str) -> list[str]:
        return list(self.graph.predecessors(node_id))

    def successors(self, node_id: str) -> list[str]:
        return list(self.graph.successors(node_id))

    def descendants(self, node_id: str) -> list[str]:
        return list(nx.descendants(self.graph, node_id))

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty = all good)."""
        warnings: list[str] = []
        if not nx.is_directed_acyclic_graph(self.graph):
            warnings.append("Graph contains cycles — prerequisite checking will be unreliable.")
        isolates = list(nx.isolates(self.graph))
        if isolates:
            warnings.append(f"{len(isolates)} isolated nodes (no edges): {isolates[:5]}")
        return warnings

    def __repr__(self) -> str:
        return (
            f"CurriculumGraph(domain={self.domain!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )


def load_dag(path: str | Path) -> CurriculumGraph:
    """
    Load a curriculum DAG from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the curriculum JSON file.

    Returns
    -------
    CurriculumGraph

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    _validate_schema(data, path)

    domain = data.get("domain", path.stem)
    meta = {k: v for k, v in data.items() if k not in ("nodes", "edges", "domain")}

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in data["edges"]:
        G.add_edge(edge["from"], edge["to"])

    curriculum = CurriculumGraph(domain=domain, graph=G, meta=meta)

    warnings = curriculum.validate()
    for w in warnings:
        import warnings as _w
        _w.warn(f"[PLRS] {w}", stacklevel=2)

    return curriculum


def _validate_schema(data: dict, path: Path) -> None:
    if "nodes" not in data:
        raise ValueError(f"{path}: Missing required key 'nodes'")
    if "edges" not in data:
        raise ValueError(f"{path}: Missing required key 'edges'")
    for i, node in enumerate(data["nodes"]):
        if "id" not in node:
            raise ValueError(f"{path}: Node at index {i} missing required key 'id'")
    for i, edge in enumerate(data["edges"]):
        if "from" not in edge or "to" not in edge:
            raise ValueError(f"{path}: Edge at index {i} missing 'from' or 'to'")
