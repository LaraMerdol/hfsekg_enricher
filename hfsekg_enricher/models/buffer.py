"""
models/buffer.py
================
RepairBuffer: the in-memory write buffer that accumulates pending Neo4j
upserts and relationship creations before they are flushed in a single
batched Cypher transaction.

Keeping it as a plain dataclass with *no* imports from other project
modules makes it easy to unit-test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RepairBuffer:
    """
    Accumulates pending graph writes (node upserts + relationships) that will
    be flushed together via ``_apply_repairs`` once ``total_size`` crosses the
    configured ``FLUSH_THRESHOLD``.

    Each field corresponds either to a node label or a relationship type in
    the HFSEKG knowledge graph.  All items are plain dicts so they can be
    passed directly as Cypher parameters.
    """

    # ------------------------------------------------------------------
    # Node upserts
    # ------------------------------------------------------------------
    model_updates:    List[Dict[str, Any]] = field(default_factory=list)
    users:            List[Dict[str, Any]] = field(default_factory=list)
    organizations:    List[Dict[str, Any]] = field(default_factory=list)
    tasks:            List[Dict[str, Any]] = field(default_factory=list)
    datasets:         List[Dict[str, Any]] = field(default_factory=list)
    spaces:           List[Dict[str, Any]] = field(default_factory=list)
    papers:           List[Dict[str, Any]] = field(default_factory=list)
    collections:      List[Dict[str, Any]] = field(default_factory=list)
    base_model_stubs: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # PUBLISHED relationships
    # ------------------------------------------------------------------
    published_model_user:   List[Dict[str, str]] = field(default_factory=list)
    published_model_org:    List[Dict[str, str]] = field(default_factory=list)
    published_dataset_user: List[Dict[str, str]] = field(default_factory=list)
    published_dataset_org:  List[Dict[str, str]] = field(default_factory=list)
    published_space_user:   List[Dict[str, str]] = field(default_factory=list)
    published_space_org:    List[Dict[str, str]] = field(default_factory=list)
    published_paper_user:   List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Semantic relationships
    # ------------------------------------------------------------------
    defined_model_task:   List[Dict[str, str]] = field(default_factory=list)
    defined_dataset_task: List[Dict[str, str]] = field(default_factory=list)
    trained_on:           List[Dict[str, str]] = field(default_factory=list)
    cites_model_paper:    List[Dict[str, str]] = field(default_factory=list)
    cites_dataset_paper:  List[Dict[str, str]] = field(default_factory=list)
    cites_space_paper:    List[Dict[str, str]] = field(default_factory=list)
    uses_model:           List[Dict[str, str]] = field(default_factory=list)
    uses_dataset:         List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Collection membership
    # ------------------------------------------------------------------
    contains_model:   List[Dict[str, str]] = field(default_factory=list)
    contains_dataset: List[Dict[str, str]] = field(default_factory=list)
    contains_space:   List[Dict[str, str]] = field(default_factory=list)
    contains_paper:   List[Dict[str, str]] = field(default_factory=list)
    owned_by_user:    List[Dict[str, str]] = field(default_factory=list)
    owned_by_org:     List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Social relationships
    # ------------------------------------------------------------------
    likes_model:      List[Dict[str, str]] = field(default_factory=list)
    likes_dataset:    List[Dict[str, str]] = field(default_factory=list)
    likes_space:      List[Dict[str, str]] = field(default_factory=list)
    likes_paper:      List[Dict[str, str]] = field(default_factory=list)
    likes_collection: List[Dict[str, str]] = field(default_factory=list)
    follows_user:     List[Dict[str, str]] = field(default_factory=list)
    follows_org:      List[Dict[str, str]] = field(default_factory=list)
    affiliated_with:  List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Model lineage relationships
    # ------------------------------------------------------------------
    adapter_of:    List[Dict[str, str]] = field(default_factory=list)
    finetuned_from: List[Dict[str, str]] = field(default_factory=list)
    merge_of:      List[Dict[str, str]] = field(default_factory=list)
    quantized_of:  List[Dict[str, str]] = field(default_factory=list)
    based_of:      List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Buffer introspection
    # ------------------------------------------------------------------

    def total_size(self) -> int:
        """Return the total number of pending items across all fields."""
        return sum(len(v) for v in self.__dict__.values() if isinstance(v, list))

    def reset(self) -> None:
        """Clear all lists in-place, reusing the same object reference."""
        self.__dict__.update(RepairBuffer().__dict__)
