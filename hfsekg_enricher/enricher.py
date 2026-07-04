"""
enricher.py
===========
Pipeline orchestrator for the HFSEKG graph enrichment pipeline.

This module contains two classes:

``PipelineContext``
    A single shared-state object passed to every pass function.  It holds
    the repair buffer, all existing-state sets (used for deduplication),
    worker clients, the Neo4j writer, audit/error lists, and the
    threading primitives that synchronise flushes with worker threads.

``HFGraphEnricher``
    Top-level entry point.  Builds the context from configuration, runs all
    enabled passes in order, performs the final flush, and writes the audit
    and error CSVs.

Pass execution order
--------------------
0. Lineage BFS  — crawl HF lineage graph from seed models; write edges to
                  CSV; register newly discovered ancestor stubs so Pass 1
                  enriches them automatically.
1. Models       — fetch full HF metadata for every seed + discovered model.
2. Datasets     — enrich datasets discovered via model tags in Pass 1.
3. Spaces       — discover and enrich Spaces linked to models/datasets.
4. Collections  — discover and enrich Collections containing known artifacts.
5. Papers       — enrich Paper nodes with full arXiv metadata.
6. (Social/likes — not yet implemented.)
7. Users        — enrich User nodes; detect and relabel Organization nodes.
8. SE context   — import benchmarks, SE tasks, and SE activities.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import Session

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    HF_TOKENS, MAX_WORKERS, FLUSH_THRESHOLD,
    ENABLE_SPACES_PASS, ENABLE_COLLECTION_PASS,
    ENABLE_PAPER_PASS, ENABLE_SE_CONTEXT_PASS,
    AUDIT_CSV, ERRORS_CSV,
)
from models import RepairBuffer
from db.worker_client import WorkerClient
from db.neo4j_writer  import Neo4jWriter
from parsers.helpers  import write_rows_to_csv

from passes.pass0_lineage     import run_pass0_lineage
from passes.pass1_models      import run_pass1_models
from passes.pass2_datasets    import run_pass2_datasets
from passes.pass3_spaces      import run_pass3_spaces
from passes.pass4_collections import run_pass4_collections
from passes.pass5_papers      import run_pass5_papers
from passes.pass7_users       import run_pass7_users
from passes.pass8_se_context  import run_pass8_se_context

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

class PipelineContext:
    """
    Shared state carrier passed to every pass function.

    All ``existing_*`` sets and relationship-pair sets are populated from
    the graph at startup and kept in sync by each pass, ensuring the
    pipeline is safe to re-run incrementally without creating duplicates.

    Attributes
    ----------
    workers:
        One :class:`WorkerClient` per HF token.  Worker ``i`` always uses
        token ``i``, keeping rate-limit tracking independent per token.
    writer:
        :class:`Neo4jWriter` — all graph writes go through this.
    repair:
        :class:`RepairBuffer` accumulating pending writes between flushes.
    audit_rows / error_rows:
        Append-only lists written to CSV at pipeline end.
    pause_event:
        Set when workers may run; cleared briefly during a Neo4j flush so
        that worker threads pause rather than racing the write.
    model_tag_map / dataset_tag_map:
        Read-only tag classification maps loaded once at startup.
    discovered_*_ids:
        Sets populated by passes to share newly found artifact IDs with
        downstream passes (e.g. Pass 1 → Pass 2 for datasets).
    """

    def __init__(
        self,
        workers:         List[WorkerClient],
        writer:          Neo4jWriter,
        model_tag_map:   Dict[str, Dict[str, str]],
        dataset_tag_map: Dict[str, Dict[str, str]],
        session:         Session,
    ) -> None:
        self.workers         = workers
        self.writer          = writer
        self.model_tag_map   = model_tag_map
        self.dataset_tag_map = dataset_tag_map

        self.repair      = RepairBuffer()
        self.audit_rows: List[Dict[str, Any]] = []
        self.error_rows: List[Dict[str, Any]] = []

        self.pause_event = threading.Event()
        self.pause_event.set()   # workers run by default

        # ---- load existing graph state ----
        log.info("Loading existing graph state…")
        g = writer

        self.existing_models      = g.get_existing_ids(session, "Model",        "id")
        self.existing_users       = g.get_existing_ids(session, "User",         "username")
        self.existing_orgs        = g.get_existing_ids(session, "Organization", "id")
        self.existing_tasks       = g.get_existing_ids(session, "Task",         "id")
        self.existing_datasets    = g.get_existing_ids(session, "Dataset",      "id")
        self.existing_spaces      = g.get_existing_ids(session, "Space",        "id")
        self.existing_papers      = g.get_existing_ids(session, "Paper",        "id")
        self.existing_collections = g.get_existing_ids(session, "Collection",   "slug")

        self.published_model_user   = g.get_existing_relation_pairs(session, "PUBLISHED",      "User",         "username", "Model",       "id")
        self.published_model_org    = g.get_existing_relation_pairs(session, "PUBLISHED",      "Organization", "id",       "Model",       "id")
        self.published_dataset_user = g.get_existing_relation_pairs(session, "PUBLISHED",      "User",         "username", "Dataset",     "id")
        self.published_dataset_org  = g.get_existing_relation_pairs(session, "PUBLISHED",      "Organization", "id",       "Dataset",     "id")
        self.published_space_user   = g.get_existing_relation_pairs(session, "PUBLISHED",      "User",         "username", "Space",       "id")
        self.published_space_org    = g.get_existing_relation_pairs(session, "PUBLISHED",      "Organization", "id",       "Space",       "id")
        self.published_paper_user   = g.get_existing_relation_pairs(session, "PUBLISHED",      "User",         "username", "Paper",       "id")

        self.defined_model_task     = g.get_existing_relation_pairs(session, "DEFINED_FOR",    "Model",        "id",       "Task",        "id")
        self.defined_dataset_task   = g.get_existing_relation_pairs(session, "DEFINED_FOR",    "Dataset",      "id",       "Task",        "id")
        self.trained_on             = g.get_existing_relation_pairs(session, "TRAINED_ON",     "Model",        "id",       "Dataset",     "id")
        self.cites_model_paper      = g.get_existing_relation_pairs(session, "CITES",          "Model",        "id",       "Paper",       "id")
        self.cites_dataset_paper    = g.get_existing_relation_pairs(session, "CITES",          "Dataset",      "id",       "Paper",       "id")
        self.cites_space_paper      = g.get_existing_relation_pairs(session, "CITES",          "Space",        "id",       "Paper",       "id")
        self.uses_model             = g.get_existing_relation_pairs(session, "USES_MODEL",     "Space",        "id",       "Model",       "id")
        self.uses_dataset           = g.get_existing_relation_pairs(session, "USES_DATASET",   "Space",        "id",       "Dataset",     "id")

        self.contains_model         = g.get_existing_relation_pairs(session, "CONTAINS",       "Collection",   "slug",     "Model",       "id")
        self.contains_dataset       = g.get_existing_relation_pairs(session, "CONTAINS",       "Collection",   "slug",     "Dataset",     "id")
        self.contains_space         = g.get_existing_relation_pairs(session, "CONTAINS",       "Collection",   "slug",     "Space",       "id")
        self.contains_paper         = g.get_existing_relation_pairs(session, "CONTAINS",       "Collection",   "slug",     "Paper",       "id")
        self.owned_by_user          = g.get_existing_relation_pairs(session, "OWNED_BY",       "Collection",   "slug",     "User",        "username")
        self.owned_by_org           = g.get_existing_relation_pairs(session, "OWNED_BY",       "Collection",   "slug",     "Organization","id")

        self.likes_model            = g.get_existing_relation_pairs(session, "LIKES",          "User",         "username", "Model",       "id")
        self.likes_dataset          = g.get_existing_relation_pairs(session, "LIKES",          "User",         "username", "Dataset",     "id")
        self.likes_space            = g.get_existing_relation_pairs(session, "LIKES",          "User",         "username", "Space",       "id")
        self.likes_paper            = g.get_existing_relation_pairs(session, "LIKES",          "User",         "username", "Paper",       "id")
        self.likes_collection       = g.get_existing_relation_pairs(session, "LIKES",          "User",         "username", "Collection",  "slug")
        self.follows_user           = g.get_existing_relation_pairs(session, "FOLLOWS",        "User",         "username", "User",        "username")
        self.follows_org            = g.get_existing_relation_pairs(session, "FOLLOWS",        "User",         "username", "Organization","id")
        self.affiliated_with        = g.get_existing_relation_pairs(session, "AFFILIATED_WITH","User",         "username", "Organization","id")

        self.adapter_of    = g.get_existing_relation_pairs(session, "IS_ADAPTER_OF",    "Model", "id", "Model", "id")
        self.finetuned_from = g.get_existing_relation_pairs(session, "IS_FINETUNED_FROM","Model", "id", "Model", "id")
        self.merge_of      = g.get_existing_relation_pairs(session, "IS_MERGE_OF",      "Model", "id", "Model", "id")
        self.quantized_of  = g.get_existing_relation_pairs(session, "IS_QUANTIZED_FROM","Model", "id", "Model", "id")
        self.based_of      = g.get_existing_relation_pairs(session, "IS_BASED_ON",      "Model", "id", "Model", "id")

        # ---- discovery sets populated by passes ----
        self.discovered_dataset_ids:    Set[str] = set()
        self.discovered_space_ids:      Set[str] = set()
        self.discovered_paper_ids:      Set[str] = set()
        self.discovered_usernames:      Set[str] = set()
        self.discovered_org_ids:        Set[str] = set()
        self.discovered_collection_ids: Set[str] = set()

        log.info("Graph state loaded.")

    # ------------------------------------------------------------------
    # Flush helpers used by every pass
    # ------------------------------------------------------------------

    def flush_if_needed(self, force: bool = False) -> None:
        """
        Write the repair buffer to Neo4j if it has reached ``FLUSH_THRESHOLD``.

        Workers pause on ``pause_event`` while the flush is in progress so
        that no new items are added mid-transaction.

        Parameters
        ----------
        force:
            If ``True``, flush regardless of buffer size (used at the end
            of each pass to commit the final partial batch).
        """
        if self.repair.total_size() == 0:
            return
        if self.repair.total_size() >= FLUSH_THRESHOLD or force:
            log.info("Flushing batch of size ~%d", self.repair.total_size())
            self.pause_event.clear()
            try:
                self.writer.apply_repairs(self.repair)
                self.repair.reset()
            finally:
                self.pause_event.set()


# ---------------------------------------------------------------------------
# Main enricher
# ---------------------------------------------------------------------------

class HFGraphEnricher:
    """
    Top-level orchestrator for the HFSEKG graph enrichment pipeline.

    Builds all infrastructure objects (workers, writer, tag maps), constructs
    a :class:`PipelineContext`, and runs the enabled passes in order.

    Parameters
    ----------
    model_ids:
        Optional explicit set of model IDs for Pass 1.  ``None`` means "load
        from allModels.csv" (the normal operating mode).
    neo4j_uri / neo4j_user / neo4j_password:
        Connection details for the Neo4j instance.
    """

    def __init__(
        self,
        model_ids:      Optional[Set[str]],
        neo4j_uri:      str,
        neo4j_user:     str,
        neo4j_password: str,
    ) -> None:
        self.writer      = Neo4jWriter(neo4j_uri, neo4j_user, neo4j_password)
        self.workers     = [WorkerClient(token, idx) for idx, token in enumerate(HF_TOKENS)]
        self._model_ids  = model_ids

    def close(self) -> None:
        """Release all network connections."""
        for w in self.workers:
            w.close()
        self.writer.close()

    # ------------------------------------------------------------------
    # Tag map loading
    # ------------------------------------------------------------------

    def _load_tag_classifications(self) -> Tuple[Dict, Dict]:
        """
        Fetch the HF tag classification maps used by model and dataset parsers.

        Returns ``(model_tag_map, dataset_tag_map)`` where each map is a
        ``{tag_id: {type, label}}`` dict.
        """
        log.info("Loading tag classifications…")
        worker      = self.workers[0]
        raw_model   = worker._get_json("https://huggingface.co/api/models-tags-by-type")
        raw_dataset = worker._get_json("https://huggingface.co/api/datasets-tags-by-type")
        return (
            self._normalize_tag_map(raw_model   or {}),
            self._normalize_tag_map(raw_dataset or {}),
        )

    @staticmethod
    def _normalize_tag_map(raw: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Flatten the nested HF tags-by-type payload into a flat id→{type,label} map."""
        out: Dict[str, Dict[str, str]] = {}
        for bucket_name, items in raw.items():
            if not isinstance(items, list):
                continue
            for item in items:
                tag_id = item.get("id")
                if tag_id:
                    out[tag_id] = {
                        "type":  item.get("type",  bucket_name),
                        "label": item.get("label", tag_id),
                    }
        return out

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full enrichment pipeline.

        Pass execution order:

        0. :func:`run_pass0_lineage` — BFS lineage extraction; writes edges
           to ``LINEAGE_EDGES_CSV``; pre-populates ancestor stubs for Pass 1.
        1. :func:`run_pass1_models` — full HF metadata enrichment for all
           seed + discovered models.
        2. :func:`run_pass2_datasets` — dataset enrichment.
        3. :func:`run_pass3_spaces` — space enrichment (if enabled).
        4. :func:`run_pass4_collections` — collection enrichment (if enabled).
        5. :func:`run_pass5_papers` — paper enrichment (if enabled).
        7. :func:`run_pass7_users` — user enrichment + org detection.
        8. :func:`run_pass8_se_context` — benchmark and SE context import.

        After all passes: final buffer flush, ``audit.csv`` and
        ``errors.csv`` written to disk.
        """
        model_tag_map, dataset_tag_map = self._load_tag_classifications()

        with self.writer.driver.session() as session:
            ctx = PipelineContext(
                workers         = self.workers,
                writer          = self.writer,
                model_tag_map   = model_tag_map,
                dataset_tag_map = dataset_tag_map,
                session         = session,
            )

            # Inject tag maps into workers so pass helpers can reach them
            for w in self.workers:
                w._tag_map_placeholder = model_tag_map

            ctx.dataset_tag_map = dataset_tag_map

            # # --- Pass 0: lineage BFS (must run before Pass 1) ---
            # run_pass0_lineage(ctx)

            # # --- Pass 1: model metadata enrichment ---
            # run_pass1_models(ctx)

            # # --- Pass 2: dataset enrichment ---
            # run_pass2_datasets(ctx)

            # # --- Pass 3: spaces (optional) ---
            # if ENABLE_SPACES_PASS:
            #     run_pass3_spaces(ctx, all=False)

            # # --- Pass 4: collections (optional) ---
            # if ENABLE_COLLECTION_PASS:
            #     run_pass4_collections(ctx, all=False)

            # --- Pass 5: papers (optional) ---
            if ENABLE_PAPER_PASS:
                run_pass5_papers(ctx)

            # --- Pass 6: social/likes — not yet implemented ---

            # --- Pass 7: user enrichment + org detection ---
            run_pass7_users(ctx)

            # --- Pass 8: SE context / benchmarks ---
            if ENABLE_SE_CONTEXT_PASS:
                run_pass8_se_context(ctx)

            # ---- final flush ----
            if ctx.repair.total_size() > 0:
                log.info("Final flush: %d items", ctx.repair.total_size())
                ctx.writer.apply_repairs(ctx.repair)
                ctx.repair.reset()

            # ---- write audit / error CSVs ----
            write_rows_to_csv(str(AUDIT_CSV),  ctx.audit_rows)
            write_rows_to_csv(str(ERRORS_CSV), ctx.error_rows)
            log.info("Done. %s and %s written.", AUDIT_CSV, ERRORS_CSV)
