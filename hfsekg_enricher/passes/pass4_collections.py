"""
passes/pass4_collections.py
============================
Pass 4 — Collection enrichment.

Discovers Collections that contain known Models, Datasets, Papers, or Spaces
via the HF API, fetches full Collection metadata, and creates Collection nodes
with CONTAINS and OWNED_BY relationships.

This pass was previously commented out in the monolithic script.

Entry point
-----------
``run_pass4_collections(ctx)``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from models import user_stub, org_stub
from parsers import clean
from parsers.bundle_parsers import parse_collection_bundle

log = logging.getLogger(__name__)


def run_pass4_collections(ctx: "PipelineContext", all: bool = True) -> None:  # type: ignore[name-defined]
    """
    Discover and enrich Collection nodes.

    Strategy
    --------
    1. For every known model, dataset, paper, and space, query which
       collections they appear in.
    2. Deduplicate collections across sources.
    3. Fetch full collection info and build CONTAINS + OWNED_BY edges.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext`.
    """
    log.info("PASS 4: discovering collections from model/dataset/paper/space relations…")

    selected_model_ids   = ctx.writer.get_graph_model_ids()
    selected_dataset_ids = ctx.writer.get_graph_dataset_ids()
    selected_paper_ids   = ctx.writer.get_graph_paper_ids()
    selected_space_ids   = ctx.writer.get_graph_space_ids()
    selected_se_model_ids = ctx.writer.get_graph_se_model_ids()

    coll_to_models:   Dict[str, Set[str]] = {}
    coll_to_datasets: Dict[str, Set[str]] = {}
    coll_to_papers:   Dict[str, Set[str]] = {}
    coll_to_spaces:   Dict[str, Set[str]] = {}
    coll_to_author:   Dict[str, Optional[Any]] = {}

    def _record_collection_hit(coll_obj: Any, source_id: str, source_kind: str) -> None:
        """Register a collection discovery from an item source."""
        slug = clean(
            getattr(coll_obj, "slug", None)
            or (coll_obj.get("slug") if isinstance(coll_obj, dict) else None)
        )
        if not slug:
            return
        owner = getattr(coll_obj, "owner", None) or (
            coll_obj.get("owner") if isinstance(coll_obj, dict) else None
        )
        for mapping in [coll_to_models, coll_to_datasets, coll_to_papers, coll_to_spaces]:
            mapping.setdefault(slug, set())

        if source_kind == "model":    coll_to_models[slug].add(source_id)
        elif source_kind == "dataset": coll_to_datasets[slug].add(source_id)
        elif source_kind == "paper":   coll_to_papers[slug].add(source_id)
        elif source_kind == "space":   coll_to_spaces[slug].add(source_id)

        if owner and slug not in coll_to_author:
            coll_to_author[slug] = owner

    # --- query collections by item type ---
    items_list = (
        [
            (selected_model_ids, "model"),
            (selected_dataset_ids, "dataset"),
            (selected_paper_ids, "paper"),
            (selected_space_ids, "space"),
        ]
        if all
        else [
            (selected_se_model_ids, "model"),
        ]
    )
    for items, kind in items_list:
        for idx, item_id in enumerate(sorted(items)):
            worker = ctx.workers[idx % len(ctx.workers)]
            try:
                for coll_obj in worker.list_collections_by_item(item_id, kind):
                    _record_collection_hit(coll_obj, item_id, kind)
            except Exception as exc:
                ctx.error_rows.append({
                    "stage": f"collection_by_{kind}", "id": item_id,
                    "status": "error", "error": str(exc),
                })

    all_coll_ids = (
        coll_to_models.keys()
        | coll_to_datasets.keys()
        | coll_to_papers.keys()
    )
    log.info("Relevant collections found: %d", len(all_coll_ids))

    for idx, coll_id in enumerate(sorted(all_coll_ids)):
        ctx.pause_event.wait()
        try:
            matched_models   = coll_to_models.get(coll_id,   set())
            matched_datasets = coll_to_datasets.get(coll_id, set())
            matched_papers   = coll_to_papers.get(coll_id,   set())
            owner            = coll_to_author.get(coll_id)

            ctx.discovered_collection_ids.add(coll_id)

            if coll_id not in ctx.existing_collections:
                worker = ctx.workers[idx % len(ctx.workers)]
                info   = worker.fetch_collection_info(coll_id)
                if info is None:
                    ctx.error_rows.append({
                        "stage": "collection_info", "id": coll_id, "status": "not_found",
                    })
                    continue

                bundle = parse_collection_bundle(info)
                ctx.repair.collections.append(bundle["collection_node"])
                ctx.existing_collections.add(coll_id)

                if bundle["owner"]:
                    owner = bundle["owner"].get("name") if isinstance(bundle["owner"], dict) else bundle["owner"]

                matched_models   = matched_models   | bundle["models"].intersection(selected_model_ids)
                matched_datasets = matched_datasets | bundle["datasets"].intersection(selected_dataset_ids)
                matched_papers   = matched_papers   | bundle["papers"].intersection(selected_paper_ids)

            # Owner relationship
            if owner:
                owner_str = clean(owner)
                if owner_str:
                    ctx.discovered_usernames.add(owner_str)
                    if owner_str not in ctx.existing_users:
                        ctx.repair.users.append(user_stub(owner_str))
                        ctx.existing_users.add(owner_str)
                    if (coll_id, owner_str) not in ctx.owned_by_user:
                        ctx.repair.owned_by_user.append({"username": owner_str, "collection_id": coll_id})
                        ctx.owned_by_user.add((coll_id, owner_str))

            for model_id in matched_models:
                if (coll_id, model_id) not in ctx.contains_model:
                    ctx.repair.contains_model.append({"collection_id": coll_id, "model_id": model_id})
                    ctx.contains_model.add((coll_id, model_id))

            for dataset_id in matched_datasets:
                if (coll_id, dataset_id) not in ctx.contains_dataset:
                    ctx.repair.contains_dataset.append({"collection_id": coll_id, "dataset_id": dataset_id})
                    ctx.contains_dataset.add((coll_id, dataset_id))

            for paper_id in matched_papers:
                if (coll_id, paper_id) not in ctx.contains_paper:
                    ctx.repair.contains_paper.append({"collection_id": coll_id, "paper_id": paper_id})
                    ctx.contains_paper.add((coll_id, paper_id))

            ctx.flush_if_needed()

        except Exception as exc:
            ctx.error_rows.append({
                "stage": "collection", "id": coll_id, "status": "error", "error": str(exc),
            })

    ctx.flush_if_needed(force=True)
    log.info("PASS 4 complete.")
