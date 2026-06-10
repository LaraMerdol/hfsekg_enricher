"""
passes/pass3_spaces.py
======================
Pass 3 — Space enrichment.

Discovers Spaces that reference known Models or Datasets via the HF API,
fetches full Space metadata, and creates Space nodes with USES_MODEL,
USES_DATASET, and PUBLISHED relationships.

This pass was previously commented out in the monolithic script.  It is now
a callable function; it runs only when ``ENABLE_SPACES_PASS`` is ``True`` in
``config.py``.

Entry point
-----------
``run_pass3_spaces(ctx)``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from models import user_stub, space_stub
from parsers import clean, extract_username
from db.worker_client import WorkerClient

log = logging.getLogger(__name__)


def run_pass3_spaces(ctx: "PipelineContext", all: bool = True) -> None:  # type: ignore[name-defined]
    """
    Discover and enrich Space nodes linked to known models and datasets.

    Strategy
    --------
    1. Query the HF API for spaces that reference each known model and dataset.
    2. Deduplicate across sources using an in-memory mapping of
       ``space_id → {model_ids}``, ``space_id → {dataset_ids}``.
    3. For each discovered space: fetch full metadata, upsert the Space node,
       and create USES_MODEL, USES_DATASET, and PUBLISHED edges.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext`.
    """
    log.info("PASS 3: discovering spaces from model/dataset relations…")

    selected_model_ids   = ctx.writer.get_graph_model_ids()
    selected_dataset_ids = ctx.writer.get_graph_dataset_ids()
    selected_se_model_ids = ctx.writer.get_graph_se_model_ids()

    # Accumulate space → source mappings before fetching details
    space_to_models:   Dict[str, Set[str]] = {}
    space_to_datasets: Dict[str, Set[str]] = {}
    space_to_author:   Dict[str, Optional[str]] = {}

    def _record_space_hit(space_obj: Any, source_id: str, source_kind: str) -> None:
        """Register a space discovery from a model or dataset source."""
        sid = clean(
            getattr(space_obj, "id", None)
            or (space_obj.get("id") if isinstance(space_obj, dict) else None)
        )
        if not sid:
            return
        author = clean(
            getattr(space_obj, "author", None)
            or (space_obj.get("author") if isinstance(space_obj, dict) else None)
        )
        space_to_models.setdefault(sid, set())
        space_to_datasets.setdefault(sid, set())
        if source_kind == "model":
            space_to_models[sid].add(source_id)
        elif source_kind == "dataset":
            space_to_datasets[sid].add(source_id)
        if author and sid not in space_to_author:
            space_to_author[sid] = author

    # --- query spaces by model ---
    if all:
        for idx, model_id in enumerate(sorted(selected_model_ids)):
            worker = ctx.workers[idx % len(ctx.workers)]
            try:
                for space_obj in worker.list_spaces_by_model(model_id):
                    _record_space_hit(space_obj, model_id, "model")
            except Exception as exc:
                ctx.error_rows.append({
                    "stage": "space_by_model", "id": model_id,
                    "status": "error", "error": str(exc),
                })

        # --- query spaces by dataset ---
        for idx, dataset_id in enumerate(sorted(selected_dataset_ids)):
            worker = ctx.workers[idx % len(ctx.workers)]
            try:
                for space_obj in worker.list_spaces_by_dataset(dataset_id):
                    _record_space_hit(space_obj, dataset_id, "dataset")
            except Exception as exc:
                ctx.error_rows.append({
                    "stage": "space_by_dataset", "id": dataset_id,
                    "status": "error", "error": str(exc),
                })
    else:
        for idx, model_id in enumerate(sorted(selected_se_model_ids)):
            worker = ctx.workers[idx % len(ctx.workers)]
            try:
                for space_obj in worker.list_spaces_by_model(model_id):
                    _record_space_hit(space_obj, model_id, "model")
            except Exception as exc:
                ctx.error_rows.append({
                    "stage": "space_by_model", "id": model_id,
                    "status": "error", "error": str(exc),
                })      

    all_space_ids = space_to_models.keys() | space_to_datasets.keys()
    log.info("Relevant spaces found: %d", len(all_space_ids))

    repair_not_needed = ctx.writer.get_graph_space_ids()

    for idx, space_id in enumerate(sorted(all_space_ids)):
        ctx.pause_event.wait()
        try:
            matched_models   = space_to_models.get(space_id, set())
            matched_datasets = space_to_datasets.get(space_id, set())
            author           = space_to_author.get(space_id)

            ctx.discovered_space_ids.add(space_id)

            # Fetch full info to discover additional links even if node exists
            if space_id not in repair_not_needed:
                worker = ctx.workers[idx % len(ctx.workers)]
                info   = worker.fetch_space_info(space_id)
                if info is None:
                    ctx.error_rows.append({
                        "stage": "space_info", "id": space_id, "status": "not_found",
                    })
                    continue

                from parsers import parse_space_bundle
                bundle = parse_space_bundle(info)
                ctx.repair.spaces.append(bundle["space_node"])
                ctx.existing_spaces.add(space_id)

                if bundle["author"]:
                    author = bundle["author"]

                matched_models   = matched_models   | bundle["models"].intersection(selected_model_ids)
                matched_datasets = matched_datasets | bundle["datasets"].intersection(selected_dataset_ids)
            
            if author:
                ctx.discovered_usernames.add(author)
                if author not in ctx.existing_users:
                    ctx.repair.users.append(user_stub(author))
                    ctx.existing_users.add(author)
                if (author, space_id) not in ctx.published_space_user:
                    ctx.repair.published_space_user.append({"username": author, "space_id": space_id})
                    ctx.published_space_user.add((author, space_id))

            for model_id in matched_models:
                if (space_id, model_id) not in ctx.uses_model:
                    ctx.repair.uses_model.append({"space_id": space_id, "model_id": model_id})
                    ctx.uses_model.add((space_id, model_id))

            for dataset_id in matched_datasets:
                if dataset_id not in ctx.existing_datasets:
                    ctx.repair.datasets.append(space_stub(dataset_id))  # minimal stub
                    ctx.existing_datasets.add(dataset_id)
                if (space_id, dataset_id) not in ctx.uses_dataset:
                    ctx.repair.uses_dataset.append({"space_id": space_id, "dataset_id": dataset_id})
                    ctx.uses_dataset.add((space_id, dataset_id))

            ctx.flush_if_needed()

        except Exception as exc:
            ctx.error_rows.append({
                "stage": "space", "id": space_id, "status": "error", "error": str(exc),
            })

    ctx.flush_if_needed(force=True)
    log.info("PASS 3 complete.")
