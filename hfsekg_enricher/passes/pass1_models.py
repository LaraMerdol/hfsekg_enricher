"""
passes/pass1_models.py
======================
Pass 1 — Model enrichment.

Reads the list of model IDs from ``allModels.csv``, partitions them across
worker threads (one per token), and enriches each Model node with full
metadata, publishing relationships, task assignments, dataset/paper links,
and model-lineage edges.

Entry point
-----------
``run_pass1_models(ctx)``
"""

from __future__ import annotations

import concurrent.futures
import csv
import threading
import logging
from typing import Any, Dict, List, Set

from config import (
    MAX_WORKERS, ALL_MODELS_CSV, FETCH_MODEL_README,
    CREATE_MISSING_BASE_MODEL_STUBS,
)
from models import (
    RepairBuffer, user_stub, task_stub,
    dataset_stub, paper_stub, base_model_stub,
)
from parsers import parse_model_bundle
from db.worker_client import WorkerClient

log = logging.getLogger(__name__)


def run_pass1_models(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """
    Enrich all Model nodes in the graph.

    Strategy
    --------
    1. Load target model IDs from ``allModels.csv`` (BOM-safe).
    2. Remove IDs already present in the graph (incremental-run support).
    3. Partition remaining IDs evenly across ``MAX_WORKERS`` worker buckets.
    4. Run each bucket in its own thread; results are merged into the shared
       ``RepairBuffer`` under a ``state_lock``.
    5. Flush the buffer to Neo4j periodically and once more after all workers
       finish.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext` carrying workers, writer, buffer,
        audit/error lists, and existing-state sets.
    """
    log.info("PASS 1: loading model IDs from %s", ALL_MODELS_CSV)

    # Load target IDs from CSV (handle UTF-8 BOM column name)
    with open(ALL_MODELS_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_csv_ids: Set[str] = {
            row.get("n.id") or next(iter(row.values()), "")
            for row in reader
        }
        all_csv_ids.discard("")

    # Preserve the original CSV-listed model IDs so we can tag them as
    # SE models when writing nodes to Neo4j.
    se_model_ids: Set[str] = set(all_csv_ids)


    with ctx.writer.driver.session() as session:
        already_done = set(ctx.writer.get_model_ids_to_repair(session))

    target_ids = sorted(all_csv_ids - already_done)
    log.info(
        "Models — CSV: %d | already done: %d | to process: %d",
        len(all_csv_ids), len(already_done), len(target_ids),
    )

    # Partition IDs: worker i takes indices i, i+MAX_WORKERS, i+2*MAX_WORKERS, …
    buckets: List[List[str]] = [[] for _ in ctx.workers]
    for idx, mid in enumerate(target_ids):
        buckets[idx % MAX_WORKERS].append(mid)

    state_lock = threading.Lock()

    def _worker_fn(worker: WorkerClient, bucket: List[str]) -> None:
        for model_id in bucket:
            ctx.pause_event.wait()  # block if a flush is in progress

            # Fetch and parse
            result = _process_one_model(model_id, worker)

            if result["status"] != "ok":
                ctx.error_rows.append({
                    "stage": "model", "id": model_id,
                    "status": result["status"],
                    "error":  result.get("error"),
                })
                ctx.audit_rows.append({
                    "entity_type": "Model", "entity_id": model_id,
                    "status": result["status"],
                })
                continue

            bundle = result["bundle"]
            # Mark model updates originating from the SE CSV so the writer
            # can add an explicit `SEModel` label.
            try:
                if bundle and isinstance(bundle.get("model_update"), dict):
                    bundle["model_update"]["is_se_model"] = model_id in se_model_ids
            except Exception:
                pass
            issues: List[str] = []

            with state_lock:
                _integrate_model_bundle(model_id, bundle, ctx, issues)
                ctx.flush_if_needed()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_worker_fn, ctx.workers[i], buckets[i])
            for i in range(MAX_WORKERS)
        ]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Model worker thread raised: %s", exc)

    ctx.flush_if_needed(force=True)
    log.info("PASS 1 complete.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_one_model(model_id: str, worker: WorkerClient) -> Dict[str, Any]:
    """Fetch and parse a single model; returns a status dict."""
    try:
        info = worker.fetch_model_info(model_id)
        if info is None:
            return {"model_id": model_id, "status": "not_found"}
        readme = worker.fetch_model_readme(model_id) if FETCH_MODEL_README else None
        bundle = parse_model_bundle(info, readme, worker._tag_map_placeholder)  # tag map injected by context
        return {"model_id": model_id, "status": "ok", "bundle": bundle}
    except Exception as exc:
        log.error("w%d | error on model %s: %s", worker.worker_id, model_id, exc)
        return {"model_id": model_id, "status": "error", "error": str(exc)}


def _integrate_model_bundle(
    model_id: str,
    bundle: Dict[str, Any],
    ctx: "PipelineContext",  # type: ignore[name-defined]
    issues: List[str],
) -> None:
    """
    Merge a parsed model bundle into the shared repair buffer and state sets.

    Covers: model node update, author stub + PUBLISHED edge, task stubs +
    DEFINED_FOR edges, dataset stubs + TRAINED_ON edges, paper stubs + CITES
    edges, and all lineage edges (IS_FINETUNED_FROM, IS_ADAPTER_OF, etc.).
    """
    repair = ctx.repair

    repair.model_updates.append(bundle["model_update"])

    author = bundle["author"]
    if author:
        if author not in ctx.existing_users:
            repair.users.append(user_stub(author))
            ctx.existing_users.add(author)
            issues.append("created_user_node")
        if (author, model_id) not in ctx.published_model_user:
            repair.published_model_user.append({"username": author, "model_id": model_id})
            ctx.published_model_user.add((author, model_id))
            issues.append("user_published_model")

    for task_id in bundle["task_ids"]:
        if task_id not in ctx.existing_tasks:
            repair.tasks.append(task_stub(task_id))
            ctx.existing_tasks.add(task_id)
            issues.append("created_task_node")
        if (model_id, task_id) not in ctx.defined_model_task:
            repair.defined_model_task.append({"model_id": model_id, "task_id": task_id})
            ctx.defined_model_task.add((model_id, task_id))
            issues.append("model_defined_for")

    for dataset_id in bundle["datasets"]:
        if dataset_id not in ctx.existing_datasets:
            ctx.discovered_dataset_ids.add(dataset_id)
            repair.datasets.append(dataset_stub(dataset_id))
            ctx.existing_datasets.add(dataset_id)
            issues.append("created_dataset_stub")
        if (model_id, dataset_id) not in ctx.trained_on:
            repair.trained_on.append({"model_id": model_id, "dataset_id": dataset_id})
            ctx.trained_on.add((model_id, dataset_id))
            issues.append("model_trained_on")

    for paper_id in bundle["papers"]:
        if paper_id not in ctx.existing_papers:
            ctx.discovered_paper_ids.add(paper_id)
            repair.papers.append(paper_stub(paper_id))
            ctx.existing_papers.add(paper_id)
            issues.append("created_paper_stub")
        if (model_id, paper_id) not in ctx.cites_model_paper:
            repair.cites_model_paper.append({"model_id": model_id, "paper_id": paper_id})
            ctx.cites_model_paper.add((model_id, paper_id))
            issues.append("model_cites_paper")

    # # Lineage edges
    # lineage_map = [
    #     ("adapter_models",   repair.adapter_of,     ctx.adapter_of),
    #     ("finetune_models",  repair.finetuned_from, ctx.finetuned_from),
    #     ("merge_models",     repair.merge_of,       ctx.merge_of),
    #     ("quantized_models", repair.quantized_of,   ctx.quantized_of),
    #     ("based_models",     repair.based_of,       ctx.based_of),
    # ]
    # for field_name, target_buf, existing_pairs in lineage_map:
    #     for base_id in bundle[field_name]:
    #         if base_id not in ctx.existing_models and CREATE_MISSING_BASE_MODEL_STUBS:
    #             repair.base_model_stubs.append(base_model_stub(base_id))
    #             ctx.existing_models.add(base_id)
    #         if base_id in ctx.existing_models:
    #             if (model_id, base_id) not in existing_pairs:
    #                 target_buf.append({"model_id": model_id, "base_model_id": base_id})
    #                 existing_pairs.add((model_id, base_id))
    #                 issues.append(f"{field_name}_rel")

    ctx.audit_rows.append({
        "entity_type": "Model",
        "entity_id":   model_id,
        "status":      "ok",
        "issues":      ";".join(issues),
    })
