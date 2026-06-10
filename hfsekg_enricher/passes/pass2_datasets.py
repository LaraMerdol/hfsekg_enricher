"""
passes/pass2_datasets.py
========================
Pass 2 — Dataset enrichment.

Processes all dataset IDs discovered during Pass 1 (via ``base_model:``
and ``dataset:`` tags on models).  For each dataset it fetches full HF
metadata (and optionally the README), then updates the Dataset node and
creates PUBLISHED, DEFINED_FOR, and CITES relationships.

Entry point
-----------
``run_pass2_datasets(ctx)``
"""

from __future__ import annotations

import concurrent.futures
import threading
import logging
from typing import Any, Dict, List

from config import MAX_WORKERS, FETCH_DATASET_README
from models import RepairBuffer, user_stub, task_stub, paper_stub
from parsers import parse_dataset_bundle
from db.worker_client import WorkerClient

log = logging.getLogger(__name__)


def run_pass2_datasets(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """
    Enrich all Dataset nodes discovered in Pass 1.

    Strategy
    --------
    Mirrors Pass 1: partition IDs across worker buckets, process in parallel,
    merge results into the shared buffer under a lock, flush periodically.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext`.
    """
    log.info("PASS 2: processing %d discovered datasets…", len(ctx.discovered_dataset_ids))

    buckets: List[List[str]] = [[] for _ in ctx.workers]
    for idx, did in enumerate(sorted(ctx.discovered_dataset_ids)):
        buckets[idx % MAX_WORKERS].append(did)

    state_lock = threading.Lock()

    def _worker_fn(worker: WorkerClient, bucket: List[str]) -> None:
        for dataset_id in bucket:
            ctx.pause_event.wait()

            result = _process_one_dataset(dataset_id, worker, ctx)

            if result["status"] != "ok":
                ctx.error_rows.append({
                    "stage": "dataset", "id": dataset_id,
                    "status": result["status"],
                    "error":  result.get("error"),
                })
                continue

            bundle = result["bundle"]
            issues: List[str] = []

            with state_lock:
                _integrate_dataset_bundle(dataset_id, bundle, ctx, issues)
                ctx.flush_if_needed()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_worker_fn, ctx.workers[i], buckets[i])
            for i in range(MAX_WORKERS)
        ]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Dataset worker thread raised: %s", exc)

    ctx.flush_if_needed(force=True)
    log.info("PASS 2 complete. Processed %d datasets.", len(ctx.discovered_dataset_ids))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_one_dataset(
    dataset_id: str,
    worker: WorkerClient,
    ctx: "PipelineContext",  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """Fetch and parse a single dataset; returns a status dict."""
    try:
        info = worker.fetch_dataset_info(dataset_id)
        if info is None:
            return {"dataset_id": dataset_id, "status": "not_found"}
        readme = worker.fetch_dataset_readme(dataset_id) if FETCH_DATASET_README else None
        bundle = parse_dataset_bundle(info, readme, ctx.dataset_tag_map)
        return {"dataset_id": dataset_id, "status": "ok", "bundle": bundle}
    except Exception as exc:
        log.error("Error processing dataset %s: %s", dataset_id, exc)
        return {"dataset_id": dataset_id, "status": "error", "error": str(exc)}


def _integrate_dataset_bundle(
    dataset_id: str,
    bundle: Dict[str, Any],
    ctx: "PipelineContext",  # type: ignore[name-defined]
    issues: List[str],
) -> None:
    """
    Merge a parsed dataset bundle into the shared repair buffer and state sets.

    Covers: dataset node update, author stub + PUBLISHED edge, task stubs +
    DEFINED_FOR edges, paper stubs + CITES edges.
    """
    repair = ctx.repair

    repair.datasets.append(bundle["dataset_update"])

    author = bundle["author"]
    if author:
        if author not in ctx.existing_users:
            repair.users.append(user_stub(author))
            ctx.existing_users.add(author)
            issues.append("created_user_node")
        if (author, dataset_id) not in ctx.published_dataset_user:
            repair.published_dataset_user.append({"username": author, "dataset_id": dataset_id})
            ctx.published_dataset_user.add((author, dataset_id))
            issues.append("user_published_dataset")

    for task_id in bundle["task_ids"]:
        if task_id not in ctx.existing_tasks:
            repair.tasks.append(task_stub(task_id))
            ctx.existing_tasks.add(task_id)
            issues.append("created_task_node")
        if (dataset_id, task_id) not in ctx.defined_dataset_task:
            repair.defined_dataset_task.append({"dataset_id": dataset_id, "task_id": task_id})
            ctx.defined_dataset_task.add((dataset_id, task_id))
            issues.append("dataset_defined_for")

    for paper_id in bundle["papers"]:
        if paper_id not in ctx.existing_papers:
            repair.papers.append(paper_stub(paper_id))
            ctx.existing_papers.add(paper_id)
            issues.append("created_paper_stub")
        if (dataset_id, paper_id) not in ctx.cites_dataset_paper:
            repair.cites_dataset_paper.append({"dataset_id": dataset_id, "paper_id": paper_id})
            ctx.cites_dataset_paper.add((dataset_id, paper_id))
            issues.append("dataset_cites_paper")

    ctx.audit_rows.append({
        "entity_type": "Dataset",
        "entity_id":   dataset_id,
        "status":      "ok",
        "issues":      ";".join(issues),
    })
