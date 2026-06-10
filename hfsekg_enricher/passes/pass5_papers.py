"""
passes/pass5_papers.py
======================
Pass 5 — Paper enrichment.

Fetches full metadata for all Paper nodes discovered in previous passes
(via ``arxiv:`` tags on models and datasets).  Updates Paper nodes with
title, summary, publication date, and upvote count.

This pass was previously commented out in the monolithic script.

Entry point
-----------
``run_pass5_papers(ctx)``
"""

from __future__ import annotations

import concurrent.futures
import threading
import logging
from typing import Any, Dict, List

from config import MAX_WORKERS
from parsers.bundle_parsers import parse_paper_bundle
from models import user_stub

log = logging.getLogger(__name__)


def run_pass5_papers(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """
    Enrich all Paper nodes with full metadata from the HF Papers API.

    Strategy
    --------
    Parallel fetch across worker threads; integrate results into the shared
    buffer under a lock.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext`.
    """
    selected_paper_ids = ctx.writer.get_graph_paper_ids()
    log.info("PASS 5: enriching %d paper nodes…", len(selected_paper_ids))

    buckets: List[List[str]] = [[] for _ in ctx.workers]
    for idx, pid in enumerate(sorted(selected_paper_ids)):
        buckets[idx % MAX_WORKERS].append(pid)

    state_lock = threading.Lock()

    def _worker_fn(worker, bucket: List[str]) -> None:
        for paper_id in bucket:
            ctx.pause_event.wait()
            try:
                info = worker.fetch_paper_info(paper_id)
                if info is None:
                    ctx.error_rows.append({
                        "stage": "paper_info", "id": paper_id, "status": "not_found",
                    })
                    continue

                bundle = parse_paper_bundle(info)

                # Stabilise paper ID
                if not bundle["paper_node"]["id"]:
                    bundle["paper_node"]["id"] = paper_id

                with state_lock:
                    ctx.repair.papers.append(bundle["paper_node"])

                    # Author stubs
                    for username in bundle["author_usernames"]:
                        if username not in ctx.existing_users:
                            ctx.repair.users.append(user_stub(username))
                            ctx.existing_users.add(username)

                    ctx.flush_if_needed()

            except Exception as exc:
                ctx.error_rows.append({
                    "stage": "paper", "id": paper_id, "status": "error", "error": str(exc),
                })

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_worker_fn, ctx.workers[i], buckets[i])
            for i in range(MAX_WORKERS)
        ]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Paper worker thread raised: %s", exc)

    ctx.flush_if_needed(force=True)
    log.info("PASS 5 complete.")
