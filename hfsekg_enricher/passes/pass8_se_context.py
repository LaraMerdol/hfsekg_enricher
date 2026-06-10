"""
passes/pass8_se_context.py
=========================
Pass 8 — SE context import.

This pass imports three SE-specific data sources into Neo4j:

- Benchmark rows from ``benchmark.csv`` as :Benchmark nodes and
  ``EVALUATED_ON`` relationships.
- SE task mappings from ``se_task_mappings.csv`` as :SETask nodes and
  ``SUITABLE_FOR`` relationships from :Model nodes.
- SE activity nodes and ``USED_FOR`` relationships from the same mapping
  file.

Only models carrying the ``:SEModel`` label are processed.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config import BENCHMARK_DATA_CSV, SE_TASK_MAPPING_CSV
from parsers import clean

log = logging.getLogger(__name__)


def _row_is_empty(row: Dict[str, Any]) -> bool:
    return all(clean(value) == "" for value in row.values())


def _parse_score(score_raw: str) -> Tuple[Optional[float], bool]:
    score_raw = clean(score_raw)
    if not score_raw:
        return None, False
    try:
        return float(score_raw), True
    except ValueError:
        return None, False


def _make_benchmark_id(benchmark_name: str, implementation: str, language: str) -> str:
    return "||".join([clean(benchmark_name), clean(implementation), clean(language)])


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _ensure_constraints(session) -> None:
    statements = [
        "CREATE CONSTRAINT benchmark_id IF NOT EXISTS FOR (b:Benchmark) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT se_task_id IF NOT EXISTS FOR (t:SETask) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT se_activity_id IF NOT EXISTS FOR (a:SEActivity) REQUIRE a.id IS UNIQUE",
    ]
    for statement in statements:
        try:
            session.run(statement)
        except Exception:
            log.debug("Constraint creation failed or already exists: %s", statement)


def _import_benchmarks(session, benchmark_rows: Iterable[Dict[str, str]], se_model_ids: Set[str], ctx) -> None:
    created = 0
    skipped = 0

    for row in benchmark_rows:
        if _row_is_empty(row):
            continue

        model_id = clean(row.get("model_id"))
        if not model_id:
            continue
        if model_id not in se_model_ids:
            worker = None
            try:
                worker = ctx.workers[0] if getattr(ctx, "workers", None) else None
            except Exception:
                worker = None

            if not worker:
                log.warning("No worker available to resolve model id %s; skipping", model_id)
                continue

            try:
                model_info = worker.fetch_model_info(model_id)
            except Exception as exc:  # pragma: no cover - best-effort lookup
                log.warning("Failed to fetch model info for %s: %s; skipping", model_id, exc)
                continue

            canonical = model_info.get("id") if isinstance(model_info, dict) else None
            if canonical and canonical in se_model_ids:
                model_id = canonical
            else:
                log.warning("SE mapping refers to model_id=%s which is not an SEModel after HF lookup; skipping", model_id)
                continue

        benchmark_name = clean(row.get("benchmark_name"))
        implementation = clean(row.get("implementation"))
        language = clean(row.get("language"))
        metric = clean(row.get("metric"))
        score_raw = clean(row.get("score"))
        repository_id = clean(row.get("repository_id"))

        if not benchmark_name:
            skipped += 1
            continue

        score_value, score_valid = _parse_score(score_raw)
        if score_raw and not score_valid:
            skipped += 1
            continue

        benchmark_id = _make_benchmark_id(benchmark_name, implementation, language)

        session.run(
            """
            MATCH (m:Model {id: $model_id})
            WHERE m:SEModel
            MERGE (b:Benchmark {id: $benchmark_id})
            ON CREATE SET
              b.name = $benchmark_name,
              b.implementation = $implementation,
              b.language = $language,
              b.repository_id = $repository_id
            ON MATCH SET
              b.name = $benchmark_name,
              b.implementation = $implementation,
              b.language = $language,
              b.repository_id = coalesce(b.repository_id, $repository_id)
            MERGE (m)-[r:EVALUATED_ON {metric: $metric}]->(b)
            SET r.score = $score
            """,
            model_id=model_id,
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            implementation=implementation,
            language=language,
            repository_id=repository_id,
            metric=metric,
            score=score_value,
        )
        created += 1

    log.info("Benchmark rows imported: %d created/merged, %d skipped", created, skipped)


def _normalize_mapping_id(value: str) -> str:
    return clean(value).lower()


def _import_se_context(session, mapping_rows: Iterable[Dict[str, str]], se_model_ids: Set[str], ctx) -> None:
    task_rows: List[Dict[str, str]] = []
    activity_rows: List[Dict[str, str]] = []
    relationship_rows: List[Dict[str, str]] = []

    for row in mapping_rows:
        if _row_is_empty(row):
            continue

        model_id = clean(row.get("model_id"))
        if not model_id:
            continue
        if model_id not in se_model_ids:
            worker = None
            try:
                worker = ctx.workers[0] if getattr(ctx, "workers", None) else None
            except Exception:
                worker = None

            if not worker:
                log.warning("No worker available to resolve model id %s; skipping", model_id)
                continue

            try:
                model_info = worker.fetch_model_info(model_id)
            except Exception as exc:  # pragma: no cover - best-effort lookup
                log.warning("Failed to fetch model info for %s: %s; skipping", model_id, exc)
                continue

            canonical = model_info.get("id") if isinstance(model_info, dict) else None
            if canonical and canonical in se_model_ids:
                model_id = canonical
            else:
                log.warning("SE mapping refers to model_id=%s which is not an SEModel after HF lookup; skipping", model_id)
                continue

        task_label = clean(row.get("SE_task_name") or row.get("setask") or row.get("task"))
        activity_label = clean(row.get("SE_activity") or row.get("seactivity") or row.get("activity"))
        reasoning = clean(row.get("se_reasoning"))

        if task_label:
            task_rows.append({"id": _normalize_mapping_id(task_label), "label": task_label})
        if activity_label:
            activity_rows.append({"id": _normalize_mapping_id(activity_label), "label": activity_label})

        if task_label:
            relationship_rows.append({
                "model_id": model_id,
                "se_task_id": _normalize_mapping_id(task_label),
                "se_task_label": task_label,
                "se_activity_id": _normalize_mapping_id(activity_label) if activity_label else "",
                "se_activity_label": activity_label,
                "reasoning": reasoning,
            })

    task_seen: Set[str] = set()
    activity_seen: Set[str] = set()

    for row in task_rows:
        if row["id"] in task_seen:
            continue
        task_seen.add(row["id"])
        session.run(
            """
            MERGE (t:SETask {id: $id})
            SET t.label = coalesce(t.label, $label)
            """,
            id=row["id"],
            label=row["label"],
        )

    for row in activity_rows:
        if row["id"] in activity_seen:
            continue
        activity_seen.add(row["id"])
        session.run(
            """
            MERGE (a:SEActivity {id: $id})
            SET a.label = coalesce(a.label, $label)
            """,
            id=row["id"],
            label=row["label"],
        )

    created = 0
    for row in relationship_rows:
        if not row["model_id"] or not row["se_task_id"]:
            continue

        session.run(
            """
            MATCH (m:Model {id: $model_id})
            WHERE m:SEModel
            MATCH (t:SETask {id: $se_task_id})
            MERGE (m)-[r:SUITABLE_FOR]->(t)
            ON CREATE SET r.reasoning = CASE WHEN $reasoning IS NULL OR $reasoning = '' THEN NULL ELSE $reasoning END
            """,
            model_id=row["model_id"],
            se_task_id=row["se_task_id"],
            reasoning=row["reasoning"],
        )
        created += 1

        if row["se_activity_id"]:
            session.run(
                """
                MATCH (t:SETask {id: $se_task_id})
                MATCH (a:SEActivity {id: $se_activity_id})
                MERGE (t)-[:USED_FOR]->(a)
                """,
                se_task_id=row["se_task_id"],
                se_activity_id=row["se_activity_id"],
            )

    log.info("SE task rows imported: %d relationships created/merged", created)


def run_pass8_se_context(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """Import benchmark and SE context data for SEModel nodes only."""
    log.info("PASS 8: importing benchmark and SE context data…")

    se_model_ids = ctx.writer.get_graph_se_model_ids()
    if not se_model_ids:
        log.info("No SEModel nodes found; skipping benchmark and SE context import.")
        return

    benchmark_rows = _load_csv_rows(BENCHMARK_DATA_CSV)
    mapping_rows = _load_csv_rows(SE_TASK_MAPPING_CSV)

    with ctx.writer.driver.session() as session:
        _ensure_constraints(session)
        _import_benchmarks(session, benchmark_rows, se_model_ids, ctx)
        _import_se_context(session, mapping_rows, se_model_ids, ctx)

    log.info("PASS 8 complete.")