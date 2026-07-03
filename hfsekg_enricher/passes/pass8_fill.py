"""
passes/pass8_se_context.py
=========================
Pass 8 — SE context import.

This pass imports three SE-specific data sources into Neo4j:

- Benchmark rows from ``benchmark.csv`` as :Benchmark nodes and
  ``EVALUATED_ON`` relationships.
- SE task mappings from ``se_task_mappings.csv`` as :SETask nodes and
  ``SUITABLE_FOR`` relationships from :Model nodes.
- SE task -> SE activity links read from a JSON taxonomy file
  (``SE_ACTIVITY_MAPPING_JSON``). For every task in the taxonomy the pass
  ensures the :SEActivity node exists and creates the ``USED_FOR``
  relationship. If a task from the taxonomy is not yet present in the graph
  (matched case-insensitively), a new :SETask node is created and connected
  to its activity.

Only models carrying the ``:SEModel`` label are processed.
"""

from __future__ import annotations

import csv
import json
import logging
from os import path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
# Allow running this file directly (e.g. ``python passes/pass8_se_context.py``).
# When run as a script, only the ``passes/`` directory is on sys.path, so the
# project-root modules ``config`` and ``parsers`` are not importable. Add the
# project root (the parent of this file's directory) to the path in that case.

 
from config import BENCHMARK_DATA_CSV, SE_TASK_MAPPING_CSV, SE_ACTIVITY_MAPPING_JSON
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


def _load_activity_mapping_json(json_path: Path) -> List[Tuple[str, str]]:
    """Read SE task -> SE activity links from a JSON taxonomy file.

    Returns a list of ``(task_label, activity_label)`` tuples. Several JSON
    shapes are accepted so the file can be the SEMODS taxonomy in whichever
    form it was exported:

    1. ``{"Activity": ["task", "task", ...], ...}``
    2. ``{"Activity": {"tasks": ["task", ...]}, ...}``
    3. ``{"Task": "Activity", ...}``
    4. ``[{"se_task_name": "...", "se_activity": "..."}, ...]``
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)

    task_keys = ("se_task_name", "SE_task_name", "setask", "task", "task_name", "name", "label")
    activity_keys = ("se_activity", "SE_activity", "seactivity", "activity", "activity_name")

    def first_value(obj: Dict[str, Any], keys: Tuple[str, ...]) -> str:
        for key in keys:
            if key in obj and clean(obj[key]):
                return clean(obj[key])
        return ""

    mappings: List[Tuple[str, str]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                # {"Activity": ["task", ...]} or {"Activity": [{task obj}, ...]}
                for item in value:
                    if isinstance(item, str):
                        mappings.append((item, key))
                    elif isinstance(item, dict):
                        task_label = first_value(item, task_keys)
                        activity_label = first_value(item, activity_keys) or key
                        if task_label:
                            mappings.append((task_label, activity_label))
            elif isinstance(value, dict):
                # {"Activity": {"tasks": [...]}}
                tasks = value.get("tasks") or value.get("se_tasks") or []
                if isinstance(tasks, list):
                    for item in tasks:
                        if isinstance(item, str):
                            mappings.append((item, key))
                        elif isinstance(item, dict):
                            task_label = first_value(item, task_keys)
                            if task_label:
                                mappings.append((task_label, key))
            elif isinstance(value, str):
                # {"Task": "Activity"}
                mappings.append((key, value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                task_label = first_value(item, task_keys)
                activity_label = first_value(item, activity_keys)
                if task_label and activity_label:
                    mappings.append((task_label, activity_label))

    return mappings


def _import_se_context(session, mapping_rows: Iterable[Dict[str, str]], se_model_ids: Set[str], ctx) -> None:
    """Create SETask nodes and SUITABLE_FOR (model -> task) relationships from the CSV.

    Task -> activity links are no longer derived here; they are imported from
    the JSON taxonomy in :func:`_import_se_activity_links`.
    """
    task_rows: List[Dict[str, str]] = []
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
        reasoning = clean(row.get("se_reasoning"))

        if task_label:
            task_rows.append({"id": _normalize_mapping_id(task_label), "label": task_label})
            relationship_rows.append({
                "model_id": model_id,
                "se_task_id": _normalize_mapping_id(task_label),
                "se_task_label": task_label,
                "reasoning": reasoning,
            })

    task_seen: Set[str] = set()

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

    log.info("SE task rows imported: %d SUITABLE_FOR relationships created/merged", created)


def _link_task_activity(session, task_id: str, task_label: str, activity_id: str, activity_label: str) -> str:
    """Ensure the activity exists and link the task to it.

    The task is looked up case-insensitively (by normalized id or label). If no
    matching :SETask exists in the graph, a new node is created. Returns either
    ``"created"`` or ``"linked"`` depending on whether a new node was added.
    """

    # Case-insensitive lookup of an existing SETask.
    record = session.run(
        """
        MATCH (t:SETask)
        WHERE toLower(t.id) = $task_id
           OR toLower(coalesce(t.label, '')) = $task_label_lower
        RETURN t.id AS id
        LIMIT 1
        """,
        task_id=task_id,
        task_label_lower=task_label.lower(),
    ).single()

    if record is None:
        # Task not in graph -> create a new SETask node and connect the activity.
        session.run(
            """
            MERGE (t:SETask {id: $task_id})
            SET t.label = coalesce(t.label, $task_label)
            WITH t
            MATCH (a:SEActivity {id: $activity_id})
            MERGE (t)-[:USED_FOR]->(a)
            """,
            task_id=task_id,
            task_label=task_label,
            activity_id=activity_id,
        )
        return "created"

    # Task already exists -> just connect it to the activity.
    session.run(
        """
        MATCH (t:SETask {id: $existing_id})
        MATCH (a:SEActivity {id: $activity_id})
        MERGE (t)-[:USED_FOR]->(a)
        """,
        existing_id=record["id"],
        activity_id=activity_id,
    )
    return "linked"



def _import_se_activity_links(session, activity_mappings: Iterable[Tuple[str, str]]) -> None:
    """Import SETask -> SEActivity (USED_FOR) links from the JSON taxonomy."""
    created = 0
    linked = 0
    skipped = 0
    seen: Set[str] = set()

    for mapping in activity_mappings:
        task_label = clean(mapping.get("seTask_id"))
        activity_label = clean(mapping.get("seActivity_id"))

        if not task_label or not activity_label:
            skipped += 1
            continue

        task_id = _normalize_mapping_id(task_label)
        activity_id = _normalize_mapping_id(activity_label)

        key = f"{task_id}->{activity_id}"
        if key in seen:
            continue
        seen.add(key)

        result = _link_task_activity(session, task_id, task_label, activity_id, activity_label)
        if result == "created":
            created += 1
        else:
            linked += 1

    log.info(
        "SE task/activity links from JSON: %d new SETask nodes created, %d existing tasks linked, %d skipped",
        created, linked, skipped,
    )


def run_pass8_fill(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """Import benchmark and SE context data for SEModel nodes only."""
    log.info("PASS 8: importing benchmark and SE context data…")

    activity_mappings = _load_activity_mapping_json(SE_ACTIVITY_MAPPING_JSON)

    with ctx.writer.driver.session() as session:
        _import_se_activity_links(session, activity_mappings)

    log.info("PASS 8 complete.")