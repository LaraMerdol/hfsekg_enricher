"""
parsers/helpers.py
==================
Pure utility functions shared across all parsers and passes.

None of these functions have side effects or external dependencies — they
are easy to unit-test independently.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> Optional[str]:
    """
    Coerce *value* to a stripped string.

    Returns ``None`` for ``None``, empty strings, and whitespace-only strings.
    This is the canonical normaliser used throughout the pipeline before
    persisting any string to Neo4j.
    """
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


# ---------------------------------------------------------------------------
# Row de-duplication
# ---------------------------------------------------------------------------

def unique_rows(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """
    Return *rows* with duplicates removed based on the given *keys*.

    The first occurrence of each unique key-tuple is kept; subsequent
    duplicates are silently dropped.

    Parameters
    ----------
    rows:
        Input list of dicts.
    keys:
        Tuple of dict keys that together form the uniqueness signature.
    """
    seen: Set[tuple] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        sig = tuple(row.get(k) for k in keys)
        if sig not in seen:
            seen.add(sig)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_rows_to_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write *rows* to a CSV file at *path*.

    Column names are derived from the union of all keys across all rows,
    sorted alphabetically for reproducibility.  Does nothing if *rows* is
    empty.
    """
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Username extraction
# ---------------------------------------------------------------------------

def extract_username(raw: Any) -> Optional[str]:
    """
    Extract a username string from a variety of input shapes.

    Handles plain strings, dicts (with common key names), and objects with
    ``username`` / ``name`` / ``id`` attributes.  Returns ``None`` when no
    username can be found.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return clean(raw)
    if isinstance(raw, dict):
        return clean(
            raw.get("user")
            or raw.get("username")
            or raw.get("name")
            or raw.get("id")
        )
    return clean(
        getattr(raw, "user",     None)
        or getattr(raw, "username", None)
        or getattr(raw, "name",     None)
        or getattr(raw, "id",       None)
    )


def extract_username_from_liker(raw: Any) -> Optional[str]:
    """
    Variant of :func:`extract_username` used when iterating over liker
    objects returned by ``HfApi.list_repo_likers``.

    The liker payload uses slightly different key names so this function
    checks both orderings.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return clean(raw)
    if isinstance(raw, dict):
        return clean(
            raw.get("username")
            or raw.get("user")
            or raw.get("name")
            or raw.get("id")
        )
    return clean(
        getattr(raw, "username", None)
        or getattr(raw, "name",     None)
        or getattr(raw, "id",       None)
    )
