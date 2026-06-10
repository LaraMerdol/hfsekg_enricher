"""
models/stubs.py
===============
Stub factory functions.

Each function returns the minimal dict needed to MERGE a placeholder node
into Neo4j when the full metadata has not yet been fetched.  They are used
whenever a relationship is discovered to an entity whose full record will be
enriched in a later pass (or a later run).
"""

from __future__ import annotations

from typing import Any, Dict


def user_stub(username: str) -> Dict[str, Any]:
    """Minimal User node — enriched in Pass 7."""
    return {
        "username":     username,
        "fullname":     None,
        "type":         None,
        "isPro":        None,
        "numModels":    None,
        "numDatasets":  None,
        "numSpaces":    None,
        "numDiscussions": None,
        "numPapers":    None,
        "numUpvotes":   None,
        "numLikes":     None,
        "numFollowers": None,
        "numFollowing": None,
        "details":      None,
        "createdAt":    None,
    }


def org_stub(org_id: str) -> Dict[str, Any]:
    """Minimal Organization node."""
    return {"id": org_id, "fullname": None}


def task_stub(task_id: str) -> Dict[str, Any]:
    """Minimal Task node — label derived from the task identifier."""
    return {"id": task_id, "label": task_id.replace("-", " ").title()}


def dataset_stub(dataset_id: str) -> Dict[str, Any]:
    """Minimal Dataset node — enriched in Pass 2."""
    return {
        "id":           dataset_id,
        "name":         dataset_id.split("/")[-1],
        "createdAt":    None,
        "lastModified": None,
        "downloads":    None,
        "likes":        None,
        "license":      None,
        "size":         None,
        "description":  None,
        "languages":    [],
        "libraries":    [],
        "formats":      [],
        "modalities":   [],
        "other":        [],
    }


def space_stub(space_id: str) -> Dict[str, Any]:
    """Minimal Space node — enriched in Pass 3."""
    return {
        "id":           space_id,
        "name":         space_id.split("/")[-1],
        "createdAt":    None,
        "lastModified": None,
        "likes":        None,
        "tags":         [],
        "cardData":     None,
    }


def paper_stub(paper_id: str) -> Dict[str, Any]:
    """Minimal Paper node — enriched in Pass 5."""
    return {
        "id":          paper_id,
        "title":       None,
        "summary":     None,
        "publishedAt": None,
        "upvotes":     None,
    }


def model_stub(model_id: str) -> Dict[str, Any]:
    """Minimal Model node placeholder."""
    return {
        "id":           model_id,
        "name":         model_id.split("/")[-1],
        "createdAt":    None,
        "lastModified": None,
        "downloads":    None,
        "likes":        None,
        "region":       None,
        "license":      None,
        "author":       None,
        "pipeline_tag": None,
        "description":  None,
        "languages":    [],
        "libraries":    [],
        "other":        [],
    }


def base_model_stub(model_id: str) -> Dict[str, Any]:
    """Stub for a base/ancestor model referenced by lineage tags but not yet in graph."""
    return {"id": model_id, "name": model_id.split("/")[-1]}
