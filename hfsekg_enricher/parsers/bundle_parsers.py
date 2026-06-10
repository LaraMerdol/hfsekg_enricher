"""
parsers/bundle_parsers.py
=========================
Pure parsing functions that transform raw Hugging Face API responses into
structured "bundles" — plain dicts consumed by the pipeline passes.

Every function here is a pure transformation (no I/O, no DB access) so it
can be tested independently with sample API fixtures.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from parsers.helpers import clean


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def parse_model_bundle(
    model_info: Dict[str, Any],
    readme: Optional[str],
    model_tag_map: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Parse a raw HF model-info dict into a structured bundle.

    The returned bundle contains:

    - ``model_update`` — flat dict ready to be upserted as a Model node.
    - ``author`` — the model author/publisher string (or ``None``).
    - ``task_ids`` — set of pipeline-tag / task-category strings.
    - ``datasets`` — set of dataset IDs referenced via tags.
    - ``papers`` — set of arXiv IDs referenced via ``arxiv:`` tags.
    - ``adapter_models``, ``finetune_models``, ``merge_models``,
      ``quantized_models``, ``based_models`` — sets of base model IDs for
      each lineage relationship type.

    Parameters
    ----------
    model_info:
        Raw dict returned by ``WorkerClient.fetch_model_info``.
    readme:
        Raw README.md text (or ``None`` if not fetched / not found).
    model_tag_map:
        Mapping of tag-id → ``{type, label}`` from the HF models-tags API,
        used to classify raw tag strings into semantic categories.
    """
    model_id     = model_info["id"]
    author       = clean(model_info.get("author"))
    tags         = list(model_info.get("tags") or [])
    pipeline_tag = clean(model_info.get("pipeline_tag"))

    region:        Optional[str]  = None
    license_value: Optional[str]  = None
    languages:     List[str]      = []
    libraries:     List[str]      = []
    other:         List[str]      = []

    datasets:         Set[str] = set()
    papers:           Set[str] = set()
    adapter_models:   Set[str] = set()
    finetune_models:  Set[str] = set()
    merge_models:     Set[str] = set()
    quantized_models: Set[str] = set()
    based_models:     Set[str] = set()
    task_ids:         Set[str] = set()

    if pipeline_tag:
        task_ids.add(pipeline_tag)

    for raw_tag in tags:
        tag = clean(raw_tag)
        if not tag:
            continue

        if tag.startswith(("dataset:", "datasets:")):
            dataset_id = clean(tag.split(":", 1)[1])
            if dataset_id:
                datasets.add(dataset_id)

        elif tag.startswith("arxiv:"):
            paper_id = clean(tag.split(":", 1)[1])
            if paper_id:
                papers.add(paper_id)

        elif tag.startswith("base_model:"):
            # Format:  base_model:<relation_kind>:<base_model_id>
            parts = tag.split(":")
            if len(parts) == 3:
                rel_kind, base_id = parts[1], clean(parts[2])
                if base_id:
                    {
                        "adapter":   adapter_models,
                        "finetune":  finetune_models,
                        "merge":     merge_models,
                        "quantized": quantized_models,
                    }.get(rel_kind, set()).add(base_id)

        else:
            cls      = model_tag_map.get(tag, {})
            tag_type = cls.get("type", "other")
            label    = cls.get("label", tag)

            if tag_type == "region":          region        = label
            elif tag_type == "license":       license_value = label
            elif tag_type == "language":      languages.append(label)
            elif tag_type == "library":       libraries.append(label)
            elif tag_type == "pipeline_tag":  task_ids.add(tag)
            else:                             other.append(label)

    return {
        "model_update": {
            "id":           model_id,
            "name":         model_id.split("/")[-1],
            "createdAt":    model_info.get("createdAt") or model_info.get("created_at"),
            "lastModified": model_info.get("lastModified"),
            "downloads":    model_info.get("downloads"),
            "likes":        model_info.get("likes"),
            "region":       region,
            "license":      license_value,
            "author":       author,
            "pipeline_tag": pipeline_tag,
            "description":  readme,
            "languages":    languages,
            "libraries":    libraries,
            "other":        other,
        },
        "author":           author,
        "task_ids":         task_ids,
        "datasets":         datasets,
        "papers":           papers,
        "adapter_models":   adapter_models,
        "finetune_models":  finetune_models,
        "merge_models":     merge_models,
        "quantized_models": quantized_models,
        "based_models":     based_models,
    }


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------

def parse_dataset_bundle(
    dataset_info: Dict[str, Any],
    readme: Optional[str],
    dataset_tag_map: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Parse a raw HF dataset-info dict into a structured bundle.

    The returned bundle contains:

    - ``dataset_update`` — flat dict ready to be upserted as a Dataset node.
    - ``author`` — the dataset author/publisher string (or ``None``).
    - ``task_ids`` — set of task-category strings.
    - ``papers`` — set of arXiv IDs referenced via ``arxiv:`` tags.
    """
    dataset_id = dataset_info["id"]
    author     = clean(dataset_info.get("author"))
    tags       = list(dataset_info.get("tags") or [])

    task_ids:   Set[str]  = set()
    papers:     Set[str]  = set()
    languages:  List[str] = []
    libraries:  List[str] = []
    formats:    List[str] = []
    modalities: List[str] = []
    other:      List[str] = []
    license_value: Optional[str] = None
    size:          Optional[str] = None

    for raw_tag in tags:
        tag = clean(raw_tag)
        if not tag:
            continue

        if tag.startswith("arxiv:"):
            papers.add(tag.split(":", 1)[1])
        elif tag.startswith("task_categories:"):
            task_ids.add(tag.split(":", 1)[1])

        cls      = dataset_tag_map.get(tag, {})
        tag_type = cls.get("type", "other")
        label    = cls.get("label", tag)

        if tag_type == "library":               libraries.append(label)
        elif tag_type == "license":             license_value = label
        elif tag_type == "language":            languages.append(label)
        elif tag_type == "format":              formats.append(label)
        elif tag_type == "modality":            modalities.append(label)
        elif tag_type == "size_categories":     size = label
        elif tag_type != "task_categories":     other.append(label)

    return {
        "dataset_update": {
            "id":           dataset_id,
            "name":         dataset_id.split("/")[-1],
            "createdAt":    dataset_info.get("createdAt") or dataset_info.get("created_at"),
            "lastModified": dataset_info.get("lastModified") or dataset_info.get("last_modified"),
            "downloads":    dataset_info.get("downloads"),
            "likes":        dataset_info.get("likes"),
            "license":      license_value,
            "size":         size,
            "description":  readme,
            "languages":    languages,
            "libraries":    libraries,
            "formats":      formats,
            "modalities":   modalities,
            "other":        other,
        },
        "author":   author,
        "task_ids": task_ids,
        "papers":   papers,
    }


# ---------------------------------------------------------------------------
# Space bundle
# ---------------------------------------------------------------------------

def parse_space_bundle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw HF space-info dict into a structured bundle.

    The returned bundle contains:

    - ``space_node`` — flat dict ready to be upserted as a Space node.
    - ``author`` — the space author/publisher string (or ``None``).
    - ``models`` — set of model IDs the space references.
    - ``datasets`` — set of dataset IDs the space references.
    """
    card_data = raw.get("cardData")
    return {
        "space_node": {
            "id":           raw["id"],
            "name":         raw["id"].split("/")[-1],
            "createdAt":    raw.get("createdAt") or raw.get("created_at"),
            "lastModified": raw.get("lastModified") or raw.get("last_modified"),
            "likes":        raw.get("likes", 0),
            "tags":         list(raw.get("tags") or []),
            "cardData":     json.dumps(card_data, ensure_ascii=False) if card_data is not None else None,
        },
        "author":   clean(raw.get("author")),
        "models":   {clean(x) for x in (raw.get("models")   or []) if clean(x)},
        "datasets": {clean(x) for x in (raw.get("datasets") or []) if clean(x)},
    }


# ---------------------------------------------------------------------------
# Collection bundle
# ---------------------------------------------------------------------------

def parse_collection_bundle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw HF collection-info dict into a structured bundle.

    The returned bundle contains:

    - ``collection_node`` — flat dict ready to be upserted as a Collection node.
    - ``owner`` — raw owner dict (may be user or org).
    - ``models``, ``datasets``, ``papers``, ``spaces`` — sets of item IDs
      grouped by artifact type.
    """
    result: Dict[str, Any] = {
        "collection_node": {
            "slug":        raw["slug"],
            "lastUpdated": raw.get("lastUpdated"),
            "upvotes":     raw.get("upvotes", 0),
            "title":       raw["title"],
            "size":        len(raw["items"]) if "items" in raw else 0,
            "theme":       raw["theme"],
        },
        "owner":    raw.get("owner"),
        "papers":   set(),
        "datasets": set(),
        "models":   set(),
        "spaces":   set(),
    }

    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        repo_type = item.get("repoType")
        item_id   = clean(item.get("id"))
        if not item_id:
            continue
        if repo_type == "model":    result["models"].add(item_id)
        elif repo_type == "dataset": result["datasets"].add(item_id)
        elif repo_type == "paper":   result["papers"].add(item_id)
        elif repo_type == "space":   result["spaces"].add(item_id)

    return result


# ---------------------------------------------------------------------------
# Paper bundle
# ---------------------------------------------------------------------------

def parse_paper_bundle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a raw HF paper-info dict (``/api/papers/{arxiv_id}``) into a bundle.

    The returned bundle contains:

    - ``paper_node`` — flat dict ready to be upserted as a Paper node.
    - ``author_usernames`` — set of HF usernames of authors.
    - ``author_names_only`` — list of display-only author names (no HF account).
    - ``models``, ``datasets``, ``spaces`` — sets of linked artifact IDs.
    """
    paper_id = clean(
        raw.get("id")
        or (raw.get("paper", {}).get("id") if isinstance(raw.get("paper"), dict) else None)
        or raw.get("arxivId")
        or raw.get("arxiv_id")
    )

    author_usernames: Set[str]  = set()
    author_names:     List[str] = []

    for author in raw.get("authors") or []:
        if isinstance(author, dict):
            username = clean(
                author.get("username")
                or author.get("user")
                or author.get("hf_username")
                or (author.get("name") if "/" not in (author.get("name") or "") else None)
            )
            display_name = clean(
                author.get("name") or author.get("fullname") or author.get("full_name")
            )
            if username:
                author_usernames.add(username)
            elif display_name:
                author_names.append(display_name)
        elif isinstance(author, str):
            author_names.append(author)

    # Collect linked artifacts from both top-level and nested fields
    linked_models:   Set[str] = set()
    linked_datasets: Set[str] = set()
    linked_spaces:   Set[str] = set()

    def _collect(items: list, target: Set[str]) -> None:
        for item in items or []:
            item_id = clean(item.get("id") if isinstance(item, dict) else item)
            if item_id:
                target.add(item_id)

    _collect(raw.get("models"),   linked_models)
    _collect(raw.get("datasets"), linked_datasets)
    _collect(raw.get("spaces"),   linked_spaces)

    linked = raw.get("linked") or raw.get("artifacts") or {}
    if isinstance(linked, dict):
        _collect(linked.get("models"),   linked_models)
        _collect(linked.get("datasets"), linked_datasets)
        _collect(linked.get("spaces"),   linked_spaces)

    return {
        "paper_node": {
            "id":          paper_id,
            "title":       raw.get("title"),
            "summary":     raw.get("summary") or raw.get("abstract"),
            "publishedAt": raw.get("publishedAt") or raw.get("published_at") or raw.get("published"),
            "upvotes":     raw.get("upvotes") or raw.get("num_upvotes") or 0,
        },
        "author_usernames": author_usernames,
        "author_names_only": author_names,
        "models":   linked_models,
        "datasets": linked_datasets,
        "spaces":   linked_spaces,
    }


# ---------------------------------------------------------------------------
# User overview
# ---------------------------------------------------------------------------

def parse_user_overview(username: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a raw HF user-overview response into a User node dict.

    Parameters
    ----------
    username:
        The HF username (used as the node key).
    raw:
        Raw JSON dict from ``/api/users/{username}/overview``.
    """
    return {
        "username":       username,
        "fullname":       clean(raw.get("fullname")),
        "type":           clean(raw.get("type")),
        "isPro":          raw.get("isPro"),
        "numModels":      raw.get("numModels"),
        "numDatasets":    raw.get("numDatasets"),
        "numSpaces":      raw.get("numSpaces"),
        "numDiscussions": raw.get("numDiscussions"),
        "numPapers":      raw.get("numPapers"),
        "numUpvotes":     raw.get("numUpvotes"),
        "numLikes":       raw.get("numLikes"),
        "numFollowers":   raw.get("numFollowers"),
        "numFollowing":   raw.get("numFollowing"),
        "details":        clean(raw.get("details")),
        "createdAt":      raw.get("createdAt"),
    }
