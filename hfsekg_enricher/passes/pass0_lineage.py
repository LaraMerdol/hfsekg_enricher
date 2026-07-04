"""
passes/pass0_lineage.py
=======================
Pass 0 — Lineage graph construction via BFS.

This pass runs **before** any metadata enrichment (Pass 1 onward).  It
reads the seed model IDs from ``allModels.csv``, then performs a
breadth-first traversal of the HF lineage graph:

  seed model → explicit / heuristic / LLM-inferred base model
             → that base model → its base model → …

up to ``LINEAGE_MAX_DEPTH`` hops, following only edges whose inferred
confidence meets ``LINEAGE_MIN_TRAVERSE_CONFIDENCE``.

Every discovered edge is streamed to ``LINEAGE_EDGES_CSV`` (one row per
edge, fsynced immediately so a crash mid-run loses at most one edge).
High-confidence edges can optionally be written directly to Neo4j via
``LINEAGE_WRITE_TO_NEO4J``.

Inference tier cascade (per model)
-----------------------------------
1. **Explicit** — ``base_model:<rel>:<id>`` tags, ``cardData.base_model``
   field, HF ``baseModels`` API field.  Confidence: ``high``.
2. **Heuristic** — regex patterns on the repo name, family-hint lookup,
   adapter/PEFT tag signals.  Confidence: ``low``–``medium``.
3. **LLM** (optional) — structured JSON prompt to the OpenAI Responses API
   when explicit metadata is absent or below ``high`` confidence.

Entry point
-----------
``run_pass0_lineage(ctx)``

The function signature matches all other pass functions so it can be called
identically from ``enricher.py``.  It does **not** write to the shared
``RepairBuffer``; instead it appends Model stubs discovered during BFS to
``ctx.repair`` so that Pass 1 can pick them up for full enrichment.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from neo4j import GraphDatabase

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment,misc]

from config import (
    ALL_MODELS_CSV,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FETCH_MODEL_README,
    LINEAGE_HF_TOKEN,
    LINEAGE_MAX_DEPTH, LINEAGE_MAX_RETRIES,
    LINEAGE_MIN_TRAVERSE_CONFIDENCE,
    LINEAGE_OPENAI_MODEL, LINEAGE_OPENAI_API_KEY,
    LINEAGE_USE_OPENAI, LINEAGE_INTERACTIVE,
    LINEAGE_EDGES_CSV, LINEAGE_WRITE_TO_NEO4J,
)
from models.stubs import base_model_stub
from parsers import parse_model_bundle
from parsers.helpers import clean
from db.worker_client import WorkerClient
from passes.pass1_models import _integrate_model_bundle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maps the short relationship-kind string to the Neo4j relationship type.
REL_KIND_TO_EDGE: Dict[str, str] = {
    "adapter":   "IS_ADAPTER_OF",
    "finetune":  "IS_FINETUNED_FROM",
    "quantized": "IS_QUANTIZED_FROM",
    "merge":     "IS_MERGE_OF",
}
VALID_REL_KINDS: Set[str] = set(REL_KIND_TO_EDGE.keys())

#: Numeric ordering used to compare confidence strings.
CONFIDENCE_ORDER: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}

#: Composite relationship-type string for Cypher ``MATCH`` clauses.
LINEAGE_REL_TYPES = "IS_FINETUNED_FROM|IS_ADAPTER_OF|IS_QUANTIZED_FROM|IS_MERGE_OF"

EDGES_CSV_HEADER = ["model_id", "base_id", "inference_type", "confidence", "inference_source"]

# Regex patterns applied to the repo-name segment for heuristic inference.
NAME_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(?P<base>.+?)(?:[-_](?:int8|int4|4bit|8bit|gptq|awq|gguf|bnb-4bit|bnb-8bit))$", re.I), "quantized", "medium"),
    (re.compile(r"^(?P<base>.+?)(?:[-_](?:lora|qlora|adapter|peft))$",                              re.I), "adapter",   "medium"),
    (re.compile(r"^(?P<base>.+?)(?:[-_](?:sft|instruct|chat|finetune|fine-tuned|finetuned))$",      re.I), "finetune",  "low"),
]

# Well-known family name fragments → canonical HF model id.
FAMILY_HINTS: Dict[str, str] = {
    "t5_large":    "google-t5/t5-large",
    "t5-base":     "google-t5/t5-base",
    "t5-small":    "google-t5/t5-small",
    "codet5-base": "Salesforce/codet5-base",
    "codet5-large":"Salesforce/codet5-large",
    "codegen-":    "Salesforce/codegen-350M-multi",
    "starcoder":   "bigcode/starcoderbase",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageCandidate:
    """
    A single candidate lineage edge discovered for a model.

    Attributes
    ----------
    child_id:
        The model whose parent we are trying to identify.
    kind:
        One of ``"adapter"``, ``"finetune"``, ``"quantized"``, ``"merge"``.
    base_id:
        The inferred parent/base model id.
    source:
        How the edge was discovered (``"explicit_tag"``, ``"explicit_card"``,
        ``"explicit_baseModels"``, ``"heuristic_name"``, ``"openai_inference"``…).
    confidence:
        ``"high"``, ``"medium"``, or ``"low"``.
    evidence:
        Human-readable string explaining why this candidate was proposed.
    """
    child_id:   str
    kind:       str
    base_id:    str
    source:     str
    confidence: str
    evidence:   str


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _clean_lower(value: object) -> Optional[str]:
    """``clean()`` + lowercase."""
    text = clean(value)
    return text.lower() if text else None


def _normalize_model_id(value: object) -> Optional[str]:
    """Strip and remove leading/trailing slashes from a model id."""
    text = clean(value)
    return text.strip("/") if text else None


def _confidence_at_least(got: str, need: str) -> bool:
    """Return True if *got* confidence is >= *need* confidence."""
    return CONFIDENCE_ORDER.get(got, 0) >= CONFIDENCE_ORDER.get(need, 0)


def _unique_preserve_order(items: Iterable[LineageCandidate]) -> List[LineageCandidate]:
    """Deduplicate a list of candidates while preserving first-seen order."""
    seen: Set[Tuple[str, str, str, str, str]] = set()
    out: List[LineageCandidate] = []
    for item in items:
        key = (item.child_id, item.kind, item.base_id, item.source, item.confidence)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _to_jsonable(value: Any) -> Any:
    """Recursively convert an arbitrary object to a JSON-serialisable form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def _extract_description_from_card_data(card_data: Any) -> Optional[str]:
    """Pull the ``description`` field out of a model card data payload."""
    card_data = _to_jsonable(card_data)
    if not isinstance(card_data, dict):
        return None
    description = card_data.get("description")
    if isinstance(description, str):
        description = description.strip()
        return description or None
    return clean(description)


# ---------------------------------------------------------------------------
# Explicit parsers  (high-confidence sources)
# ---------------------------------------------------------------------------

def _parse_lineage_from_tags(model_id: str, tags: Iterable[object]) -> List[LineageCandidate]:
    """
    Extract lineage from ``base_model:<kind>:<id>`` tags.

    These are the most reliable source: HF writes them automatically when a
    model card uses the ``base_model:`` YAML key.
    """
    out: List[LineageCandidate] = []
    for raw in tags or []:
        tag = clean(raw)
        if not tag or not tag.startswith("base_model:"):
            continue
        parts = tag.split(":", 2)
        if len(parts) != 3:
            continue
        _, kind, base_id_raw = parts
        kind    = kind.strip().lower()
        base_id = _normalize_model_id(base_id_raw)
        if base_id and kind in VALID_REL_KINDS:
            out.append(LineageCandidate(
                child_id=model_id, kind=kind, base_id=base_id,
                source="explicit_tag", confidence="high", evidence=tag,
            ))
    return out


def _parse_lineage_from_card_data(model_id: str, card_data: Dict[str, Any]) -> List[LineageCandidate]:
    """
    Extract lineage from ``cardData.base_model`` and
    ``cardData.base_model_relation`` fields.
    """
    out: List[LineageCandidate] = []
    card_data = _to_jsonable(card_data)
    if not isinstance(card_data, dict):
        return out

    rel = _clean_lower(card_data.get("base_model_relation")) or "finetune"
    if rel not in VALID_REL_KINDS:
        rel = "finetune"

    base_model = card_data.get("base_model")
    base_ids: List[str] = []
    if isinstance(base_model, str):
        norm = _normalize_model_id(base_model)
        if norm:
            base_ids.append(norm)
    elif isinstance(base_model, list):
        for item in base_model:
            norm = _normalize_model_id(item)
            if norm:
                base_ids.append(norm)

    for base_id in base_ids:
        out.append(LineageCandidate(
            child_id=model_id, kind=rel, base_id=base_id,
            source="explicit_card", confidence="high",
            evidence=f"cardData.base_model={base_id}; cardData.base_model_relation={rel}",
        ))
    return out


def _parse_lineage_from_base_models_field(model_id: str, base_models: Any) -> List[LineageCandidate]:
    """
    Extract lineage from the HF API ``baseModels`` field (returned when
    ``expand=["baseModels"]`` is passed to ``model_info``).
    """
    out: List[LineageCandidate] = []
    if not base_models:
        return out

    if isinstance(base_models, list):
        for item in base_models:
            if isinstance(item, str):
                base_id = _normalize_model_id(item)
                if base_id:
                    out.append(LineageCandidate(
                        child_id=model_id, kind="finetune", base_id=base_id,
                        source="explicit_baseModels", confidence="high",
                        evidence=str(item),
                    ))
            elif isinstance(item, dict):
                base_id = _normalize_model_id(
                    item.get("id") or item.get("modelId") or item.get("name")
                )
                rel = _clean_lower(
                    item.get("base_model_relation")
                    or item.get("relation")
                    or item.get("type")
                ) or "finetune"
                if rel not in VALID_REL_KINDS:
                    rel = "finetune"
                if base_id:
                    out.append(LineageCandidate(
                        child_id=model_id, kind=rel, base_id=base_id,
                        source="explicit_baseModels", confidence="high",
                        evidence=json.dumps(item, ensure_ascii=False),
                    ))
    return out


# ---------------------------------------------------------------------------
# LLM inference tier
# ---------------------------------------------------------------------------

class _OpenAILineageInferer:
    """
    Structured-JSON inference via the OpenAI Responses API.

    Called only when the explicit + heuristic tiers fail to produce any
    high-confidence candidates.  The prompt asks the model to return a ranked
    list of ``{kind, base_id, confidence, evidence}`` objects in JSON.
    """

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.enabled = bool(api_key and OpenAI is not None)
        self.model   = model
        self.client  = OpenAI(api_key=api_key) if self.enabled else None  # type: ignore[misc]

    def infer(self, model_id: str, model_info: Dict[str, Any]) -> List[LineageCandidate]:
        """Return LLM-inferred candidates for *model_id*, or ``[]`` if disabled."""
        if not self.enabled:
            return []

        card_data   = _to_jsonable(model_info.get("cardData") or {})
        tags        = list(model_info.get("tags") or [])
        description = clean(model_info.get("description"))

        card_summary: Dict[str, Any] = {}
        if isinstance(card_data, dict):
            for key in ("base_model", "base_model_relation", "model-index", "tags",
                        "datasets", "language", "license", "pipeline_tag",
                        "library_name", "widget", "inference"):
                if key in card_data:
                    card_summary[key] = card_data[key]

        payload = {
            "model_id":      model_id,
            "description":   description,
            "tags":          tags,
            "card_summary":  card_summary,
            "full_card_data": card_data,
        }

        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind":       {"type": "string", "enum": sorted(VALID_REL_KINDS)},
                            "base_id":    {"type": "string"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "evidence":   {"type": "string"},
                        },
                        "required": ["kind", "base_id", "confidence", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

        prompt = f"""You are a Hugging Face model lineage classifier. Given model metadata, identify the single most likely base/parent model.

BEFORE SEARCHING FOR A PARENT — check these disqualifiers. If ANY match, return empty candidates immediately:

- Card says "non-official", "unofficial repo", "mirror of", "re-upload of", "copy of", or similar → this is a copy, not a derivative
- Card says "trained from scratch", "randomly initialized", "no base model" → root model, no parent

SCAN THIS METADATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}

EVIDENCE TO LOOK FOR (in priority order):
1. Explicit phrases in description: "fine-tuned from [ID]", "based on [ID]", "adapter of [ID]", "quantized from [ID]", "merged from [ID] and [ID]", "original model: [ID]"
2. Explicity stated model quantized, adapter , merged, finetuned from another model and give link or the nameof the base model in the card or description
3. Model ID pattern: if the repo name is "[base]-lora", "[base]-gguf", "[base]-4bit", "[base]-instruct", "[base]-finetuned" — the base is likely [base]
   → For quantized repos: the parent must be an original fp16/bf16 model, never another quantized repo

RELATIONSHIP KINDS:
- finetune: trained further on top of a single base model
- adapter: LoRA/PEFT/QLoRA adapter weights only (small param count, PEFT library tag)
- quantized: GGUF/GPTQ/AWQ/ExLlamaV2/FP8/CTranslate2/4bit/8bit compression
- merge: weights merged from TWO OR MORE source models

OUTPUT RULES:
- finetune / adapter / quantized: return exactly 1 candidate (these always have one parent)
- merge: return one candidate PER source model mentioned — each with kind=merge
- If evidence is ambiguous or absent → return empty candidates array
- confidence=high only for explicit text like "fine-tuned from X" or card field base_model=X
- confidence=medium for strong name patterns like "[base]-gguf"
- confidence=low for weak hints only
- NEVER invent a base model not mentioned in the metadata

Below are eight worked examples showing how to apply the checklist. Each shows the reasoning trace followed by the final JSON.
 
EXAMPLE 1 (finetune, explicit statement):
Metadata: model_id="acme/chatty-7b", description="Fine-tuned from meta-llama/Llama-2-7b-hf on our internal instruction dataset."
Reasoning: Step 1, no disqualifier phrases present. Step 2, description explicitly states "Fine-tuned from meta-llama/Llama-2-7b-hf" - clear explicit statement, single base. Step 4, kind=finetune, confidence=high.
Answer: {{"candidates": [{{"kind": "finetune", "base_id": "meta-llama/Llama-2-7b-hf", "confidence": "high", "evidence": "description states fine-tuned from meta-llama/Llama-2-7b-hf"}}]}}
 
EXAMPLE 2 (adapter, explicit statement):
Metadata: model_id="acme/llama2-7b-medical-lora", tags=["peft", "lora"], card_summary.base_model="meta-llama/Llama-2-7b-hf"
Reasoning: Step 1, no disqualifiers. Step 2, card field base_model names the parent directly and peft/lora tags confirm this is adapter weights, not a full finetune. Step 4, kind=adapter, confidence=high.
Answer: {{"candidates": [{{"kind": "adapter", "base_id": "meta-llama/Llama-2-7b-hf", "confidence": "high", "evidence": "base_model field plus peft/lora tags"}}]}}
 
EXAMPLE 3 (quantized, explicit statement):
Metadata: model_id="TheBloke/Llama-2-7B-Chat-GGUF", description="GGUF quantized version of meta-llama/Llama-2-7b-chat-hf."
Reasoning: Step 1, no disqualifiers. Step 2, description explicitly names the original fp16 model being quantized. Step 4, kind=quantized, confidence=high.
Answer: {{"candidates": [{{"kind": "quantized", "base_id": "meta-llama/Llama-2-7b-chat-hf", "confidence": "high", "evidence": "description states GGUF quantized version of meta-llama/Llama-2-7b-chat-hf"}}]}}
 
EXAMPLE 4 (merge, explicit statement with two sources):
Metadata: model_id="acme/frankenmerge-13b", description="Merged from mistralai/Mistral-7B-v0.1 and NousResearch/Nous-Hermes-2 using SLERP."
Reasoning: Step 1, no disqualifiers. Step 2, description names two source models joined by a merge method. Step 4, kind=merge for each source, confidence=high.
Answer: {{"candidates": [{{"kind": "merge", "base_id": "mistralai/Mistral-7B-v0.1", "confidence": "high", "evidence": "description lists as a merge source"}}, {{"kind": "merge", "base_id": "NousResearch/Nous-Hermes-2", "confidence": "high", "evidence": "description lists as a merge source"}}]}}
 
EXAMPLE 5 (quantized, name pattern only):
Metadata: model_id="acme/openhermes-2.5-mistral-7b-4bit", description="4-bit quantized weights for local inference.", tags=["4bit"]
Reasoning: Step 1, no disqualifiers. Step 2, no explicit base model named anywhere in the card. Step 3, repo name follows "[base]-4bit" pattern where [base]="openhermes-2.5-mistral-7b" plausibly resolves to an existing fp16 model. Step 4, kind=quantized, confidence=medium since this rests on a name pattern rather than an explicit statement.
Answer: {{"candidates": [{{"kind": "quantized", "base_id": "teknium/OpenHermes-2.5-Mistral-7B", "confidence": "medium", "evidence": "repo name follows [base]-4bit pattern"}}]}}
 
EXAMPLE 6 (negative case - mirror upload disqualifier):
Metadata: model_id="community-mirror/llama-2-7b-chat", description="This is a non-official mirror of the original Llama 2 7B chat weights for faster regional download."
Reasoning: Step 1, description explicitly says "non-official mirror of" - this is a disqualifier. Stop immediately.
Answer: {{"candidates": []}}
 
EXAMPLE 7 (negative case - trained from scratch disqualifier):
Metadata: model_id="acme/proprietary-encoder-v1", description="Trained from scratch on our proprietary corpus with a randomly initialized transformer encoder."
Reasoning: Step 1, description explicitly says "trained from scratch" and "randomly initialized" - this is a disqualifier. Stop immediately.
Answer: {{"candidates": []}}
 
EXAMPLE 8 (negative case - name-pattern hallucination):
Metadata: model_id="acme/best-instruct-model", description="Our top instruction-tuned model for chat.", tags=["text-generation"]

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

        try:
            response = self.client.responses.create(  # type: ignore[union-attr]
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type":   "json_schema",
                        "name":   "lineage_candidates",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            data = json.loads(response.output_text)
        except Exception as exc:
            log.warning("OpenAI inference failed for %s: %s", model_id, exc)
            return []

        out: List[LineageCandidate] = []
        for item in data.get("candidates", []):
            kind       = _clean_lower(item.get("kind"))
            base_id    = _normalize_model_id(item.get("base_id"))
            confidence = _clean_lower(item.get("confidence"))
            evidence   = clean(item.get("evidence")) or "llm inference"
            if (kind in VALID_REL_KINDS and base_id
                    and confidence in CONFIDENCE_ORDER
                    and base_id != model_id):
                out.append(LineageCandidate(
                    child_id=model_id, kind=kind, base_id=base_id,
                    source="openai_inference", confidence=confidence,
                    evidence=evidence,
                ))
        return _unique_preserve_order(out)


# ---------------------------------------------------------------------------
# Interactive override helper
# ---------------------------------------------------------------------------

def _prompt_user_override(candidate: LineageCandidate) -> Optional[LineageCandidate]:
    """
    Pause and ask the user to accept, skip, or override a candidate edge.

    Only used when ``LINEAGE_INTERACTIVE=True``.
    """
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Child   : {candidate.child_id}")
    print(f"  Base    : {candidate.base_id}")
    print(f"  Kind    : {candidate.kind}")
    print(f"  Conf    : {candidate.confidence}")
    print(f"  Source  : {candidate.source}")
    print(f"  Evidence: {candidate.evidence[:120]}")
    print(f"{sep}")
    print("  /                  -> accept as-is")
    print("  skip               -> discard this edge")
    print("  <model_id> <kind>  -> override")
    print(f"{sep}")

    while True:
        try:
            raw = input("  Your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Interrupted — accepting candidate as-is]")
            return candidate

        if raw in ("", "/"):
            return candidate
        if raw.lower() == "skip":
            log.info("User skipped edge: %s -[%s]-> %s",
                     candidate.child_id, candidate.kind, candidate.base_id)
            return None
        parts = raw.split()
        if len(parts) == 2:
            override_base = _normalize_model_id(parts[0])
            override_kind = parts[1].strip().lower()
            if override_kind not in VALID_REL_KINDS:
                print(f"  Unknown kind '{override_kind}'. Valid: {sorted(VALID_REL_KINDS)}")
                continue
            if not override_base:
                print("  Invalid model id.")
                continue
            return LineageCandidate(
                child_id=candidate.child_id, kind=override_kind,
                base_id=override_base, source="user_override",
                confidence="high",
                evidence=f"manually set by user (was: {candidate.base_id} / {candidate.kind})",
            )
        print("  Unrecognised input. Type '/', 'skip', or '<model_id> <kind>'.")


# ---------------------------------------------------------------------------
# LineageFinder — the BFS engine
# ---------------------------------------------------------------------------

class LineageFinder:
    """
    Breadth-first lineage extractor for a set of seed model IDs.

    Responsibilities
    ----------------
    - Fetch model metadata from the HF API (with retry + rate-limit handling).
    - Resolve/canonicalise model IDs through a three-tier pipeline:
        1. Exact/case-insensitive Neo4j lookup.
        2. HF model_info (case-insensitive).
        3. Name-segment–based Neo4j search, then HF popularity search.
    - Assemble lineage candidates from three inference tiers:
        1. Explicit (tags, cardData, baseModels field).
        2. Heuristic (name patterns, family hints, tag signals).
        3. LLM (OpenAI structured JSON, when enabled and needed).
    - Write validated edges to a CSV (and optionally Neo4j).
    - Enqueue newly discovered base-model IDs for BFS traversal.

    Parameters
    ----------
    neo4j_uri / neo4j_user / neo4j_password:
        Neo4j connection details used for ID resolution and optional writes.
    hf_token:
        Hugging Face API token for model_info calls.
    max_depth:
        Maximum BFS depth.
    max_retries:
        Per-model HF API retry budget.
    min_traverse_confidence:
        Only follow an edge into the BFS queue at or above this level.
    openai_api_key / openai_model:
        Credentials and model for the LLM inference tier.
    use_openai_inference:
        Enable/disable the LLM tier.
    interactive:
        If True, pause before each edge for human confirmation.
    edges_output_path:
        Path to the output CSV file.
    write_to_neo4j:
        If True, also MERGE high-confidence edges into Neo4j.
    """

    def __init__(
        self,
        neo4j_uri:               str,
        neo4j_user:              str,
        neo4j_password:          str,
        hf_token:                Optional[str] = None,
        max_depth:               int           = LINEAGE_MAX_DEPTH,
        max_retries:             int           = LINEAGE_MAX_RETRIES,
        min_traverse_confidence: str           = LINEAGE_MIN_TRAVERSE_CONFIDENCE,
        openai_api_key:          Optional[str] = LINEAGE_OPENAI_API_KEY,
        openai_model:            str           = LINEAGE_OPENAI_MODEL,
        use_openai_inference:    bool          = LINEAGE_USE_OPENAI,
        interactive:             bool          = LINEAGE_INTERACTIVE,
        edges_output_path:       str | Path    = LINEAGE_EDGES_CSV,
        write_to_neo4j:          bool          = LINEAGE_WRITE_TO_NEO4J,
    ) -> None:
        self.driver                  = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.hf_api                  = HfApi(token=hf_token) if hf_token else HfApi()
        self.max_depth               = max_depth
        self.max_retries             = max_retries
        self.min_traverse_confidence = min_traverse_confidence
        self.interactive             = interactive
        self.write_to_neo4j          = write_to_neo4j

        self.processed_models:   Set[str]                   = set()
        self.created_edges:      Set[Tuple[str, str, str]]  = set()
        self.not_found_models:   Set[str]                   = set()
        self._resolution_cache:  Dict[str, Tuple[Optional[str], str]] = {}

        self.openai_inferer: Optional[_OpenAILineageInferer] = (
            _OpenAILineageInferer(openai_api_key, openai_model)
            if use_openai_inference
            else None
        )

        # Reuse the same worker-client style fetch path as Pass 1 so README
        # text is available before lineage extraction runs.
        self.worker = WorkerClient(LINEAGE_HF_TOKEN, worker_id=0)
        self.worker._tag_map_placeholder = {}

        # Open the edge CSV immediately and keep it open for streaming writes.
        self.edges_output_path = Path(edges_output_path)
        self._edges_file       = self.edges_output_path.open("w", encoding="utf-8", newline="")
        self._edges_writer     = csv.writer(self._edges_file, quoting=csv.QUOTE_ALL)
        self._edges_writer.writerow(EDGES_CSV_HEADER)
        self._edges_file.flush()
        os.fsync(self._edges_file.fileno())
        self.edges_written: int = 0

    def close(self) -> None:
        """Flush and close the edge CSV, then close the Neo4j driver."""
        try:
            if self._edges_file and not self._edges_file.closed:
                self._edges_file.flush()
                self._edges_file.close()
        finally:
            self.worker.close()
            self.driver.close()

    def _process_one_model(self, model_id: str) -> Dict[str, Any]:
        """Prefetch a model with the Pass 1-style worker client path."""
        try:
            info = self.worker.fetch_model_info(model_id)
            if info is None:
                return {"model_id": model_id, "status": "not_found"}

            readme = self.worker.fetch_model_readme(model_id) if FETCH_MODEL_README else None
            bundle = parse_model_bundle(info, readme, self.worker._tag_map_placeholder)

            # Keep the richer metadata path for lineage extraction, but make sure
            # README-backed text is already available if HF metadata lacks it.
            if bundle.get("model_update", {}).get("description"):
                info["description"] = info.get("description") or bundle["model_update"]["description"]
            if readme is not None:
                info["readme"] = readme

            return {"model_id": model_id, "status": "ok", "info": info, "bundle": bundle}
        except Exception as exc:
            log.error("w%d | error on model %s: %s", self.worker.worker_id, model_id, exc)
            return {"model_id": model_id, "status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Neo4j read helpers
    # ------------------------------------------------------------------

    def _get_model_description_from_graph(self, model_id: str) -> Optional[str]:
        """Look up a model's description in Neo4j (used as fallback metadata)."""
        with self.driver.session() as session:
            record = session.run(
                "MATCH (m:Model {id: $model_id}) RETURN m.description AS description LIMIT 1",
                model_id=model_id,
            ).single()
        return clean(record.get("description")) if record else None

    def _get_canonical_model_id_from_graph(self, model_id: str) -> Optional[str]:
        """Case-insensitive Neo4j lookup; prefers exact-case match."""
        with self.driver.session() as session:
            rec = session.run(
                """
                MATCH (m:Model)
                WHERE toLower(m.id) = toLower($id)
                RETURN m.id AS id
                ORDER BY CASE WHEN m.id = $id THEN 0 ELSE 1 END
                LIMIT 1
                """,
                id=model_id,
            ).single()
        return rec["id"] if rec is not None else None

    def _find_models_by_name_suffix(self, name: str) -> List[str]:
        """Return all Model ids whose last segment matches *name* (case-insensitive)."""
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (m:Model) WHERE toLower(m.id) ENDS WITH toLower($suffix) RETURN m.id AS id",
                suffix=f"/{name}",
            )
            return [r["id"] for r in rows]

    # ------------------------------------------------------------------
    # HF API helpers
    # ------------------------------------------------------------------

    def _hf_try_model_info(self, repo_id: str) -> Optional[str]:
        """Single HF model_info call; returns the canonical id or None."""
        try:
            info = self.hf_api.model_info(repo_id)
            return getattr(info, "id", None) or repo_id
        except RepositoryNotFoundError:
            return None
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                return repo_id   # exists but gated
            log.warning("HF error for %s: %s", repo_id, exc)
            return None
        except Exception as exc:
            log.warning("HF unexpected error for %s: %s", repo_id, exc)
            return None

    def _hf_resolve_repo_id(self, repo_id: str) -> Optional[str]:
        """Try exact → lowercase → search to find the canonical HF id."""
        hit = self._hf_try_model_info(repo_id)
        if hit:
            return hit
        lowered = repo_id.lower()
        if lowered != repo_id:
            hit = self._hf_try_model_info(lowered)
            if hit:
                return hit
        try:
            models = list(self.hf_api.list_models(search=repo_id, limit=50))
        except Exception as exc:
            log.warning("HF search failed for %s: %s", repo_id, exc)
            return None
        for m in models:
            if getattr(m, "id", "").lower() == lowered:
                return m.id
        return None

    def _hf_most_popular_with_name(self, name: str) -> Optional[str]:
        """Return the most-downloaded HF model whose repo name matches *name*."""
        try:
            models = list(self.hf_api.list_models(
                search=name, sort="downloads", direction=-1, limit=25,
            ))
        except Exception as exc:
            log.warning("HF search failed for %s: %s", name, exc)
            return None
        name_lower = name.lower()
        exact = [m for m in models if getattr(m, "id", "").split("/")[-1].lower() == name_lower]
        candidates = exact or models
        return candidates[0].id if candidates else None

    # ------------------------------------------------------------------
    # ID resolution pipeline
    # ------------------------------------------------------------------

    def resolve_model_id(self, model_id: str) -> Tuple[Optional[str], str]:
        """
        Verify and canonicalise a model id before writing it to Neo4j.

        Resolution order
        ~~~~~~~~~~~~~~~~
        1. Cache hit (avoid redundant work within a single run).
        2. Neo4j exact/case-insensitive lookup → ``"in_neo4j"``.
        3. HF ``model_info`` (case-insensitive) → ``"hf_verified"``.
        4. Neo4j name-suffix search:
           - unique match → ``"neo4j_name_match"``.
           - ambiguous → continue to step 5.
        5. HF popularity search → ``"hf_name_search"`` or
           ``"neo4j_name_match: '<id>' is top-downloaded among Neo4j hits"``.

        Returns
        -------
        ``(resolved_id_or_None, status_note)``
        """
        norm = _normalize_model_id(model_id)
        if not norm:
            return None, "unresolved: empty id"

        cache_key = norm.lower()
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        # 1. Neo4j
        graph_hit = self._get_canonical_model_id_from_graph(norm)
        if graph_hit:
            result: Tuple[Optional[str], str] = (graph_hit, "in_neo4j")
            self._resolution_cache[cache_key] = result
            return result

        # 2. HF direct
        hf_hit = self._hf_resolve_repo_id(norm)
        if hf_hit:
            result = (hf_hit, "hf_verified")
            self._resolution_cache[cache_key] = result
            return result

        # 3. Name-based recovery
        name    = norm if "/" not in norm else norm.split("/", 1)[1]
        matches = self._find_models_by_name_suffix(name)
        if len(matches) == 1:
            result = (matches[0], f"neo4j_name_match: unique match for '{name}'")
            self._resolution_cache[cache_key] = result
            return result
        if len(matches) > 1:
            log.info("Ambiguous Neo4j match for '%s': %s", name, matches)

        popular = self._hf_most_popular_with_name(name)
        if popular:
            matches_lower = {x.lower() for x in matches}
            if matches and popular.lower() in matches_lower:
                result = (popular, f"neo4j_name_match: '{popular}' is top-downloaded among Neo4j hits")
            else:
                result = (popular, f"hf_name_search: top-downloaded HF model for '{name}'")
            self._resolution_cache[cache_key] = result
            return result

        result = (None, f"unresolved: no HF repo and no Neo4j match for '{name}'")
        self._resolution_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Metadata fetching
    # ------------------------------------------------------------------

    def fetch_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch rich model metadata from the HF API with retry + fallback.

        Requests ``baseModels``, ``cardData``, ``config``, ``siblings``,
        ``tags``, ``transformersInfo``, ``pipeline_tag``, and
        ``library_name`` via the ``expand`` parameter.

        Falls back to a stub dict containing only the graph-stored
        description when the HF API returns 404 but we have local data.

        Returns ``None`` when the model cannot be found anywhere.
        """
        expand        = ["baseModels", "cardData", "config", "siblings", "tags",
                         "transformersInfo", "pipeline_tag", "library_name"]
        #graph_desc    = self._get_model_description_from_graph(model_id)

        def _build(info: Any) -> Dict[str, Any]:
            return {
                "id":              clean(getattr(info, "id", None)) or model_id,
                "tags":            list(getattr(info, "tags",            []) or []),
                "cardData":        getattr(info, "cardData",        None),
                "description":     (
                    _extract_description_from_card_data(getattr(info, "cardData", None))
                    or clean(getattr(info, "description", None))
                ),
                "config":          getattr(info, "config",          None),
                "siblings":        getattr(info, "siblings",        None),
                "transformersInfo":getattr(info, "transformersInfo",None),
                "baseModels":      getattr(info, "baseModels",      None),
                "pipeline_tag":    getattr(info, "pipeline_tag",    None),
                "library_name":    getattr(info, "library_name",    None),
            }

        def _stub() -> Optional[Dict[str, Any]]:
            #if graph_desc is not None:
            #    return {"id": model_id, "tags": [], "cardData": {}, "description": graph_desc,
            #            "config": {}, "siblings": [], "transformersInfo": {},
            #            "baseModels": None, "pipeline_tag": None, "library_name": None}
            return None

        def _retry_delay(error: Exception, attempt: int) -> float:
            response     = getattr(error, "response", None)
            retry_after  = (getattr(response, "headers", {}) or {}).get("Retry-After")
            if retry_after is not None:
                try:
                    return min(120.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    pass
            status = getattr(response, "status_code", None)
            if status in (429, 503, 504):
                return min(120.0, 5.0 * (attempt + 1))
            return min(30.0, (2 ** attempt) + 0.25)

        for attempt in range(self.max_retries + 1):
            try:
                try:
                    info = self.hf_api.model_info(model_id, expand=expand)
                except TypeError:
                    info = self.hf_api.model_info(model_id)
                return _build(info)

            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404:
                    self.not_found_models.add(model_id)
                    return _stub()
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(exc, attempt))
                    continue
                log.warning("HF request failed for %s after retries (status=%s): %s", model_id, status, exc)
                return _stub()

            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(min(30, (2 ** attempt) + 0.25))
                    continue
                log.warning("HF request failed for %s after retries: %s", model_id, exc)
                return _stub()

        return None

    # ------------------------------------------------------------------
    # Candidate assembly
    # ------------------------------------------------------------------

    def _all_high_confidence(self, candidates: List[LineageCandidate]) -> bool:
        return all(c.confidence == "high" for c in candidates)

    def extract_lineage_candidates(
        self, model_id: str, model_info: Dict[str, Any]
    ) -> List[LineageCandidate]:
        """
        Run the three-tier inference cascade and return the best candidate
        set for *model_id*.

        Tier order:
        1. Explicit (tags + cardData + baseModels field).
        2. Heuristic (if explicit is absent or non-high).
        3. LLM (if still absent or non-high, and OpenAI is enabled).

        Within the final candidate list, when the same (child, kind, base)
        triple appears multiple times, the highest-confidence copy wins.
        """
        # Prefer explicit sources in this order: tags -> cardData -> baseModels.
        # If a higher-priority explicit source yields candidates, do NOT
        # include lower-priority explicit candidates to avoid mixing evidence.
        tags_candidates = _parse_lineage_from_tags(model_id, model_info.get("tags") or [])
        if tags_candidates:
            explicit: List[LineageCandidate] = tags_candidates
        else:
            card_candidates = _parse_lineage_from_card_data(model_id, model_info.get("cardData") or {})
            if card_candidates:
                explicit = card_candidates
            else:
                explicit = _parse_lineage_from_base_models_field(model_id, model_info.get("baseModels"))

        candidates: List[LineageCandidate] = []
        if explicit and self._all_high_confidence(explicit):
            candidates = explicit
        else:
            # If no high-confidence explicit candidates, fall back to LLM inference
            # when available.
            if (not candidates or not self._all_high_confidence(candidates)) and self.openai_inferer:
                candidates.extend(self.openai_inferer.infer(model_id, model_info))

        # Deduplicate: keep highest-confidence copy per (child, kind, base)
        best: Dict[Tuple[str, str, str], LineageCandidate] = {}
        for c in candidates:
            key  = (c.child_id, c.kind, c.base_id)
            prev = best.get(key)
            if prev is None or CONFIDENCE_ORDER[c.confidence] > CONFIDENCE_ORDER[prev.confidence]:
                best[key] = c
        return list(best.values())

    # ------------------------------------------------------------------
    # Candidate validation
    # ------------------------------------------------------------------

    def _validate_candidate(
        self, candidate: LineageCandidate
    ) -> Tuple[Optional[LineageCandidate], str]:
        """
        Resolve both IDs through the ID-resolution pipeline.

        Returns ``(validated_candidate, status)`` where validated_candidate
        is None if either ID cannot be resolved or a self-loop is detected.
        The validated candidate may have updated ``child_id`` / ``base_id``
        values if resolution changed the casing.
        """
        base_resolved, base_status = self.resolve_model_id(candidate.base_id)
        if not base_resolved:
            return None, f"base {base_status}"

        child_resolved, child_status = self.resolve_model_id(candidate.child_id)
        if not child_resolved:
            return None, f"child {child_status}"

        if child_resolved.lower() == base_resolved.lower():
            return None, "self-loop after resolution"

        note_bits = []
        if base_resolved  != candidate.base_id:
            note_bits.append(f"base_id '{candidate.base_id}' -> '{base_resolved}'")
        if child_resolved != candidate.child_id:
            note_bits.append(f"child_id '{candidate.child_id}' -> '{child_resolved}'")

        validated = replace(
            candidate,
            child_id  = child_resolved,
            base_id   = base_resolved,
            evidence  = (candidate.evidence or "") + (
                " | validated: " + "; ".join(note_bits) if note_bits else ""
            ),
        )
        return validated, f"ok ({base_status})"

    # ------------------------------------------------------------------
    # Neo4j write helpers
    # ------------------------------------------------------------------

    def _write_lineage_edge(self, candidate: LineageCandidate) -> None:
        """Append a single validated edge to the edges CSV and fsync."""
        inference_source = (
            candidate.source if candidate.source == "openai_inference"
            else f"{candidate.source} | {candidate.evidence}"
        )
        self._edges_writer.writerow([
            candidate.child_id,
            candidate.base_id,
            candidate.kind,
            candidate.confidence,
            inference_source,
        ])
        self._edges_file.flush()
        os.fsync(self._edges_file.fileno())
        self.edges_written += 1

    def _merge_lineage_edge(self, candidate: LineageCandidate) -> None:
        """MERGE a high-confidence edge directly into Neo4j."""
        edge_type = REL_KIND_TO_EDGE[candidate.kind]
        query = f"""
        MERGE (m:Model {{id: $child_id}})
        SET m.name = coalesce(m.name, split($child_id, "/")[-1])

        MERGE (b:Model {{id: $base_id}})
        SET b.name = coalesce(b.name, split($base_id, "/")[-1])

        MERGE (m)-[r:{edge_type}]->(b)
        ON CREATE SET
            r.source     = $source,
            r.confidence = $confidence,
            r.evidence   = $evidence,
            r.inferred   = $inferred
        ON MATCH SET
            r.source     = coalesce(r.source, $source),
            r.confidence = CASE
                WHEN r.confidence = "high" THEN "high"
                WHEN r.confidence = "medium" AND $confidence = "low" THEN "medium"
                ELSE $confidence END,
            r.evidence   = coalesce(r.evidence, $evidence),
            r.inferred   = coalesce(r.inferred, $inferred)
        """
        with self.driver.session() as session:
            session.run(
                query,
                child_id   = candidate.child_id,
                base_id    = candidate.base_id,
                source     = candidate.source,
                confidence = candidate.confidence,
                evidence   = candidate.evidence,
                inferred   = not candidate.source.startswith("explicit_"),
            )

    # ------------------------------------------------------------------
    # Main BFS loop
    # ------------------------------------------------------------------

    def run_for_seeds(
        self,
        seed_model_ids: Iterable[str],
        new_model_stubs_out: Optional[List[str]] = None,
        ctx: Optional["PipelineContext"] = None,
    ) -> None:
        """
        BFS lineage traversal starting from *seed_model_ids*.

        Algorithm
        ~~~~~~~~~
        A deque is initialised with ``(model_id, depth=0)`` for every seed.
        At each step:

        1. Pop the next ``(model_id, depth)`` pair.
        2. Skip if already processed in this run.
        3. Fetch HF metadata.
        4. Run the three-tier inference cascade.
        5. For each candidate:

           a. Optionally pause for interactive user confirmation.
           b. Validate both IDs through the resolution pipeline.
           c. Write the edge to CSV (and optionally Neo4j).
           d. If ``depth < max_depth`` and confidence >= threshold, enqueue
              the base model for further traversal.

        Parameters
        ----------
        seed_model_ids:
            Starting model IDs (all depths start at 0).
        new_model_stubs_out:
            If provided, all newly discovered base-model IDs that were not
            already in ``processed_models`` will be appended here.  Pass 1
            can use this list to ensure newly-found ancestors are also
            enriched in the same pipeline run.
        """
        seeds = [s for s in ((_normalize_model_id(x)) for x in seed_model_ids) if s]
        seeds_set = set(seeds)
        log.info("Starting BFS lineage extraction for %d seed models", len(seeds))
        not_found_seed_se_models = 0

        queue: Deque[Tuple[str, int]] = deque((s, 0) for s in seeds)
        self.processed_models.update(ctx.writer.get_graph_model_ids()) if ctx else None
        while queue:
            model_id, depth = queue.popleft()
            model_id = _normalize_model_id(model_id)
            if not model_id or model_id in self.processed_models:
                continue
            self.processed_models.add(model_id)

            prefetch = self._process_one_model(model_id)
            if prefetch.get("status") != "ok":
                if prefetch.get("status") == "not_found" and model_id in seeds_set:
                    not_found_seed_se_models += 1
                log.warning("Skipping %s after worker prefetch: %s", model_id, prefetch.get("status"))
                continue

            # If a PipelineContext was provided, integrate the parsed model
            # bundle into the shared repair buffer using the same logic as
            # Pass 1. This seeds the repair buffer so Pass 1 can pick up
            # newly discovered nodes without a separate discovery step.
            bundle = prefetch.get("bundle")
            if ctx is not None and bundle:
                try:
                    issues: List[str] = []
                    # Tag models that originated from the initial seed CSV so
                    # they receive the `SEModel` label during the write phase.
                    try:
                        if bundle and isinstance(bundle.get("model_update"), dict):
                            bundle["model_update"]["is_se_model"] = model_id in seeds_set
                    except Exception:
                        pass
                    _integrate_model_bundle(model_id, bundle, ctx, issues)
                    ctx.flush_if_needed()
                    if issues:
                        log.info("Pass0 integrated model %s into repair buffer: %s", model_id, ",".join(issues))
                except Exception as exc:
                    log.warning("Failed to integrate %s into repair buffer: %s", model_id, exc)

            model_info = self.fetch_model_metadata(model_id)
            if model_info is None:
                log.warning("Model not found on HF: %s", model_id)
                continue

            prefetch_info = prefetch.get("info") or {}
            if prefetch_info.get("description") and not model_info.get("description"):
                model_info["description"] = prefetch_info["description"]

            candidates = self.extract_lineage_candidates(model_id, model_info)
            if not candidates:
                continue

            for candidate in candidates:
                if self.interactive:
                    candidate = _prompt_user_override(candidate)
                    if candidate is None:
                        continue

                validated, status = self._validate_candidate(candidate)
                if validated is None:
                    log.warning("Skipping edge %s -[%s]-> %s : %s",
                                candidate.child_id, candidate.kind, candidate.base_id, status)
                    continue
                if validated is not candidate:
                    log.info("Validated edge %s -[%s]-> %s (%s)",
                             validated.child_id, validated.kind, validated.base_id, status)
                candidate = validated

                edge_key = (candidate.child_id.lower(), candidate.kind, candidate.base_id.lower())
                if edge_key not in self.created_edges:
                    self._write_lineage_edge(candidate)
                    if self.write_to_neo4j and candidate.confidence == "high":
                        self._merge_lineage_edge(candidate)
                    self.created_edges.add(edge_key)

                # Enqueue base model for further BFS traversal
                if depth < self.max_depth and _confidence_at_least(
                    candidate.confidence, self.min_traverse_confidence
                ):
                    base_id = candidate.base_id
                    if base_id not in self.processed_models:
                        queue.append((base_id, depth + 1))
                        if new_model_stubs_out is not None:
                            new_model_stubs_out.append(base_id)

        log.info("Lineage BFS complete.")
        log.info("Processed unique models  : %d", len(self.processed_models))
        log.info("Edges written to %s : %d", self.edges_output_path, self.edges_written)
        log.info("HF 404 model ids         : %d", len(self.not_found_models))
        log.info("Not-found seed SE models  : %d", not_found_seed_se_models)


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------

def run_pass0_lineage(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """
    Pass 0 — BFS lineage extraction.

    Reads seed model IDs from ``allModels.csv``, runs the full BFS lineage
    crawler, and registers any newly discovered base-model IDs as stubs in
    ``ctx.repair`` so that Pass 1 will fetch their full metadata.

    The lineage edges themselves are written to ``LINEAGE_EDGES_CSV``
    (and optionally Neo4j).  They are **not** routed through the
    ``RepairBuffer`` — they travel on their own I/O path so a crash mid-BFS
    never causes edge loss.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext` from ``enricher.py``.
    """
    log.info("PASS 0: BFS lineage extraction…")

    # Load seed IDs from the same CSV used by Pass 1
    import csv as _csv
    seed_ids: Set[str] = set()
    with open(ALL_MODELS_CSV, "r", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            mid = _normalize_model_id(row.get("n.id") or next(iter(row.values()), ""))
            if mid:
                seed_ids.add(mid)
    #remove the ones arleadty in neo4j ctx.writer.get_graph_se_model_ids()
    seed_ids = seed_ids - ctx.writer.get_graph_se_model_ids()
    log.info("Loaded %d seed model IDs from %s", len(seed_ids), ALL_MODELS_CSV)

    new_stubs: List[str] = []

    finder = LineageFinder(
        neo4j_uri    = NEO4J_URI,
        neo4j_user   = NEO4J_USER,
        neo4j_password = NEO4J_PASSWORD,
        hf_token     = LINEAGE_HF_TOKEN,
    )
    try:
        finder.run_for_seeds(seed_ids, new_model_stubs_out=new_stubs, ctx=ctx)
    finally:
        finder.close()

    # Register newly discovered ancestor stubs in the repair buffer so Pass 1
    # enriches them alongside the original seed models.
    added = 0
    for mid in new_stubs:
        if mid and mid not in ctx.existing_models:
            ctx.repair.base_model_stubs.append(base_model_stub(mid))
            ctx.existing_models.add(mid)
            added += 1

    if added:
        ctx.flush_if_needed(force=True)
        log.info("Pass 0 registered %d new ancestor stubs for Pass 1.", added)

    log.info("PASS 0 complete.")
