"""
config.py
=========
Central configuration for the HFSEKG graph enrichment pipeline.

All tunable constants live here so that no other module needs to be
edited for environment-specific changes (tokens, DB URIs, feature flags).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "01234567"

# ---------------------------------------------------------------------------
# Hugging Face API tokens
# One token is assigned exclusively to one worker thread, which eliminates
# cross-token rate-limit contention entirely.
# ---------------------------------------------------------------------------
HF_TOKENS: list[str] = [

]

if not HF_TOKENS or any(t.startswith("hf_TOKEN_") for t in HF_TOKENS):
    raise ValueError("Replace placeholder HF_TOKENS with real tokens before running.")

# ---------------------------------------------------------------------------
# Rate limiting  (per token, safe defaults)
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS       = 400   # max requests allowed …
RATE_LIMIT_WINDOW_SECONDS = 400   # … within this many seconds

# ---------------------------------------------------------------------------
# Parallelism  (1 worker per token — keeps rate limits independent)
# ---------------------------------------------------------------------------
MAX_WORKERS = len(HF_TOKENS)

# ---------------------------------------------------------------------------
# Feature flags — toggle individual pipeline passes without code changes
# ---------------------------------------------------------------------------
FETCH_MODEL_README              = True
FETCH_DATASET_README            = True
ENABLE_SPACES_PASS              = True   # Pass 3
ENABLE_COLLECTION_PASS          = True   # Pass 4
ENABLE_PAPER_PASS               = True   # Pass 5
ENABLE_SOCIAL_AND_LIKERS_PASS   = True   # Pass 6  (not yet implemented)
ENABLE_SE_CONTEXT_PASS          = True   # Pass 8
CREATE_MISSING_BASE_MODEL_STUBS = False   # create stub Model nodes for unknown base models

# ---------------------------------------------------------------------------
# Write-buffer flush threshold
# The RepairBuffer is written to Neo4j when total_size() exceeds this.
# ---------------------------------------------------------------------------
FLUSH_THRESHOLD = 50

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ALL_MODELS_CSV   = Path(r"C:\Users\laram\Downloads\hfsekg_enricher_v2\AllSEmodels.csv")
CHECKPOINT_FILE  = Path(r"C:\Users\laram\Downloads\hfsekg_enricher_v2\export.csv")
BENCHMARK_DATA_CSV = Path(r"C:\Users\laram\Downloads\hfsekg_enricher_v2\benchmark_updated.csv")
SE_TASK_MAPPING_CSV = Path(r"C:\Users\laram\Downloads\hfsekg_enricher_v2\se_task_mappings_updated.csv")
AUDIT_CSV        = Path("audit.csv")
ERRORS_CSV       = Path("errors.csv")

# ---------------------------------------------------------------------------
# Pass 0 — Lineage BFS configuration
# ---------------------------------------------------------------------------

# Single HF token used by the lineage finder (uses its own HfApi instance,
# not the multi-worker pool used by later passes).
LINEAGE_HF_TOKEN: str = HF_TOKENS[0]

# Maximum ancestor traversal depth.  30 is deliberately generous; most real
# lineage chains are < 5 hops.
LINEAGE_MAX_DEPTH: int = 30

# Maximum HF API retries per model before giving up.
LINEAGE_MAX_RETRIES: int = 5

# Only follow a discovered base-model into the BFS queue when its edge has
# at least this confidence level.  Choices: "low" | "medium" | "high".
LINEAGE_MIN_TRAVERSE_CONFIDENCE: str = "medium"

# OpenAI model used for the LLM-fallback inference tier.
LINEAGE_OPENAI_MODEL: str = "gpt-4o-mini"

# OpenAI API key for the LLM inference tier.  Set to None to disable LLM
# inference entirely (explicit + heuristic tiers will still run).
LINEAGE_OPENAI_API_KEY: str | None = (

)

# Enable the LLM inference tier (requires LINEAGE_OPENAI_API_KEY to be set).
LINEAGE_USE_OPENAI: bool = True

# Whether to pause before each edge and ask for human confirmation.
# Always False in automated pipeline runs.
LINEAGE_INTERACTIVE: bool = False

# Output CSV that receives every discovered lineage edge (append-friendly,
# crash-safe: rows are fsynced after each write).
LINEAGE_EDGES_CSV = Path("lineage_results.csv")

# Write high-confidence edges directly to Neo4j during Pass 0 (in addition
# to the CSV).  Set False to produce CSV-only output and import later.
LINEAGE_WRITE_TO_NEO4J: bool = True
