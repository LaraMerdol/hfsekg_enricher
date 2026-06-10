# HFSEKG Enricher

HFSEKG Enricher is a Hugging Face graph enrichment pipeline that builds and maintains a Neo4j knowledge graph from model, dataset, space, collection, paper, user, and software-engineering context data.

The project is organized as a staged enrichment pipeline. Each pass focuses on one kind of entity or one kind of graph relationship, and all passes share a common in-memory repair buffer before data is flushed to Neo4j.

## What This Project Does

At a high level, the pipeline:

1. Starts from seed model IDs in `AllSEmodels.csv`.
2. Crawls model lineage to discover ancestor models.
3. Enriches models, datasets, spaces, collections, papers, users, and organizations.
4. Imports SE-specific benchmark and task context from CSV files.
5. Writes the resulting nodes and relationships into Neo4j.

The pipeline is designed to be incremental. It reads the existing graph state at startup and avoids creating duplicate nodes or relationships when the pipeline is run again.

## Project Layout

```text
hfsekg_enricher/
  main.py                # Entry point
  enricher.py            # Pipeline orchestration and shared context
  config.py              # All runtime configuration and feature flags
  db/
    worker_client.py     # Hugging Face API client used by each worker
    neo4j_writer.py      # Neo4j read/write layer
  models/
    buffer.py            # In-memory repair buffer
    stubs.py             # Minimal stub node factories
  parsers/
    bundle_parsers.py    # Pure parsers that turn HF payloads into bundles
    helpers.py           # Shared parsing helpers
  passes/
    pass0_lineage.py     # Lineage BFS and candidate inference
    pass1_models.py      # Model metadata enrichment
    pass2_datasets.py    # Dataset enrichment
    pass3_spaces.py      # Space enrichment
    pass4_collections.py # Collection enrichment
    pass5_papers.py      # Paper enrichment
    pass7_users.py       # User and organization enrichment
    pass8_se_context.py  # Benchmark and SE context import
```

The repository also includes CSV inputs and outputs at the workspace root, such as `AllSEmodels.csv`, `benchmark_updated.csv`, `se_task_mappings_updated.csv`, `lineage_results.csv`, `audit.csv`, and `errors.csv`.

## Pipeline Overview

The orchestrator is implemented in `hfsekg_enricher/enricher.py`. It builds a shared `PipelineContext` that contains:

- Hugging Face worker clients.
- The Neo4j writer.
- The shared repair buffer.
- Existing-graph caches used for deduplication.
- Audit and error rows.
- A pause event that lets the pipeline stop worker activity while Neo4j flushes are happening.

The shared context is passed into every pass so each stage can coordinate through the same state.

## Pass-by-Pass Explanation

### Pass 0 — Lineage BFS

File: `hfsekg_enricher/passes/pass0_lineage.py`

Pass 0 is the lineage discovery stage. It starts from the seed model IDs in `AllSEmodels.csv` and performs a breadth-first traversal over model lineage links. This pass is the foundation for the rest of the pipeline because it discovers ancestor models that may not be listed in the seed CSV.

It uses a tiered inference strategy:

1. Explicit lineage signals from tags such as `base_model:<kind>:<id>`.
2. Heuristics based on repo naming, family hints, adapter tags, and PEFT-style suffixes.
3. Optional LLM fallback through the OpenAI Responses API when explicit evidence is missing.

This pass writes discovered lineage edges to `lineage_results.csv`. High-confidence edges can also be written directly to Neo4j.

Important behavior:

- Explicit candidate sources are prioritized before heuristics.
- The pass can fetch model README text and parse it into the model bundle so later inference sees richer description evidence.
- New base-model stubs discovered during traversal are queued for Pass 1.
- Traversal depth is controlled by `LINEAGE_MAX_DEPTH`.
- Traversal confidence gating is controlled by `LINEAGE_MIN_TRAVERSE_CONFIDENCE`.

### Pass 1 — Model Enrichment

File: `hfsekg_enricher/passes/pass1_models.py`

Pass 1 enriches every model listed in `AllSEmodels.csv` plus any model stubs discovered during Pass 0.

For each model, it:

- Fetches full Hugging Face model metadata.
- Optionally fetches the model README.
- Parses the payload into a structured bundle.
- Updates the Model node.
- Creates author and publication relationships.
- Creates task, dataset, paper, and lineage relationships.
- Marks CSV-listed seed models as `SEModel` so they can be handled by the SE-specific passes later.

This pass is the main source of model metadata in the graph.

### Pass 2 — Dataset Enrichment

File: `hfsekg_enricher/passes/pass2_datasets.py`

Pass 2 enriches datasets that were discovered during Pass 1, typically through dataset tags on model cards.

For each dataset it:

- Fetches Hugging Face dataset metadata.
- Optionally fetches the dataset README.
- Parses the dataset bundle.
- Updates the Dataset node.
- Creates author, task, and paper links.

### Pass 3 — Space Enrichment

File: `hfsekg_enricher/passes/pass3_spaces.py`

Pass 3 discovers Spaces that reference known models or datasets. It can operate in two modes:

- Default mode: scan spaces related to all known models and datasets.
- SE-focused mode: scan spaces only for `SEModel` nodes.

It then fetches full space metadata and creates:

- `Space` nodes.
- `USES_MODEL` relationships.
- `USES_DATASET` relationships.
- `PUBLISHED` relationships.

The `all` argument controls whether the pass scans the full graph or only the SE-model subset.

### Pass 4 — Collection Enrichment

File: `hfsekg_enricher/passes/pass4_collections.py`

Pass 4 discovers collections that contain known models, datasets, papers, or spaces.

It:

- Finds collections from item references.
- Fetches collection metadata.
- Creates `Collection` nodes.
- Creates `CONTAINS` relationships.
- Creates `OWNED_BY` relationships.

As with Pass 3, it can run either on the full graph or on the SE-model subset only.

### Pass 5 — Paper Enrichment

File: `hfsekg_enricher/passes/pass5_papers.py`

Pass 5 enriches paper nodes discovered through arXiv tags.

It:

- Fetches paper metadata.
- Updates the `Paper` node with title, summary, publication date, and other paper fields.
- Creates any missing author stubs.

### Pass 6 — Social / Likes

Pass 6 is not currently implemented.

The orchestrator documents this pass in the execution order, but there is no active pass module for it in this repository.

### Pass 7 — User and Organization Enrichment

File: `hfsekg_enricher/passes/pass7_users.py`

Pass 7 enriches `User` nodes that are still stubs. It also detects organization accounts and relabels them from `User` to `Organization` when appropriate.

It:

- Fetches user overview data.
- Updates the User node.
- Creates `AFFILIATED_WITH` edges.
- Creates `FOLLOWS` edges.
- Relabels accounts to `Organization` when the HF user endpoint is empty but the organization endpoint succeeds.

### Pass 8 — SE Context Import

File: `hfsekg_enricher/passes/pass8_se_context.py`

Pass 8 imports software-engineering context from CSV files and only attaches that data to nodes labeled `SEModel`.

It currently imports:

- Benchmark rows from `benchmark_updated.csv` as `Benchmark` nodes and `EVALUATED_ON` relationships.
- SE task mappings from `se_task_mappings_updated.csv` as `SETask` nodes and `SUITABLE_FOR` relationships.
- SE activity nodes and `USED_FOR` relationships from the same mapping file.

This pass is intended for benchmark-style SE analysis on the graph subset that has already been marked as `SEModel`.

## Supporting Submodules

### `db.worker_client`

This module provides the Hugging Face API client used by each worker thread.

Responsibilities:

- Manages one token per worker.
- Handles rate limiting.
- Retries transient network failures.
- Fetches model, dataset, space, paper, user, organization, and collection data.
- Lists spaces and collections associated with known entities.

If you are looking for where the pipeline talks to Hugging Face, this is the main module.

### `db.neo4j_writer`

This module owns all Neo4j reads and writes.

Responsibilities:

- Loads existing node IDs and relationship pairs into memory.
- Applies batched `MERGE` / `SET` writes from the repair buffer.
- Relabels users to organizations when needed.
- Exposes helper methods such as `get_graph_model_ids()` and `get_graph_se_model_ids()`.

This is the final write path into Neo4j.

### `models.buffer`

The `RepairBuffer` stores all pending graph writes in memory.

It keeps separate lists for:

- Node updates.
- Publication relationships.
- Semantic relationships.
- Collection membership.
- Social links.
- Model lineage links.

The buffer is flushed when it reaches `FLUSH_THRESHOLD` or when a pass finishes.

### `models.stubs`

This module contains small helper factories that create minimal placeholder nodes.

Examples:

- `user_stub()`
- `org_stub()`
- `task_stub()`
- `dataset_stub()`
- `space_stub()`
- `paper_stub()`
- `model_stub()`
- `base_model_stub()`

These stubs let the pipeline create relationship endpoints before the full metadata is available.

### `parsers.bundle_parsers`

This module turns raw Hugging Face API responses into structured bundles.

It contains pure parsing functions for:

- Models.
- Datasets.
- Spaces.
- Collections.
- Papers.
- User overview payloads.

The parsers do not talk to the network or Neo4j. They only transform payloads into dicts the passes can consume.

### `parsers.helpers`

This module contains utility functions shared across parsers and passes, such as:

- `clean()` for normalizing values.
- String extraction helpers.
- CSV writing helpers.

## Configuration Options

All runtime options live in `hfsekg_enricher/config.py`.

### Connection Settings

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

These configure the Neo4j connection.

### Hugging Face Tokens

- `HF_TOKENS`

Each token is assigned to one `WorkerClient`. The pipeline uses them independently so rate limiting is isolated per worker.

### Pass Toggles

- `FETCH_MODEL_README`
- `FETCH_DATASET_README`
- `ENABLE_SPACES_PASS`
- `ENABLE_COLLECTION_PASS`
- `ENABLE_PAPER_PASS`
- `ENABLE_SOCIAL_AND_LIKERS_PASS`
- `ENABLE_SE_CONTEXT_PASS`
- `CREATE_MISSING_BASE_MODEL_STUBS`

These flags control which parts of the pipeline are active.

### Lineage Settings

- `LINEAGE_MAX_DEPTH`
- `LINEAGE_MAX_RETRIES`
- `LINEAGE_MIN_TRAVERSE_CONFIDENCE`
- `LINEAGE_OPENAI_MODEL`
- `LINEAGE_OPENAI_API_KEY`
- `LINEAGE_USE_OPENAI`
- `LINEAGE_INTERACTIVE`
- `LINEAGE_WRITE_TO_NEO4J`

These control lineage BFS depth, retry behavior, confidence thresholds, OpenAI fallback, and whether high-confidence lineage edges are written directly to Neo4j.

### CSV Input Files

- `ALL_MODELS_CSV` points to the seed model list.
- `BENCHMARK_DATA_CSV` points to the benchmark import file.
- `SE_TASK_MAPPING_CSV` points to the SE task mapping file.
- `LINEAGE_EDGES_CSV` stores lineage edges discovered in Pass 0.
- `AUDIT_CSV` and `ERRORS_CSV` store pipeline output summaries.

## How the Pipeline Starts and Stops

The entry point is `hfsekg_enricher/main.py`.

It constructs `HFGraphEnricher`, runs the pipeline, and always closes resources in a `finally` block.

That means the pipeline cleans up in two steps:

1. `HFGraphEnricher.run()` executes the enabled passes.
2. `HFGraphEnricher.close()` releases Hugging Face clients and the Neo4j driver.

So yes, the application is designed to close connections automatically when it finishes or when it exits because of an error.

If you disable passes through the config flags or by changing the orchestration in `enricher.py`, the shutdown behavior still remains the same.

## Typical Data Flow

1. Load seed models from `AllSEmodels.csv`.
2. Build lineage and create stubs for discovered ancestors.
3. Enrich models and mark seed rows as `SEModel`.
4. Enrich datasets, spaces, collections, papers, and users.
5. Import SE benchmark and task context.
6. Flush all pending writes to Neo4j.
7. Write `audit.csv` and `errors.csv`.

## Outputs

The pipeline may produce or update:

- Neo4j nodes and relationships.
- `lineage_results.csv` for lineage inference output.
- `audit.csv` for processed entities and statuses.
- `errors.csv` for failed lookups or processing errors.

## Notes on Incremental Runs

The pipeline reads the current graph state at startup, which lets it avoid re-creating nodes and relationships that already exist.

That means it is safe to rerun the pipeline after partial completion, rate-limit interruptions, or a crash.

## Customizing or Disabling Parts of the Pipeline

If you want to run only part of the pipeline, the main controls are:

- The pass flags in `config.py`.
- The explicit pass calls inside `enricher.py`.
- The `all` parameter in passes such as `run_pass3_spaces()` and `run_pass4_collections()`.

For example:

- Set `ENABLE_SE_CONTEXT_PASS = False` to skip benchmark and SE context import.
- Set `ENABLE_SPACES_PASS = False` to skip Space enrichment.
- Set `ENABLE_COLLECTION_PASS = False` to skip Collection enrichment.
- Set `ENABLE_PAPER_PASS = False` to skip Paper enrichment.

## Development Notes

- The project is Python-based and uses `huggingface_hub`, `requests`, `neo4j`, and `openai`.
- Pass modules are intentionally separated so each stage can be tested or modified independently.
- The parsers are pure functions, which makes them easier to unit test.
- The repair buffer is the central mechanism that keeps writes batched and predictable.

## Practical Run Checklist

Before running the pipeline, verify the following:

1. Neo4j is running and reachable at the configured bolt URI.
2. `HF_TOKENS` contains valid Hugging Face tokens.
3. The CSV paths in `config.py` point to the correct files.
4. Any OpenAI settings needed for lineage inference are configured.
5. The desired pass flags are enabled.

Then run:

```bash
python main.py
```

If you only want to inspect the code path or adjust the pass order, edit `hfsekg_enricher/enricher.py`.

## Short Version

This project is a multi-pass HF-to-Neo4j enricher. Pass 0 discovers lineage, Pass 1 enriches models, later passes enrich datasets/spaces/collections/papers/users, and Pass 8 imports SE-specific benchmark context.

If you want this README to also include a diagram of the pass flow or a more explicit Neo4j schema section, that can be added next.