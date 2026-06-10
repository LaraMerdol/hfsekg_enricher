"""
main.py
=======
Entry point for the HFSEKG graph enrichment pipeline.

Usage::

    python main.py

All configuration (Neo4j credentials, HF tokens, feature flags, file paths)
lives in ``config.py``.
"""

import logging

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from enricher import HFGraphEnricher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    enricher = HFGraphEnricher(
        model_ids     = None,   # None → load from allModels.csv
        neo4j_uri     = NEO4J_URI,
        neo4j_user    = NEO4J_USER,
        neo4j_password= NEO4J_PASSWORD,
    )
    try:
        enricher.run()
    finally:
        enricher.close()
