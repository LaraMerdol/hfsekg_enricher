#!/usr/bin/env python3
"""
Enrich Neo4j Organizations with Hugging Face org_type property.

This script:
1. Queries all Organization/Org nodes from Neo4j
2. For each organization, fetches its org_type from Hugging Face profile
3. Updates the Neo4j node with the org_type property
4. Outputs results to CSV and JSON

Environment variables:
  NEO4J_URI (default: bolt://localhost:7687)
  NEO4J_USER (default: neo4j)
  NEO4J_PASS (default: )

Usage:
  python neo4j_org_type_enrichment.py
  python neo4j_org_type_enrichment.py --output-dir results --delay 0.5 --limit 50
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("Install neo4j driver: pip install neo4j")

try:
    import requests
    import pandas as pd
except ImportError:
    sys.exit("Install required packages: pip install requests pandas")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from huggingface_hub import HfApi
import requests
from bs4 import BeautifulSoup

ORG_TYPES = {
    "company", "university", "non-profit", "nonprofit", "research", 
    "research group", "research lab", "lab", "laboratory", "institute",
    "institution", "academic", "education", "school", "college",
    "government", "public sector", "ngo", "foundation", "charity",
    "community", "individual", "personal", "startup", "enterprise",
    "organization", "organisation",  "association",
    "agency", "think tank", "community", "healthcare", "industry","classroom"
}




def get_properties(api: HfApi, org_name: str) -> Optional[Dict[str, Any]]:
    """Fetch Hugging Face organization properties for Neo4j enrichment."""
    try:
        org_info = api.get_organization_overview(org_name)
        return {
            "avatar_url": getattr(org_info, "avatar_url", None),
            "name": getattr(org_info, "name", None),
            "fullname": getattr(org_info, "fullname", None),
            "details": getattr(org_info, "details", None),
            "is_verified": getattr(org_info, "is_verified", None),
            "is_following": getattr(org_info, "is_following", None),
            "num_users": getattr(org_info, "num_users", None),
            "num_models": getattr(org_info, "num_models", None),
            "num_spaces": getattr(org_info, "num_spaces", None),
            "num_datasets": getattr(org_info, "num_datasets", None),
            "num_followers": getattr(org_info, "num_followers", None),
        }
    except Exception as e:
        logger.error(f"Error fetching organization properties for {org_name}: {e}")
        return None

def get_org_type(org_name):
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 ..."})
        r = session.get(
            f"https://huggingface.co/{org_name}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        if r.status_code == 429:
            logger.warning(f"Rate limited on {org_name}, waiting 30s...")
            time.sleep(30)
            return get_org_type(org_name)  # retry once
        if r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} for {org_name}")
            return None
            
        soup = BeautifulSoup(r.text, "html.parser")
        for span in soup.find_all("span", class_="capitalize"):
            text = span.get_text(strip=True).lower()
            if text in ORG_TYPES:
                return text
        return None
    except Exception as e:
        logger.error(f"Error fetching {org_name}: {e}")
        return None

@dataclass
class OrgEnrichment:
    """Result of enriching an organization with org_type."""
    org_username: str
    org_type: Optional[str]
    hf_url: str
    status: str
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Neo4jConnector:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self) -> None:
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        with self.driver.session() as session:
            session.run("RETURN 1").consume()
        logger.info(f"✓ Connected to Neo4j: {self.uri}")

    def query(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def write(self, cypher: str, **params: Any) -> int:
        """Execute a write query (MERGE, CREATE, SET, etc.)"""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return result.consume().counters.properties_set

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            logger.info("Connection closed")


def get_all_organizations(connector: Neo4jConnector, limit: int = 0) -> List[Dict[str, Any]]:
    """Query all Organization and Org nodes from Neo4j."""
    query = """
    MATCH (o: Organization)
    WHERE o.org_type IS NULL OR o.org_type = ''   // Only get orgs that haven't been enriched yet
    RETURN coalesce(o.username, o.id) AS username
    ORDER BY username
    """
    rows = connector.query(query)
    if limit > 0:   
        return rows[:limit]
    return rows


def update_org_with_type(
    connector: Neo4jConnector, 
    org_username: str, 
    org_type: str,
    properties: Optional[Dict[str, Any]] = None
) -> bool:
    """Update an organization node with org_type and Hugging Face organization properties."""
    set_clauses = ["o.org_type = $org_type"]
    params: Dict[str, Any] = {
        "org_username": org_username,
        "org_type": org_type,
    }

    if properties:
        property_mapping = {
            "avatar_url": "avatar_url",
            "name": "name",
            "fullname": "fullname",
            "details": "details",
            "is_verified": "verified",
            "is_following": "is_following",
            "num_users": "num_users",
            "num_models": "num_models",
            "num_spaces": "num_spaces",
            "num_datasets": "num_datasets",
            "num_followers": "followers",
        }

        for source_key, neo4j_key in property_mapping.items():
            value = properties.get(source_key)
            if value is not None:
                set_clauses.append(f"o.{neo4j_key} = ${neo4j_key}")
                params[neo4j_key] = value

    query = f"""
    MATCH (o)
    WHERE o.username = $org_username
    OR (o.username IS NULL AND o.id = $org_username)
    SET {', '.join(set_clauses)}
    RETURN true
    """
    try:
        result = connector.query(query, **params)
        return len(result) > 0
    except Exception as e:
        logger.error(f"Failed to update {org_username}: {e}")
        return False


def enrich_organization(
    connector: Neo4jConnector,
    org_username: str,
    delay: float,
    api: HfApi
) -> OrgEnrichment:
    """Fetch org_type from HF and update Neo4j."""
    hf_url = f"https://huggingface.co/{org_username}"
    
    logger.info(f"Fetching org type for: {org_username}")
    
    try:
        org_type = get_org_type(org_username)
        properties =get_properties(api, org_username)
        if org_type is None:
            success = update_org_with_type(connector, org_username, '', properties)
            return OrgEnrichment(
                org_username=org_username,
                org_type=None,
                hf_url=hf_url,
                status="not_found",
                error="No org type detected in HF profile"
            )
        
        # Update Neo4j
        success = update_org_with_type(connector, org_username, org_type, properties)
        
        if delay > 0:
            time.sleep(delay)
        
        return OrgEnrichment(
            org_username=org_username,
            org_type=org_type,
            hf_url=hf_url,
            status="updated" if success else "fetch_ok_update_failed",
            error=None
        )
    
    except Exception as e:
        if delay > 0:
            time.sleep(delay)
        
        return OrgEnrichment(
            org_username=org_username,
            org_type=None,
            hf_url=hf_url,
            status="error",
            error=str(e)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j organizations with Hugging Face org_type property."
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-pass", default=os.getenv("NEO4J_PASS", "01234567"))
    parser.add_argument("--output-dir", default=str(Path.cwd()))
    parser.add_argument("--delay", type=float, default=2, help="Delay between requests (seconds)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of orgs (0 = all)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    connector = Neo4jConnector(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    
    try:
        connector.connect()
        
        # Get all organizations
        organizations = get_all_organizations(connector, limit=args.limit)
        logger.info(f"Found {len(organizations)} organizations")
        
        results: List[OrgEnrichment] = []
        api = HfApi(token="")
        for idx, org_row in enumerate(organizations, start=1):

            org_username = org_row.get("username", "")

            
            logger.info(f"[{idx}/{len(organizations)}] Processing: {org_username}")

            
            enrichment = enrich_organization(
                connector=connector,
                org_username=org_username,
                delay=args.delay,
                api=api
            )
            results.append(enrichment)
            
            if enrichment.status == "updated":
                logger.info(f"  ✓ Updated with org_type: {enrichment.org_type}")
            elif enrichment.status == "not_found":
                logger.warning(f"  ✗ No org_type found on HF profile")
            elif enrichment.status == "error":
                logger.error(f"  ✗ Error: {enrichment.error}")
        
        # Write outputs
        csv_path = output_dir / "neo4j_org_type_enrichment.csv"
        json_path = output_dir / "neo4j_org_type_enrichment.json"
        summary_path = output_dir / "neo4j_org_type_enrichment_summary.json"
        
        # CSV
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()) if results else [])
            if results:
                writer.writeheader()
                for result in results:
                    writer.writerow(result.to_dict())
        logger.info(f"Wrote {csv_path}")
        
        # JSON
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {json_path}")
        
        # Summary
        updated_count = sum(1 for r in results if r.status == "updated")
        not_found_count = sum(1 for r in results if r.status == "not_found")
        error_count = sum(1 for r in results if r.status == "error")
        
        summary = {
            "total_organizations": len(results),
            "updated": updated_count,
            "not_found": not_found_count,
            "errors": error_count,
            "by_status": {
                status: sum(1 for r in results if r.status == status)
                for status in set(r.status for r in results)
            }
        }
        
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote {summary_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("ENRICHMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total organizations: {summary['total_organizations']}")
        logger.info(f"Updated with org_type: {updated_count}")
        logger.info(f"Not found on HF: {not_found_count}")
        logger.info(f"Errors: {error_count}")
        
        return 0
    
    finally:
        connector.close()


if __name__ == "__main__":
    sys.exit(main())
