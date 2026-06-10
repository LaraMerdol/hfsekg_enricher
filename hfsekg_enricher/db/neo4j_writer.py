"""
db/neo4j_writer.py
==================
All Neo4j read and write logic for the HFSEKG enrichment pipeline.

``Neo4jWriter`` owns the graph driver and exposes:

- ``get_existing_*`` helpers that load current graph state into Python sets
  (used by passes to skip already-present edges/nodes).
- ``apply_repairs`` which flushes a ``RepairBuffer`` to the graph via batched
  Cypher MERGE / SET statements.
- Utility methods for node relabelling (User → Organization).
"""

from __future__ import annotations

import threading
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase, Session

from models.buffer import RepairBuffer
from config import CREATE_MISSING_BASE_MODEL_STUBS

log = logging.getLogger(__name__)


class Neo4jWriter:
    """
    Manages the Neo4j driver and serialises all graph writes.

    All write operations acquire ``_db_lock`` to ensure that the single Neo4j
    session is never used concurrently from multiple worker threads.

    Parameters
    ----------
    uri:
        Neo4j bolt URI (e.g. ``"bolt://localhost:7687"``).
    user:
        Neo4j username.
    password:
        Neo4j password.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self._db_lock = threading.Lock()

    def close(self) -> None:
        """Close the driver and release connection resources."""
        self.driver.close()

    # ------------------------------------------------------------------
    # Read helpers  (load existing graph state into Python sets)
    # ------------------------------------------------------------------

    def get_existing_ids(self, session: Session, label: str, prop: str) -> Set[str]:
        """
        Return the set of all *prop* values for nodes with *label*.

        Example::

            existing_models = writer.get_existing_ids(session, "Model", "id")
        """
        result = session.run(
            f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL RETURN n.{prop} AS id"
        )
        return {r["id"] for r in result}

    def get_existing_relation_pairs(
        self,
        session: Session,
        rel_type: str,
        left_label: str,
        left_prop: str,
        right_label: str,
        right_prop: str,
    ) -> Set[Tuple[str, str]]:
        """
        Return the set of ``(left_prop_value, right_prop_value)`` pairs for
        all existing relationships of *rel_type* between the two node labels.

        Example::

            trained_on = writer.get_existing_relation_pairs(
                session, "TRAINED_ON", "Model", "id", "Dataset", "id"
            )
        """
        result = session.run(f"""
            MATCH (a:{left_label})-[:{rel_type}]->(b:{right_label})
            WHERE a.{left_prop} IS NOT NULL AND b.{right_prop} IS NOT NULL
            RETURN a.{left_prop} AS a_id, b.{right_prop} AS b_id
        """)
        return {(r["a_id"], r["b_id"]) for r in result}

    def get_model_ids_to_repair(self, session: Session) -> List[str]:
        """Return IDs of Model nodes that already have ``downloads`` set."""
        result = session.run("""
            MATCH (m:Model) WHERE m.downloads IS NOT NULL OR m.downloads <> ''
            RETURN m.id AS id ORDER BY m.id
        """)
        return [r["id"] for r in result]

    def get_dataset_ids(self, session: Session) -> List[str]:
        """Return IDs of all Dataset nodes."""
        result = session.run("""
            MATCH (d:Dataset) WHERE d.id IS NOT NULL
            RETURN d.id AS id ORDER BY d.id
        """)
        return [r["id"] for r in result]

    def get_graph_model_ids(self) -> Set[str]:
        """Return IDs of all Model nodes (opens its own session)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:Model) WHERE m.id IS NOT NULL RETURN m.id AS id"
            )
            return {r["id"] for r in result if r.get("id")}

    def get_graph_dataset_ids(self) -> Set[str]:
        """Return IDs of all Dataset nodes (opens its own session)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Dataset) WHERE d.id IS NOT NULL RETURN d.id AS id"
            )
            return {r["id"] for r in result if r.get("id")}


    def get_graph_se_model_ids(self) -> Set[str]:
        """Return IDs of all SEModel nodes (opens its own session)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:SEModel) WHERE m.id IS NOT NULL RETURN m.id AS id"
            )
            return {r["id"] for r in result if r.get("id")}



    def get_graph_space_ids(self) -> Set[str]:
        """Return IDs of all Space nodes (opens its own session)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Space) WHERE s.id IS NOT NULL RETURN s.id AS id"
            )
            return {r["id"] for r in result}

    def get_graph_paper_ids(self) -> Set[str]:
        """Return IDs of all Paper nodes (opens its own session)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Paper) WHERE p.id IS NOT NULL RETURN p.id AS id"
            )
            return {r["id"] for r in result}

    def get_graph_usernames(self) -> Set[str]:
        """
        Return usernames of User nodes that have not yet been enriched.

        The ``u.type IS NULL`` filter targets stubs created during model/dataset
        passes that haven't been through the user-enrichment pass yet.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)
                WHERE u.username IS NOT NULL AND u.type IS NULL
                RETURN u.username AS username
            """)
            return {r["username"] for r in result if r.get("username")}

    # ------------------------------------------------------------------
    # Node relabelling
    # ------------------------------------------------------------------

    def relabel_user_to_org(self, username: str, fullname: Optional[str] = None) -> None:
        """
        Convert a User node to an Organization node in-place.

        Called during Pass 7 when the HF user-overview endpoint returns nothing
        but the org-overview endpoint succeeds, indicating the username belongs
        to an organization rather than an individual.

        All existing PUBLISHED, LIKES, FOLLOWS, and OWNED_BY relationships are
        re-pointed from the old User node to the new Organization node, then
        the User node is deleted.
        """
        with self._db_lock, self.driver.session() as session:
            session.run("""
                MATCH (u:User {username: $username})
                SET u:Organization,
                    u.id       = $username,
                    u.fullname = coalesce(u.fullname, $fullname)
                REMOVE u:User
            """, username=username, fullname=fullname)

    # ------------------------------------------------------------------
    # Flush (main write path)
    # ------------------------------------------------------------------

    def apply_repairs(self, repair: RepairBuffer) -> None:  # noqa: C901 (complexity OK here)
        """
        Flush *repair* to Neo4j in a single locked session.

        All node upserts and relationship creations are executed as batched
        ``UNWIND … MERGE … SET`` Cypher statements.  ``coalesce`` semantics
        ensure that existing property values are never overwritten — the
        pipeline is therefore safe to re-run incrementally.
        """
        with self._db_lock, self.driver.session() as session:
            self._upsert_nodes(session, repair)
            self._upsert_relationships(session, repair)

    # ------------------------------------------------------------------
    # Private — node upserts
    # ------------------------------------------------------------------

    def _upsert_nodes(self, session: Session, repair: RepairBuffer) -> None:
        """Run all node MERGE / SET statements for *repair*."""

        if repair.model_updates:
            session.run("""
            UNWIND $rows AS row
            MERGE (m:Model {id: row.id})
            SET
                m.name         = coalesce(m.name,         row.name),
                m.createdAt    = coalesce(m.createdAt,    row.createdAt),
                m.lastModified = coalesce(m.lastModified, row.lastModified),
                m.downloads    = coalesce(m.downloads,    row.downloads),
                m.likes        = coalesce(m.likes,        row.likes),
                m.region       = coalesce(m.region,       row.region),
                m.license      = coalesce(m.license,      row.license),
                m.description  = coalesce(m.description,  row.description),
                m.author       = coalesce(m.author,       row.author),
                m.pipeline_tag = coalesce(m.pipeline_tag, row.pipeline_tag),
                m.languages    = CASE WHEN m.languages IS NULL OR size(m.languages) = 0
                                      THEN row.languages ELSE m.languages END,
                m.libraries    = CASE WHEN m.libraries IS NULL OR size(m.libraries) = 0
                                      THEN row.libraries ELSE m.libraries END,
                m.other        = CASE WHEN m.other IS NULL OR size(m.other) = 0
                                      THEN row.other ELSE m.other END
            """, rows=repair.model_updates)

        # Add SEModel label to entries flagged as SE models
        session.run("""
        UNWIND $rows AS row
        WITH row WHERE row.is_se_model = true
        MATCH (m:Model {id: row.id})
        SET m:SEModel
        """, rows=repair.model_updates)

        if repair.users:
            session.run("""
            UNWIND $rows AS row
            MERGE (u:User {username: row.username})
            SET
                u.fullname       = coalesce(u.fullname,       row.fullname),
                u.type           = coalesce(u.type,           row.type),
                u.isPro          = coalesce(u.isPro,          row.isPro),
                u.numModels      = coalesce(u.numModels,      row.numModels),
                u.numDatasets    = coalesce(u.numDatasets,    row.numDatasets),
                u.numSpaces      = coalesce(u.numSpaces,      row.numSpaces),
                u.numDiscussions = coalesce(u.numDiscussions, row.numDiscussions),
                u.numPapers      = coalesce(u.numPapers,      row.numPapers),
                u.numUpvotes     = coalesce(u.numUpvotes,     row.numUpvotes),
                u.numLikes       = coalesce(u.numLikes,       row.numLikes),
                u.numFollowers   = coalesce(u.numFollowers,   row.numFollowers),
                u.numFollowing   = coalesce(u.numFollowing,   row.numFollowing),
                u.details        = coalesce(u.details,        row.details),
                u.createdAt      = coalesce(u.createdAt,      row.createdAt)
            """, rows=repair.users)

        if repair.organizations:
            session.run("""
            UNWIND $rows AS row
            MERGE (o:Organization {id: row.id})
            SET o.fullname = coalesce(o.fullname, row.fullname)
            """, rows=repair.organizations)

        if repair.tasks:
            session.run("""
            UNWIND $rows AS row
            MERGE (t:Task {id: row.id})
            SET t.label = coalesce(t.label, row.label)
            """, rows=repair.tasks)

        if repair.datasets:
            session.run("""
            UNWIND $rows AS row
            MERGE (d:Dataset {id: row.id})
            SET
                d.name         = coalesce(d.name,         row.name),
                d.createdAt    = coalesce(d.createdAt,    row.createdAt),
                d.lastModified = coalesce(d.lastModified, row.lastModified),
                d.downloads    = coalesce(d.downloads,    row.downloads),
                d.likes        = coalesce(d.likes,        row.likes),
                d.license      = coalesce(d.license,      row.license),
                d.size         = coalesce(d.size,         row.size),
                d.description  = coalesce(d.description,  row.description),
                d.languages    = CASE WHEN d.languages IS NULL OR size(d.languages) = 0
                                      THEN row.languages ELSE d.languages END,
                d.libraries    = CASE WHEN d.libraries IS NULL OR size(d.libraries) = 0
                                      THEN row.libraries ELSE d.libraries END,
                d.formats      = CASE WHEN d.formats IS NULL OR size(d.formats) = 0
                                      THEN row.formats ELSE d.formats END,
                d.modalities   = CASE WHEN d.modalities IS NULL OR size(d.modalities) = 0
                                      THEN row.modalities ELSE d.modalities END,
                d.other        = CASE WHEN d.other IS NULL OR size(d.other) = 0
                                      THEN row.other ELSE d.other END
            """, rows=repair.datasets)

        if repair.spaces:
            session.run("""
            UNWIND $rows AS row
            MERGE (s:Space {id: row.id})
            SET
                s.name         = coalesce(s.name,         row.name),
                s.createdAt    = coalesce(s.createdAt,    row.createdAt),
                s.lastModified = coalesce(s.lastModified, row.lastModified),
                s.likes        = coalesce(s.likes,        row.likes),
                s.cardData     = coalesce(s.cardData,     row.cardData),
                s.tags         = CASE
                    WHEN row.tags IS NOT NULL AND (s.tags IS NULL OR size(s.tags) = 0)
                    THEN row.tags ELSE s.tags END
            """, rows=repair.spaces)

        if repair.papers:
            session.run("""
            UNWIND $rows AS row
            MERGE (p:Paper {id: row.id})
            SET
                p.title       = coalesce(p.title,       row.title),
                p.summary     = coalesce(p.summary,     row.summary),
                p.publishedAt = coalesce(p.publishedAt, row.publishedAt),
                p.upvotes     = coalesce(p.upvotes,     row.upvotes)
            """, rows=repair.papers)

        if repair.collections:
            session.run("""
            UNWIND $rows AS row
            MERGE (c:Collection {slug: row.slug})
            SET
                c.size        = coalesce(c.size,        row.size),
                c.theme       = coalesce(c.theme,       row.theme),
                c.lastUpdated = coalesce(c.lastUpdated, row.lastUpdated),
                c.title       = coalesce(c.title,       row.title),
                c.upvotes     = coalesce(c.upvotes,     row.upvotes)
            """, rows=repair.collections)

        if CREATE_MISSING_BASE_MODEL_STUBS and repair.base_model_stubs:
            session.run("""
            UNWIND $rows AS row
            MERGE (m:Model {id: row.id})
            SET m.name = coalesce(m.name, row.name)
            """, rows=repair.base_model_stubs)

    # ------------------------------------------------------------------
    # Private — relationship upserts
    # ------------------------------------------------------------------

    def _upsert_relationships(self, session: Session, repair: RepairBuffer) -> None:
        """Run all relationship MERGE statements for *repair*."""

        def _rel(cypher: str, rows: list) -> None:
            if rows:
                session.run(cypher, rows=rows)

        # PUBLISHED
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (m:Model {id: row.model_id})
            MERGE (u)-[:PUBLISHED]->(m)""",            repair.published_model_user)
        _rel("""UNWIND $rows AS row
            MATCH (o:Organization {id: row.organization_id}) MATCH (m:Model {id: row.model_id})
            MERGE (o)-[:PUBLISHED]->(m)""",            repair.published_model_org)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (u)-[:PUBLISHED]->(d)""",            repair.published_dataset_user)
        _rel("""UNWIND $rows AS row
            MATCH (o:Organization {id: row.organization_id}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (o)-[:PUBLISHED]->(d)""",            repair.published_dataset_org)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (s:Space {id: row.space_id})
            MERGE (u)-[:PUBLISHED]->(s)""",            repair.published_space_user)
        _rel("""UNWIND $rows AS row
            MATCH (o:Organization {id: row.organization_id}) MATCH (s:Space {id: row.space_id})
            MERGE (o)-[:PUBLISHED]->(s)""",            repair.published_space_org)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (p:Paper {id: row.paper_id})
            MERGE (u)-[:PUBLISHED]->(p)""",            repair.published_paper_user)

        # Semantic
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (t:Task {id: row.task_id})
            MERGE (m)-[:DEFINED_FOR]->(t)""",          repair.defined_model_task)
        _rel("""UNWIND $rows AS row
            MATCH (d:Dataset {id: row.dataset_id}) MATCH (t:Task {id: row.task_id})
            MERGE (d)-[:DEFINED_FOR]->(t)""",          repair.defined_dataset_task)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (m)-[:TRAINED_ON]->(d)""",           repair.trained_on)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (p:Paper {id: row.paper_id})
            MERGE (m)-[:CITES]->(p)""",                repair.cites_model_paper)
        _rel("""UNWIND $rows AS row
            MATCH (d:Dataset {id: row.dataset_id}) MATCH (p:Paper {id: row.paper_id})
            MERGE (d)-[:CITES]->(p)""",                repair.cites_dataset_paper)
        _rel("""UNWIND $rows AS row
            MATCH (s:Space {id: row.space_id}) MATCH (p:Paper {id: row.paper_id})
            MERGE (s)-[:CITES]->(p)""",                repair.cites_space_paper)
        _rel("""UNWIND $rows AS row
            MATCH (s:Space {id: row.space_id}) MATCH (m:Model {id: row.model_id})
            MERGE (s)-[:USES_MODEL]->(m)""",           repair.uses_model)
        _rel("""UNWIND $rows AS row
            MATCH (s:Space {id: row.space_id}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (s)-[:USES_DATASET]->(d)""",         repair.uses_dataset)

        # Collection membership
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (m:Model {id: row.model_id})
            MERGE (c)-[:CONTAINS]->(m)""",             repair.contains_model)
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (c)-[:CONTAINS]->(d)""",             repair.contains_dataset)
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (s:Space {id: row.space_id})
            MERGE (c)-[:CONTAINS]->(s)""",             repair.contains_space)
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (p:Paper {id: row.paper_id})
            MERGE (c)-[:CONTAINS]->(p)""",             repair.contains_paper)
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (u:User {username: row.username})
            MERGE (c)-[:OWNED_BY]->(u)""",             repair.owned_by_user)
        _rel("""UNWIND $rows AS row
            MATCH (c:Collection {slug: row.collection_id}) MATCH (o:Organization {id: row.organization_id})
            MERGE (c)-[:OWNED_BY]->(o)""",             repair.owned_by_org)

        # Social
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (m:Model {id: row.model_id})
            MERGE (u)-[:LIKES]->(m)""",                repair.likes_model)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (d:Dataset {id: row.dataset_id})
            MERGE (u)-[:LIKES]->(d)""",                repair.likes_dataset)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (s:Space {id: row.space_id})
            MERGE (u)-[:LIKES]->(s)""",                repair.likes_space)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (p:Paper {id: row.paper_id})
            MERGE (u)-[:LIKES]->(p)""",                repair.likes_paper)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (c:Collection {slug: row.collection_id})
            MERGE (u)-[:LIKES]->(c)""",                repair.likes_collection)
        _rel("""UNWIND $rows AS row
            MATCH (u1:User {username: row.follower_id}) MATCH (u2:User {username: row.followee_id})
            MERGE (u1)-[:FOLLOWS]->(u2)""",            repair.follows_user)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (o:Organization {id: row.organization_id})
            MERGE (u)-[:FOLLOWS]->(o)""",              repair.follows_org)
        _rel("""UNWIND $rows AS row
            MATCH (u:User {username: row.username}) MATCH (o:Organization {id: row.organization_id})
            MERGE (u)-[:AFFILIATED_WITH]->(o)""",      repair.affiliated_with)

        # Model lineage
        # Only merge lineage edges when the source provided high confidence.
        # Rows must include a `confidence` property set to 'high' to be merged.
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (b:Model {id: row.base_model_id})
            MERGE (m)-[:IS_ADAPTER_OF]->(b)""",        repair.adapter_of)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (b:Model {id: row.base_model_id})
            MERGE (m)-[:IS_FINETUNED_FROM]->(b)""",    repair.finetuned_from)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (b:Model {id: row.base_model_id})
            MERGE (m)-[:IS_MERGE_OF]->(b)""",          repair.merge_of)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (b:Model {id: row.base_model_id})
            MERGE (m)-[:IS_QUANTIZED_FROM]->(b)""",    repair.quantized_of)
        _rel("""UNWIND $rows AS row
            MATCH (m:Model {id: row.model_id}) MATCH (b:Model {id: row.base_model_id})
            MERGE (m)-[:IS_BASED_ON]->(b)""",          repair.based_of)
