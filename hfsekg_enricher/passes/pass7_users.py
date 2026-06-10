"""
passes/pass7_users.py
=====================
Pass 7 — User enrichment and organization detection.

Iterates over all User nodes in the graph that have not yet been enriched
(``type IS NULL`` stub nodes), fetches their HF user overview, and:

- Updates the User node with full profile data (follower counts, pro status…).
- Creates AFFILIATED_WITH edges to any organisations listed in the user's
  profile.
- Creates FOLLOWS edges to other users and organisations.
- If the HF user-overview endpoint returns nothing but the org-overview
  endpoint succeeds, relabels the node from User → Organization.

Entry point
-----------
``run_pass7_users(ctx)``
"""

from __future__ import annotations

import logging
from typing import Optional

from models import user_stub, org_stub
from parsers import extract_username, clean
from parsers.bundle_parsers import parse_user_overview

log = logging.getLogger(__name__)


def run_pass7_users(ctx: "PipelineContext") -> None:  # type: ignore[name-defined]
    """
    Enrich User nodes and detect mis-labelled Organization nodes.

    Parameters
    ----------
    ctx:
        Shared :class:`PipelineContext`.
    """
    usernames = sorted(ctx.writer.get_graph_usernames())
    log.info("PASS 7: enriching %d users…", len(usernames))

    for idx, username in enumerate(usernames):
        worker = ctx.workers[idx % len(ctx.workers)]

        try:
            if not username or " " in username:
                continue  # skip malformed usernames

            user_info = worker.fetch_user_overview(username)

            if user_info:
                _enrich_user(username, user_info, worker, ctx)
                ctx.flush_if_needed()
                continue

            # If user endpoint returns nothing, try org endpoint
            org_info = worker.fetch_org_overview(username)
            if org_info:
                fullname = clean(
                    org_info.get("fullname")
                    or org_info.get("fullName")
                    or org_info.get("name")
                )
                ctx.writer.relabel_user_to_org(username, fullname)
                ctx.existing_users.discard(username)
                ctx.existing_orgs.add(username)

        except Exception as exc:
            ctx.error_rows.append({
                "stage": "user_enrichment", "id": username,
                "status": "error", "error": str(exc),
            })

    ctx.flush_if_needed(force=True)
    log.info("PASS 7 complete.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich_user(
    username: str,
    user_info: dict,
    worker,
    ctx: "PipelineContext",  # type: ignore[name-defined]
) -> None:
    """
    Update the User node and create social edges from the overview payload.

    Creates:
    - Enriched User node (full profile fields).
    - Organization stubs + AFFILIATED_WITH edges for the user's org memberships.
    - User stubs + FOLLOWS edges for the accounts this user follows.
    """
    ctx.repair.users.append(parse_user_overview(username, user_info))

    # Affiliations from the orgs field
    for org_name in user_info.get("orgs") or []:
        org_id = extract_username(org_name)
        if not org_id:
            continue
        if org_id not in ctx.existing_orgs:
            ctx.repair.organizations.append(org_stub(org_id))
            ctx.existing_orgs.add(org_id)
        if (username, org_id) not in ctx.affiliated_with:
            ctx.repair.affiliated_with.append({"username": username, "organization_id": org_id})
            ctx.affiliated_with.add((username, org_id))

    # Following relationships
    following_rows = worker.fetch_user_following(username) or []
    for row in following_rows:
        target = extract_username(row)
        if not target or target == username:
            continue

        target_type: Optional[str] = row.get("type") if isinstance(row, dict) else None

        if target_type == "org":
            if target not in ctx.existing_orgs:
                ctx.repair.organizations.append(org_stub(target))
                ctx.existing_orgs.add(target)
            if (username, target) not in ctx.follows_org:
                ctx.repair.follows_org.append({"username": username, "organization_id": target})
                ctx.follows_org.add((username, target))
        else:
            if target not in ctx.existing_users:
                ctx.repair.users.append(user_stub(target))
                ctx.existing_users.add(target)
            if (username, target) not in ctx.follows_user:
                ctx.repair.follows_user.append({"follower_id": username, "followee_id": target})
                ctx.follows_user.add((username, target))
