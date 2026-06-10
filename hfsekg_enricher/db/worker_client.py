"""
db/worker_client.py
===================
Per-token HTTP + HfApi client used by each parallel pipeline worker.

Design principles
-----------------
- One ``WorkerClient`` instance per Hugging Face token.  Workers never share
  a token, so rate-limit tracking is fully independent with no cross-thread
  contention on the limiter lock.
- HTTP retries are handled at the ``requests.Session`` level (via
  ``urllib3.Retry``) for transient server errors, and manually for 429
  responses that carry a ``Retry-After`` header.
- All public ``fetch_*`` / ``list_*`` methods return ``None`` (rather than
  raising) for expected missing-resource situations (404, occasionally 403).
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, RetryError
from urllib3 import Retry

from config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

import logging
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """
    Token-bucket rate limiter using a sliding window of request timestamps.

    Guarantees at most ``max_requests`` within any ``window_seconds``-wide
    window.  ``acquire()`` blocks (sleeps) until capacity is available.

    Thread-safety: the internal deque is guarded by a ``threading.Lock``.
    Because each worker holds its *own* limiter instance, contention only
    occurs when the same worker has concurrent callers — which never happens
    in the current single-threaded-per-worker design.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._lock          = threading.Lock()
        self._timestamps: deque = deque()

    def acquire(self) -> None:
        """Block until a request slot is available, then claim it."""
        while True:
            with self._lock:
                now = time.time()
                # Evict timestamps that have fallen outside the window.
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                # Must wait until the oldest timestamp expires.
                wait = self.window_seconds - (now - self._timestamps[0]) + 0.05

            time.sleep(wait)


# ---------------------------------------------------------------------------
# HTTP + HfApi client
# ---------------------------------------------------------------------------

class WorkerClient:
    """
    Self-contained HTTP + ``HfApi`` client for a single Hugging Face token.

    Each pipeline worker thread creates and owns one ``WorkerClient``
    instance, so there is no shared mutable state between workers.

    Parameters
    ----------
    token:
        A valid Hugging Face API token (``hf_…``).
    worker_id:
        Integer index used for log messages to identify which worker is
        logging.
    """

    def __init__(self, token: str, worker_id: int) -> None:
        self.token     = token
        self.worker_id = worker_id
        self.limiter   = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
        self.hf_api    = HfApi(token=token)

        # Retry adapter: 5 retries with exponential back-off for common
        # transient HTTP errors.
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[408, 500, 502, 503, 504, 522, 524],
            allowed_methods=["GET", "HEAD"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=8,
            pool_maxsize=8,
            pool_block=True,
        )

        # Authenticated session (all normal requests)
        self.session = requests.Session()
        self.session.mount("http://",  adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent":    f"HFGraphEnricher/2.0 worker={worker_id}",
            "Authorization": f"Bearer {token}",
        })

        # Unauthenticated session (fallback for 403 on public resources)
        self.session2 = requests.Session()
        self.session2.mount("https://", adapter)

    def close(self) -> None:
        """Release underlying connection pools."""
        self.session.close()
        self.session2.close()

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get_json(
        self,
        url: str,
        params: Optional[Dict] = None,
        allow_404: bool = False,
        allow_403: bool = False,
        retry_without_auth_on_403: bool = False,
    ) -> Optional[Any]:
        """
        GET *url* and return the parsed JSON body.

        Handles 429 (rate-limit), 403 (forbidden), and 404 (not found)
        according to the caller-supplied flags.  Retries up to 6 times with
        exponential back-off on ``RetryError`` and ``RequestException``.

        Returns ``None`` when *allow_404* / *allow_403* applies.
        Raises ``RuntimeError`` after exhausting the retry budget.
        """
        for attempt in range(6):
            try:
                self.limiter.acquire()
                time.sleep(0.05)  # small courtesy delay

                resp = self.session.get(url, params=params, timeout=60)

                if resp.status_code == 404:
                    if allow_404:
                        return None
                    resp.raise_for_status()

                if resp.status_code == 403:
                    if retry_without_auth_on_403:
                        resp2 = self.session2.get(url, params=params, timeout=60)
                        if resp2.status_code == 404 and allow_404:
                            return None
                        if resp2.status_code == 403 and allow_403:
                            return None
                        resp2.raise_for_status()
                        return resp2.json()
                    if allow_403:
                        return None
                    resp.raise_for_status()

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = (
                        float(retry_after)
                        if retry_after
                        else min(120, (2 ** attempt) * 5) + random.uniform(0, 1)
                    )
                    log.warning("w%d | 429 on %s — sleeping %.1fs", self.worker_id, url, sleep_s)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                return resp.json()

            except RetryError as exc:
                sleep_s = min(120, (2 ** attempt) * 5) + random.uniform(0, 1)
                log.warning("w%d | RetryError %s — %.1fs", self.worker_id, exc, sleep_s)
                time.sleep(sleep_s)

            except RequestException as exc:
                sleep_s = min(60, (2 ** attempt) * 2) + random.uniform(0, 1)
                log.warning("w%d | RequestException %s — %.1fs", self.worker_id, exc, sleep_s)
                time.sleep(sleep_s)

        raise RuntimeError(f"w{self.worker_id}: exceeded retry budget for {url}")

    def _head(self, url: str, allow_404: bool = False) -> Optional[requests.Response]:
        """Issue a HEAD request and return the response (or None on 404)."""
        self.limiter.acquire()
        time.sleep(0.05)
        resp = self.session.head(url, allow_redirects=True, timeout=30)
        if resp.status_code == 404 and allow_404:
            return None
        return resp

    def _get_text(self, url: str, allow_404: bool = False) -> Optional[str]:
        """GET *url* and return the response body as plain text."""
        self.limiter.acquire()
        time.sleep(0.05)
        resp = self.session.get(url, timeout=60)
        if resp.status_code == 404 and allow_404:
            return None
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------
    # HF API helpers
    # ------------------------------------------------------------------

    def fetch_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return a flat dict of model metadata, or ``None`` if not found."""
        try:
            info = self.hf_api.model_info(model_id)
            return {
                "id":           getattr(info, "id",           model_id),
                "author":       getattr(info, "author",       None),
                "tags":         getattr(info, "tags",         []) or [],
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "createdAt":    getattr(info, "created_at",   None) or getattr(info, "createdAt",    None),
                "lastModified": getattr(info, "last_modified", None) or getattr(info, "lastModified", None),
                "downloads":    getattr(info, "downloads",    None),
                "likes":        getattr(info, "likes",        None),
                "spaces":       getattr(info, "spaces",       []) or [],
            }
        except HfHubHTTPError as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                return None
            raise

    def fetch_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return a flat dict of dataset metadata, or ``None`` if not found."""
        try:
            info = self.hf_api.dataset_info(dataset_id)
            return {
                "id":           getattr(info, "id",            dataset_id),
                "author":       getattr(info, "author",        None),
                "tags":         getattr(info, "tags",          []) or [],
                "createdAt":    getattr(info, "created_at",    None) or getattr(info, "createdAt",    None),
                "lastModified": getattr(info, "last_modified", None) or getattr(info, "lastModified", None),
                "downloads":    getattr(info, "downloads",     None),
                "likes":        getattr(info, "likes",         None),
            }
        except HfHubHTTPError as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                return None
            raise

    def fetch_space_info(self, space_id: str) -> Optional[Dict[str, Any]]:
        """Return raw space JSON from the HF Spaces API, or ``None``."""
        return self._get_json(
            f"https://huggingface.co/api/spaces/{space_id}",
            allow_404=True,
        )

    def fetch_collection_info(self, slug: str) -> Optional[Dict[str, Any]]:
        """Return raw collection JSON from the HF Collections API, or ``None``."""
        return self._get_json(
            f"https://huggingface.co/api/collections/{slug}",
            allow_404=True,
            allow_403=True,
            retry_without_auth_on_403=True,
        )

    def fetch_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Return raw paper JSON from the HF Papers API, or ``None``."""
        return self._get_json(
            f"https://huggingface.co/api/papers/{paper_id}",
            allow_404=True,
            allow_403=True,
            retry_without_auth_on_403=True,
        )

    def fetch_user_overview(self, username: str) -> Optional[Dict[str, Any]]:
        """Return the user overview JSON, or ``None`` if the user does not exist."""
        return self._get_json(
            f"https://huggingface.co/api/users/{username}/overview",
            allow_404=True,
        )

    def fetch_org_overview(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Return the organization overview JSON, or ``None`` if not found."""
        return self._get_json(
            f"https://huggingface.co/api/organizations/{org_id}/overview",
            allow_404=True,
        )

    def fetch_user_following(self, username: str) -> Optional[Any]:
        """Return the list of accounts this user follows."""
        return self._get_json(
            f"https://huggingface.co/api/users/{username}/following",
            allow_404=True,
        )

    def fetch_user_followers(self, username: str) -> Optional[Any]:
        """Return the list of accounts that follow this user."""
        return self._get_json(
            f"https://huggingface.co/api/users/{username}/followers",
            allow_404=True,
        )

    def fetch_model_readme(self, model_id: str) -> Optional[str]:
        """
        Fetch the raw README.md for *model_id*.

        Returns ``None`` if the file does not exist or is larger than 100 MB.
        """
        url = f"https://huggingface.co/{model_id}/resolve/main/README.md"
        try:
            head = self._head(url, allow_404=True)
            if head is None or head.status_code == 404:
                return None
            content_length = int(head.headers.get("Content-Length", 0))
            if content_length > 100 * 1024 * 1024:
                log.warning("w%d | README too large for %s, skipping", self.worker_id, model_id)
                return None
            return self._get_text(url, allow_404=True)
        except Exception as exc:
            log.warning("w%d | README fetch failed for %s: %s", self.worker_id, model_id, exc)
            return None

    def fetch_dataset_readme(self, dataset_id: str) -> Optional[str]:
        """
        Fetch the raw README.md for *dataset_id*.

        Returns ``None`` if the file does not exist or is larger than 100 MB.
        """
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/README.md"
        try:
            head = self._head(url, allow_404=True)
            if head is None or head.status_code == 404:
                return None
            content_length = int(head.headers.get("Content-Length", 0))
            if content_length > 100 * 1024 * 1024:
                log.warning("w%d | README too large for %s, skipping", self.worker_id, dataset_id)
                return None
            return self._get_text(url, allow_404=True)
        except Exception as exc:
            log.warning("w%d | README fetch failed for %s: %s", self.worker_id, dataset_id, exc)
            return None

    def get_likes(self, item: str, item_type: str):
        """Yield liker objects for a model, dataset, or space."""
        return self.hf_api.list_repo_likers(repo_id=item, repo_type=item_type)

    def list_spaces_by_model(self, model_id: str):
        """Yield Space objects that reference *model_id*."""
        return self.hf_api.list_spaces(models=model_id, full=False)

    def list_spaces_by_dataset(self, dataset_id: str):
        """Yield Space objects that reference *dataset_id*."""
        return self.hf_api.list_spaces(datasets=dataset_id, full=False)

    def list_collections_by_item(self, item_id: str, item_type: str):
        """
        Yield Collection objects that contain *item_id* of *item_type*.

        *item_type* must be one of ``"model"``, ``"dataset"``, ``"paper"``,
        ``"space"``.
        """
        prefix_map = {
            "model":   "models",
            "dataset": "datasets",
            "paper":   "papers",
            "space":   "spaces",
        }
        prefixed = f"{prefix_map[item_type]}/{item_id}"
        return self.hf_api.list_collections(item=prefixed)
