"""Thin wrapper around the :mod:`wandb` public API.

Kept separate from the orchestrator so tests can fake the API without
touching network code, and so the (optional) ``wandb`` import lives in
exactly one place.

Also exposes :func:`no_proxy_env` for the case where the enterprise HTTP
proxy would otherwise hijack calls to the W&B server.  Most users will not
need it (the proxy correctly forwards to ``api.wandb.ai``), but it is
useful when working against a self-hosted W&B on the local network.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional


_logger = logging.getLogger(__name__)


@contextmanager
def no_proxy_env() -> Iterator[None]:
    """Temporarily clear HTTP(S) proxy env vars for local-network W&B servers.

    Restores the original environment on exit.  Safe to nest.
    """
    keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


class WandbClient:
    """Lazy, test-friendly handle on :class:`wandb.Api`."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._api: Any = None

    # ------------------------------------------------------------------
    def _ensure_api(self) -> Any:
        if self._api is not None:
            return self._api
        try:
            import wandb  # noqa: F401  -- ensures the package is importable
            from wandb import Api
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The `wandb` package is required for evaluation.wandb_sync. "
                "Install it with `pip install wandb`."
            ) from exc

        overrides: dict[str, Any] = {}
        if self._base_url:
            overrides["base_url"] = self._base_url
        if self._api_key:
            overrides["api_key"] = self._api_key
        self._api = Api(overrides=overrides, timeout=self._timeout)
        _logger.debug("Initialized wandb.Api(overrides=%s)", list(overrides))
        return self._api

    # ------------------------------------------------------------------
    def runs(
        self,
        path: str,
        filters: dict,
        per_page: int = 200,
        lazy: bool = False,
    ) -> Any:
        """Return an iterable of ``wandb.apis.public.Run`` objects.

        ``path`` is ``"<entity>/<project>"``.

        ``lazy`` defaults to ``False`` here, which is the opposite of the
        W&B SDK default.  The SDK's lazy mode asks the server for a
        ``LightweightRunFragment`` that omits ``config``, ``summaryMetrics``
        and ``systemMetrics``; accessing those attributes then triggers one
        GraphQL round-trip **per run**, and those follow-ups are the usual
        culprit behind empty ``config.yaml`` files (they can come back with
        raw / unflattened / empty payloads, especially across corporate
        proxies).  With ``lazy=False`` every page of the paginated listing
        uses the full ``RunFragment`` and populates ``run.config`` /
        ``run.rawconfig`` in-line, so downloads are both faster and
        deterministic.  Pass ``lazy=True`` only when iterating a very large
        project purely to enumerate names/ids.
        """
        api = self._ensure_api()
        return api.runs(
            path=path,
            filters=filters or None,
            per_page=per_page,
            lazy=lazy,
        )
