"""Server-sent events wire format (spec 14: APP-12), used by ``GET /api/evac/{id}/run`` (APP-05)."""

from __future__ import annotations

from typing import Any


def format_sse(event: str, data: Any, id: int | None = None) -> str:
    """Return one SSE message: ``"event: <event>\\n"``, then ``"id: <id>\\n"`` if ``id`` is not None, then
    ``"data: <json>\\n\\n"`` with ``json.dumps(data, separators=(",", ":"), sort_keys=True)``.

    Raises ``ValueError`` if ``event`` does not match ``^[a-z][a-z_]*$`` or ``id`` is not a non-negative int.
    (APP-12)
    """
    raise NotImplementedError
