"""Server-sent events wire format (spec 14: APP-12), used by ``GET /api/evac/{id}/run`` (APP-05)."""

from __future__ import annotations

import json
import re
from typing import Any

_EVENT_RE = re.compile(r"^[a-z][a-z_]*$")


def format_sse(event: str, data: Any, id: int | None = None) -> str:
    """Return one SSE message: ``"event: <event>\\n"``, then ``"id: <id>\\n"`` if ``id`` is not None, then
    ``"data: <json>\\n\\n"`` with ``json.dumps(data, separators=(",", ":"), sort_keys=True)``.

    Raises ``ValueError`` if ``event`` does not match ``^[a-z][a-z_]*$`` or ``id`` is not a non-negative int.
    (APP-12)
    """
    if not isinstance(event, str) or not _EVENT_RE.match(event):
        raise ValueError(f"bad SSE event name: {event!r}")
    if id is not None and (not isinstance(id, int) or isinstance(id, bool) or id < 0):
        raise ValueError(f"SSE id must be a non-negative int, got {id!r}")
    lines = [f"event: {event}\n"]
    if id is not None:
        lines.append(f"id: {id}\n")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'), sort_keys=True)}\n\n")
    return "".join(lines)
