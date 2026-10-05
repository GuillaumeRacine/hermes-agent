"""Provider-agnostic request-kwargs hygiene.

Small, dependency-free helpers applied to outgoing request kwargs right
before they leave the process, so every transport/provider gets the same
guarantees regardless of which code path built the kwargs.
"""

from __future__ import annotations

from typing import Any, Dict

# Request keys that are only meaningful when a non-empty ``tools`` list is
# sent.  Strict providers reject them otherwise -- xAI answers HTTP 400
# "A tool_choice was set on the request but no tools were specified"
# (hermes-home#330), OpenAI rejects ``parallel_tool_calls`` without tools.
TOOL_ONLY_KEYS = ("tool_choice", "parallel_tool_calls")


def drop_orphan_tool_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove tool-only keys when the request carries no tools.

    Mutates and returns ``kwargs``.  An empty/None ``tools`` entry is removed
    as well, since some SDKs iterate it without a None guard.  Requests that
    do carry tools are returned unchanged.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    if kwargs.get("tools"):
        return kwargs
    kwargs.pop("tools", None)
    for key in TOOL_ONLY_KEYS:
        kwargs.pop(key, None)
    return kwargs


__all__ = ["TOOL_ONLY_KEYS", "drop_orphan_tool_fields"]
