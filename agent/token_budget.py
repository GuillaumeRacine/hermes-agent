"""Per-session / per-turn token budget — runtime enforcement.

Companion to :mod:`agent.iteration_budget`.  ``IterationBudget`` caps how
many API *calls* a turn may make; :class:`TokenBudget` caps how many
*tokens* (prompt + completion, as billed by the provider) a turn and a
session may spend, and can force a context-compression pass when a single
request's prompt grows past a soft limit.

Configured via ``agent.token_budget`` in config.yaml (see
``hermes_cli.config.resolve_token_budget`` for the per-platform merge).
Each :class:`AIAgent` holds one instance as ``agent._token_budget``.

Enforcement seams (all in ``agent/conversation_loop.py``):

* ``record(usage)``   — called from the usage-accounting block after every
  API response (the same place ``agent.session_total_tokens`` grows).
* ``reset_turn()``    — called from ``build_turn_context`` next to the
  per-turn ``IterationBudget`` reset.
* ``breach()``        — consulted after tool execution (mirrors the
  ``max_iterations`` / guardrail-halt exits) and at the top of the loop so
  a session already over budget refuses the next turn before any API call.
* ``grant_extension()`` — the one-more-turn override triggered by a bare
  ``continue`` message while the session is in the exceeded state.

The object is deliberately dumb: it holds limits + counters and answers
questions.  All logging / user-facing output stays in the loop.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"

# Leading text of every stop-mode pause notice.  The conversation loop also
# uses it to recognise "the previous assistant turn was a budget pause" from
# the transcript alone, so a ``continue`` still resumes after the gateway
# evicted and rebuilt the cached agent (which drops in-memory flags).
PAUSE_MARKER = "Paused (token budget):"

# Model-facing text substituted for a bare ``continue`` that follows a
# budget pause.  The transcript keeps the user's literal ``continue``.
RESUME_INSTRUCTION = (
    "continue\n\n[System note: your previous turn was paused by the token "
    "budget, not finished. Resume the same task from where you stopped. Do "
    "not repeat steps whose results are already in the conversation; if the "
    "remaining work is small, finish it, otherwise make concrete progress and "
    "report what is done and what is left.]"
)


def _usage_int(usage: Any, *names: str) -> int:
    """Read the first present integer-ish field from a usage dict/object."""
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


class TokenBudget:
    """Thread-safe token counters with per-turn / per-session limits.

    ``per_session``, ``per_turn`` and ``context_soft_limit`` are token
    counts; ``0`` disables the corresponding check.  ``action`` is
    ``"stop"`` (end the turn once a limit is crossed) or ``"warn"``
    (log once per breach, never stop).
    """

    def __init__(
        self,
        per_session: int = 0,
        per_turn: int = 0,
        context_soft_limit: int = 0,
        action: str = "stop",
        platform: Optional[str] = None,
    ):
        self.per_session = max(0, int(per_session or 0))
        self.per_turn = max(0, int(per_turn or 0))
        self.context_soft_limit = max(0, int(context_soft_limit or 0))
        self.action = "warn" if str(action or "stop").lower() == "warn" else "stop"
        self.platform = platform

        self.session_tokens = 0
        self.turn_tokens = 0
        self.last_prompt_tokens = 0
        self.api_calls = 0
        self.extensions_granted = 0

        # ``exceeded`` is the sticky session-level state the ``continue``
        # override consults: it is set when the *session* limit is crossed
        # in stop mode and cleared by ``grant_extension`` / ``reset_session``.
        self.exceeded = False
        self.last_breach: Optional[str] = None

        # Once-per-breach bookkeeping for warn mode.
        self._warned_turn = False
        self._warned_session = False
        # Soft-limit bookkeeping.
        self._compression_requested = False
        self._soft_limit_warned_turn = False

        self._lock = threading.Lock()

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]], platform: Optional[str] = None) -> "TokenBudget":
        """Build from a full config dict (``resolve_token_budget`` merge)."""
        from hermes_cli.config import resolve_token_budget

        resolved = resolve_token_budget(config, platform)
        return cls(
            per_session=resolved["per_session"],
            per_turn=resolved["per_turn"],
            context_soft_limit=resolved["context_soft_limit"],
            action=resolved["action"],
            platform=resolved.get("platform") or platform,
        )

    @classmethod
    def from_resolved(cls, resolved: Dict[str, Any]) -> "TokenBudget":
        return cls(
            per_session=resolved.get("per_session", 0),
            per_turn=resolved.get("per_turn", 0),
            context_soft_limit=resolved.get("context_soft_limit", 0),
            action=resolved.get("action", "stop"),
            platform=resolved.get("platform"),
        )

    # ── properties ─────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when any limit is configured."""
        return bool(self.per_session or self.per_turn or self.context_soft_limit)

    @property
    def stops(self) -> bool:
        return self.action == "stop"

    # ── counters ────────────────────────────────────────────────────

    def reset_turn(self) -> None:
        """Start a new user turn (session counters are kept)."""
        with self._lock:
            self.turn_tokens = 0
            self._warned_turn = False
            self._soft_limit_warned_turn = False
            # A pending compression request survives into the next turn's
            # preflight; it is consumed by whichever compression seam runs
            # first.

    def reset_session(self) -> None:
        """New session (``/new`` etc.) — everything back to zero."""
        with self._lock:
            self.session_tokens = 0
            self.turn_tokens = 0
            self.last_prompt_tokens = 0
            self.api_calls = 0
            self.exceeded = False
            self.last_breach = None
            self._warned_turn = False
            self._warned_session = False
            self._compression_requested = False
            self._soft_limit_warned_turn = False

    def record(self, usage: Any) -> int:
        """Account one API response's usage.  Returns tokens added.

        Accepts the loop's ``usage_dict`` (``prompt_tokens`` /
        ``completion_tokens`` / ``total_tokens``) or any object exposing
        the same or the Anthropic-style ``input_tokens`` / ``output_tokens``
        fields.  When ``total_tokens`` is missing it is derived.
        """
        if usage is None:
            return 0
        prompt = _usage_int(usage, "prompt_tokens", "input_tokens")
        completion = _usage_int(usage, "completion_tokens", "output_tokens")
        total = _usage_int(usage, "total_tokens") or (prompt + completion)
        with self._lock:
            self.session_tokens += total
            self.turn_tokens += total
            self.last_prompt_tokens = prompt
            self.api_calls += 1
            if self.context_soft_limit and prompt > self.context_soft_limit:
                self._compression_requested = True
        return total

    # ── checks ──────────────────────────────────────────────────────

    def turn_exceeded(self) -> bool:
        return bool(self.per_turn) and self.turn_tokens > self.per_turn

    def session_exceeded(self) -> bool:
        return bool(self.per_session) and self.session_tokens > self.per_session

    def breach(self) -> Optional[str]:
        """Return ``"per_session"`` / ``"per_turn"`` / ``None``.

        Session breaches win because they are the sticky state the
        ``continue`` override reasons about.  In stop mode a session
        breach also flips :attr:`exceeded`.
        """
        with self._lock:
            reason: Optional[str] = None
            if self.per_session and self.session_tokens > self.per_session:
                reason = "per_session"
                if self.action == "stop":
                    self.exceeded = True
            elif self.per_turn and self.turn_tokens > self.per_turn:
                reason = "per_turn"
            if reason:
                self.last_breach = reason
            return reason

    def should_warn(self, reason: Optional[str]) -> bool:
        """For warn mode: True the first time ``reason`` is seen for its scope."""
        if not reason:
            return False
        with self._lock:
            if reason == "per_session":
                if self._warned_session:
                    return False
                self._warned_session = True
                return True
            if reason == "per_turn":
                if self._warned_turn:
                    return False
                self._warned_turn = True
                return True
        return False

    # ── soft limit ──────────────────────────────────────────────────

    def consume_compression_request(self) -> bool:
        """True (once) when the last prompt crossed ``context_soft_limit``."""
        with self._lock:
            requested = self._compression_requested
            self._compression_requested = False
            return requested

    def compression_requested(self) -> bool:
        return self._compression_requested

    def should_warn_soft_limit_unavailable(self) -> bool:
        """Once per turn: soft limit crossed but compression could not run."""
        with self._lock:
            if self._soft_limit_warned_turn:
                return False
            self._soft_limit_warned_turn = True
            return True

    # ── one-more-turn override ──────────────────────────────────────

    def extension_amount(self) -> int:
        """Tokens added to ``per_session`` by one ``continue`` grant."""
        if self.per_turn:
            return self.per_turn
        if self.per_session:
            return max(1, int(self.per_session * 0.2))
        return 0

    def grant_extension(self) -> int:
        """Raise the session limit so one more turn may run.

        Returns the amount added.  The new limit is at least
        ``session_tokens + amount`` so the extension is meaningful even if
        the session overshot the previous limit by a wide margin.
        """
        with self._lock:
            amount = self.extension_amount()
            if amount and self.per_session:
                self.per_session = max(self.per_session, self.session_tokens) + amount
            self.exceeded = False
            self.last_breach = None
            self._warned_session = False
            self.extensions_granted += 1
            return amount

    # ── reporting ───────────────────────────────────────────────────

    @staticmethod
    def _fmt_limit(limit: int) -> str:
        return f"{limit:,}" if limit else "unlimited"

    def stop_message(
        self,
        reason: Optional[str] = None,
        *,
        progress: Optional[str] = None,
    ) -> str:
        """User-facing pause notice for a stop-mode breach.

        ``reason`` is the value :meth:`breach` returned (``"per_turn"`` /
        ``"per_session"``); when omitted it is inferred from the counters,
        session first.  The notice names the limit that actually fired --
        the pre-#330 text always said "this session has used X of its Y
        budget" even when the *turn* cap fired, which read as if a 4M
        session budget had been blown at 1.5M (hermes-home#330).

        ``progress`` is an optional one-line "where I stopped" description
        (see :func:`describe_progress`) so the user knows what a
        ``continue`` will resume.
        """
        if reason is None:
            reason = self.last_breach
        if reason is None:
            if self.session_exceeded():
                reason = "per_session"
            elif self.turn_exceeded():
                reason = "per_turn"
        if reason == "per_turn":
            headline = (
                f"{PAUSE_MARKER} this task used {self.turn_tokens:,} tokens in a "
                f"single turn, over the per-turn limit of "
                f"{self._fmt_limit(self.per_turn)} (session total "
                f"{self.session_tokens:,} of {self._fmt_limit(self.per_session)})."
            )
            tail = (
                "Reply `continue` to resume where I left off "
                "(the per-turn counter resets on each new turn)."
            )
        else:
            headline = (
                f"{PAUSE_MARKER} this session has used {self.session_tokens:,} "
                f"tokens of its {self._fmt_limit(self.per_session)} session budget "
                f"(this turn: {self.turn_tokens:,}/{self._fmt_limit(self.per_turn)})."
            )
            amount = self.extension_amount()
            extra = f" (+{amount:,} tokens)" if amount else ""
            tail = (
                f"Reply `continue` to extend the budget{extra} and resume "
                "where I left off, or start a new session."
            )
        parts = [headline]
        if progress:
            prog = progress.strip()
            if prog and prog[-1] not in ".!?":
                prog += "."
            parts.append(f"Progress so far: {prog}")
        parts.append(tail)
        return " ".join(parts)

    def warn_message(self, reason: Optional[str] = None) -> str:
        scope = "turn" if reason == "per_turn" else "session"
        return (
            f"Token budget warning ({scope}): session {self.session_tokens:,}/"
            f"{self._fmt_limit(self.per_session)}, turn {self.turn_tokens:,}/"
            f"{self._fmt_limit(self.per_turn)} tokens."
        )

    def summary(self) -> str:
        """``<turn>/<session>[/<limit>]`` for the ``Turn ended:`` log line."""
        base = f"{self.turn_tokens}/{self.session_tokens}"
        if self.per_session:
            return f"{base}/{self.per_session}"
        return base

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_session": self.per_session,
            "per_turn": self.per_turn,
            "context_soft_limit": self.context_soft_limit,
            "action": self.action,
            "platform": self.platform,
            "session_tokens": self.session_tokens,
            "turn_tokens": self.turn_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "exceeded": self.exceeded,
            "extensions_granted": self.extensions_granted,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TokenBudget({self.as_dict()!r})"


def describe_progress(messages: Any, api_call_count: int = 0, max_tools: int = 4) -> str:
    """One-line "where I stopped" summary built from the turn's transcript.

    Pure / local: no API call, so it is safe to use exactly when the budget
    says no more tokens may be spent.  Reports the step count and the most
    recent distinct tool names, e.g. ``"stopped at step 17; recent tools:
    web_search, read_file"``.
    """
    recent: list = []
    try:
        for msg in reversed(list(messages or [])):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for tc in reversed(msg.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                if name and name not in recent:
                    recent.append(name)
                if len(recent) >= max_tools:
                    break
            if len(recent) >= max_tools:
                break
    except Exception:
        recent = []
    parts = []
    if api_call_count:
        parts.append(f"stopped at step {api_call_count}")
    if recent:
        parts.append("recent tools: " + ", ".join(reversed(recent)))
    return "; ".join(parts)


def is_budget_pause_message(content: Any) -> bool:
    """True when ``content`` is a stop-mode pause notice (see PAUSE_MARKER)."""
    return isinstance(content, str) and content.lstrip().startswith(PAUSE_MARKER)


def is_continue_message(user_message: Any) -> bool:
    """True when the user message is exactly ``continue`` (trimmed, any case)."""
    if not isinstance(user_message, str):
        return False
    return user_message.strip().lower() == "continue"


__all__ = [
    "TokenBudget",
    "TOKEN_BUDGET_EXCEEDED",
    "PAUSE_MARKER",
    "RESUME_INSTRUCTION",
    "describe_progress",
    "is_budget_pause_message",
    "is_continue_message",
]
