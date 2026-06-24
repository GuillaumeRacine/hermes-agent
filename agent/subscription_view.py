"""Surface-agnostic core for the ``/subscription`` TUI screen.

Companion to :mod:`agent.billing_view` — same fail-open philosophy: when not
logged in or the portal is unreachable, return a struct with ``logged_in=False``
and let the surface degrade gracefully (never crash). Money is decimal end-to-end
(server emits decimal strings); we only format for display.

The TUI ``SubscriptionOverlay`` is **deep-link only** — it never charges
in-terminal. The manage URL is built locally on the TUI side from the
``portal_url`` and ``org_id`` fields in the subscription state.

WS1 dependency: ``GET /api/billing/subscription`` is a NAS endpoint (WS1 Phase A).
Until it ships, the fail-open contract handles 404s — the builder returns
``logged_in=False`` and the surface degrades gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from agent.billing_view import parse_money

logger = logging.getLogger(__name__)


# =============================================================================
# Parsed sub-structures
# =============================================================================


@dataclass(frozen=True)
class SubscriptionTier:
    """A plan tier in the catalog."""

    tier_id: str
    name: str
    tier_order: int
    dollars_per_month: Optional[Decimal] = None
    monthly_credits: Optional[Decimal] = None
    is_current: bool = False
    is_enabled: bool = True


@dataclass(frozen=True)
class CurrentSubscription:
    """The user's active subscription (None fields = free / no active sub)."""

    tier_id: Optional[str] = None
    tier_name: Optional[str] = None
    monthly_credits: Optional[Decimal] = None
    credits_remaining: Optional[Decimal] = None
    cycle_ends_at: Optional[str] = None  # ISO
    pending_downgrade_tier_name: Optional[str] = None
    pending_downgrade_at: Optional[str] = None  # ISO
    is_past_due: bool = False
    cancel_at_period_end: bool = False
    cancellation_effective_at: Optional[str] = None  # ISO


@dataclass(frozen=True)
class SubscriptionState:
    """Parsed ``GET /api/billing/subscription`` — the overview screen's data.

    Fail-open: ``logged_in=False`` (and empty fields) when not logged in or the
    portal is unreachable.
    """

    logged_in: bool
    org_name: Optional[str] = None
    org_id: Optional[str] = None  # org.id from the NAS response
    role: Optional[str] = None  # "OWNER" | "ADMIN" | "MEMBER"
    context: str = "personal"  # "personal" | "team"
    current: Optional[CurrentSubscription] = None
    tiers: tuple[SubscriptionTier, ...] = ()
    portal_url: Optional[str] = None
    # When the fetch failed (vs cleanly not-logged-in), the message for the surface.
    error: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        """True for OWNER/ADMIN — the roles that can change plans."""
        return (self.role or "").upper() in ("OWNER", "ADMIN")

    @property
    def can_change_plan(self) -> bool:
        """True when the UI should offer plan-change actions (role gate from NAS)."""
        return self.is_admin


# =============================================================================
# Payload parsing
# =============================================================================


def _parse_tier(raw: Any) -> Optional[SubscriptionTier]:
    if not isinstance(raw, dict):
        return None
    tier_id = raw.get("tierId") or raw.get("id")
    name = raw.get("name")
    if not (isinstance(tier_id, str) and isinstance(name, str)):
        return None
    return SubscriptionTier(
        tier_id=tier_id,
        name=name,
        tier_order=int(raw.get("tierOrder") or raw.get("order") or 0),
        dollars_per_month=parse_money(raw.get("dollarsPerMonth") or raw.get("priceUsd")),
        monthly_credits=parse_money(raw.get("monthlyCredits")),
        is_current=bool(raw.get("isCurrent")),
        is_enabled=bool(raw.get("isEnabled", True)),
    )


def _parse_current(raw: Any) -> Optional[CurrentSubscription]:
    if not isinstance(raw, dict):
        return None
    return CurrentSubscription(
        tier_id=raw.get("tierId") or raw.get("id"),
        tier_name=raw.get("tierName") or raw.get("name"),
        monthly_credits=parse_money(raw.get("monthlyCredits")),
        credits_remaining=parse_money(raw.get("creditsRemaining")),
        cycle_ends_at=raw.get("cycleEndsAt"),
        pending_downgrade_tier_name=raw.get("pendingDowngradeTierName"),
        pending_downgrade_at=raw.get("pendingDowngradeAt"),
        is_past_due=bool(raw.get("isPastDue")),
        cancel_at_period_end=bool(raw.get("cancelAtPeriodEnd")),
        cancellation_effective_at=raw.get("cancellationEffectiveAt") or None,
    )


def subscription_state_from_payload(
    payload: dict[str, Any], *, portal_url: Optional[str] = None
) -> SubscriptionState:
    """Map a raw ``/api/billing/subscription`` JSON dict into :class:`SubscriptionState`."""
    raw_org = payload.get("org")
    org: dict[str, Any] = raw_org if isinstance(raw_org, dict) else {}

    tiers: list[SubscriptionTier] = []
    for item in payload.get("tiers") or ():
        parsed = _parse_tier(item)
        if parsed is not None:
            tiers.append(parsed)

    raw_context = payload.get("context")
    context = raw_context if raw_context in ("personal", "team") else "personal"

    return SubscriptionState(
        logged_in=True,
        org_name=org.get("name"),
        org_id=org.get("id") or None,
        role=org.get("role"),
        context=context,
        current=_parse_current(payload.get("current")),
        tiers=tuple(tiers),
        portal_url=portal_url,
    )


# =============================================================================
# Fail-open builders (the surface front doors)
# =============================================================================


def build_subscription_state(*, timeout: float = 15.0) -> SubscriptionState:
    """Fetch + parse ``GET /api/billing/subscription``. Fail-open.

    Returns ``SubscriptionState(logged_in=False)`` when not logged in. On a
    portal/HTTP failure, returns ``logged_in=False`` with ``error`` set so the
    surface can show a clear message rather than crashing.
    """
    try:
        from hermes_cli.nous_billing import (
            BillingAuthError,
            BillingError,
            _absolutize_portal_url,
            get_subscription_state,
            resolve_portal_base_url,
        )
    except Exception:
        return SubscriptionState(logged_in=False, error="billing client unavailable")

    try:
        payload = get_subscription_state(timeout=timeout)
    except BillingAuthError:
        return SubscriptionState(logged_in=False)
    except BillingError as exc:
        logger.debug("subscription ▸ /state fetch failed (fail-open)", exc_info=True)
        return SubscriptionState(logged_in=False, error=str(exc))
    except Exception:
        logger.debug("subscription ▸ /state unexpected error (fail-open)", exc_info=True)
        return SubscriptionState(logged_in=False, error="could not load subscription state")

    raw_portal = payload.get("portalUrl") if isinstance(payload, dict) else None
    portal_url = _absolutize_portal_url(raw_portal) if raw_portal else None
    if not portal_url:
        try:
            portal_url = resolve_portal_base_url()
        except Exception:
            portal_url = None

    return subscription_state_from_payload(payload, portal_url=portal_url)


