"""Eligibility decision for firing a cron job.

This helper is intentionally pure: it performs no IO and never claims or runs a
job. The caller holds the jobs lock, supplies ``now``, and performs claim
persistence.
"""

from datetime import datetime

__all__ = ["fire_claim_blocker"]


def fire_claim_blocker(job, *, now, claim_ttl_seconds=300, ensure_aware):
    """Return a skip reason string, or ``None`` when the job is eligible.

    Checks, in order:

    * ``None`` job -> ``"missing"``
    * ``state == "paused"`` -> ``"paused"``
    * falsy ``enabled`` -> ``"disabled"``
    * fresh existing ``fire_claim`` -> ``"already_claimed"``

    A malformed claim (parse, normalization, or comparison failure) is ignored
    and treated as eligible. Inputs are never mutated.
    """
    if job is None:
        return "missing"
    if job.get("state") == "paused":
        return "paused"
    if not job.get("enabled", True):
        return "disabled"

    claim = job.get("fire_claim")
    if claim:
        try:
            claimed_at = ensure_aware(datetime.fromisoformat(claim["at"]))
            if (now - claimed_at).total_seconds() < claim_ttl_seconds:
                return "already_claimed"
        except Exception:
            return None

    return None
