"""Compatibility adapter for cron job fire-claim lookups.

This module exists so that tools can be imported and run inside an *old*
Python process that is still executing cached on-disk code from before an
update. Newer tools import ``claim_job_for_fire_result`` directly from
``cron.jobs``; an old cached ``cron.jobs`` does not define that name and would
raise ``ImportError`` at tool-import time.

Importing this module always succeeds, even when the running ``cron.jobs``
lacks the new API. The delegate is resolved lazily, inside the call, so no
state is bound at import time and no delegate is cached between calls.
"""

__all__ = ["claim_job_for_fire_result"]


_RESTART_MESSAGE = (
    "cron.jobs.claim_job_for_fire_result is unavailable in this process; "
    "Hermes must restart before running cron jobs."
)


def claim_job_for_fire_result(job_id):
    """Return the result of the current ``cron.jobs`` claim API for ``job_id``.

    Imports ``cron.jobs`` at call time and looks up
    ``claim_job_for_fire_result`` on the freshly imported module. When that
    attribute is missing or not callable -- e.g. the on-disk code was updated
    but this process still holds an older cached ``cron.jobs`` -- a
    ``RuntimeError`` is raised instructing that Hermes must restart before
    running cron jobs. In that case no legacy mutation, state inspection or
    claim simulation is performed.

    When the callable exists it is called exactly once with ``job_id`` and its
    exact result object is returned; any exception it raises propagates
    unchanged.
    """
    import cron.jobs as _jobs

    delegate = getattr(_jobs, "claim_job_for_fire_result", None)
    if not callable(delegate):
        raise RuntimeError(_RESTART_MESSAGE)
    return delegate(job_id)
