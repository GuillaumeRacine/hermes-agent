"""Anti-thrash guard: real-usage scoring + sliding window (hermes-agent#64).

The guard existed but never fired. Two defects, one test file:

1. Effectiveness was scored on ``estimate_messages_tokens_rough`` -- a pass
   could report "20% saved" on the estimate while the provider's next real
   prompt was *larger* (summary appended, protected tail untouched).
2. The counter required two *consecutive* ineffective passes and reset to zero
   on any effective one. A context with fresh oversized tool output in the
   protected tail alternates effective/ineffective forever and never reaches 2.

Observed before the fix: 13 passes in one session, 182 across the log window,
~15M estimated aux-summarizer tokens.
"""

import pytest

from agent.context_compressor import (
    _MIN_SAVINGS_PCT,
    _OUTCOME_TRIGGER,
    _OUTCOME_WINDOW,
    ContextCompressor,
)


def _cc(**kw) -> ContextCompressor:
    c = ContextCompressor(model="gpt-4", quiet_mode=True, config_context_length=100_000, **kw)
    c.last_prompt_tokens = c.threshold_tokens + 1
    return c


# ── real-usage scoring ────────────────────────────────────────────────


def test_real_usage_overrides_an_optimistic_estimate():
    """A pass the estimate liked, that the provider says grew, is ineffective."""
    c = _cc()
    c.last_real_prompt_tokens = 80_000
    c._pending_effectiveness = 80_000
    # Provider reports a LARGER prompt than before the compaction.
    c.update_from_response({"prompt_tokens": 90_000})
    assert c._outcomes[-1] is True
    assert c._last_compression_savings_pct < 0


def test_real_usage_confirms_a_genuinely_effective_pass():
    c = _cc()
    c._pending_effectiveness = 80_000
    c.update_from_response({"prompt_tokens": 40_000})
    assert c._outcomes[-1] is False
    assert c._last_compression_savings_pct == pytest.approx(50.0)


def test_pending_check_is_settled_exactly_once():
    c = _cc()
    c._pending_effectiveness = 80_000
    c.update_from_response({"prompt_tokens": 40_000})
    assert c._pending_effectiveness is None
    n = len(c._outcomes)
    c.update_from_response({"prompt_tokens": 40_000})   # no pass in between
    assert len(c._outcomes) == n


def test_zero_prompt_tokens_leaves_the_check_pending():
    """A response with no usage must not be scored as a 100% saving."""
    c = _cc()
    c._pending_effectiveness = 80_000
    c.update_from_response({"prompt_tokens": 0})
    assert c._pending_effectiveness == 80_000
    assert not c._outcomes


# ── sliding window ────────────────────────────────────────────────────


def test_alternating_outcomes_still_trigger_backoff():
    """The exact pattern the old consecutive counter could never catch."""
    c = _cc()
    for pct in (0, 50, 0):          # ineffective, effective, ineffective
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": int(100_000 * (1 - pct / 100))})
    assert sum(c._outcomes) >= _OUTCOME_TRIGGER
    assert c.should_compress(c.threshold_tokens + 1) is False


def test_consecutive_counter_alone_would_not_have_fired():
    """Documents the old behaviour on the same sequence."""
    c = _cc()
    for pct in (0, 50, 0):
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": int(100_000 * (1 - pct / 100))})
    assert c._ineffective_compression_count < 2      # old guard: still silent
    assert c.should_compress(c.threshold_tokens + 1) is False   # new guard: stops


def test_effective_passes_never_block_compression():
    c = _cc()
    for _ in range(_OUTCOME_WINDOW * 2):
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": 40_000})
    assert c.should_compress(c.threshold_tokens + 1) is True


def test_window_forgets_old_failures():
    """Back-off must lift once recent passes are effective again."""
    c = _cc()
    for _ in range(_OUTCOME_TRIGGER):
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": 100_000})
    assert c.should_compress(c.threshold_tokens + 1) is False
    for _ in range(_OUTCOME_WINDOW):
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": 40_000})
    assert c.should_compress(c.threshold_tokens + 1) is True


def test_below_threshold_never_compresses_regardless_of_window():
    c = _cc()
    assert c.should_compress(1) is False


# ── fallback + lifecycle ──────────────────────────────────────────────


def test_estimate_settles_the_check_when_real_usage_never_arrives():
    """Unsettled outcomes would silently shrink the window and resume thrash."""
    c = _cc()
    c._pending_effectiveness = 100_000
    c._settle_pending_on_estimate(99_000)          # ~1% saved
    assert c._outcomes[-1] is True
    assert c._pending_effectiveness is None


def test_estimate_fallback_respects_the_savings_floor():
    c = _cc()
    c._pending_effectiveness = 100_000
    c._settle_pending_on_estimate(int(100_000 * (1 - (_MIN_SAVINGS_PCT + 5) / 100)))
    assert c._outcomes[-1] is False


def test_session_reset_clears_the_window():
    c = _cc()
    for _ in range(_OUTCOME_TRIGGER):
        c._pending_effectiveness = 100_000
        c.update_from_response({"prompt_tokens": 100_000})
    assert c.should_compress(c.threshold_tokens + 1) is False
    c.on_session_reset()
    assert not c._outcomes
    assert c._pending_effectiveness is None
    c.last_prompt_tokens = c.threshold_tokens + 1
    assert c.should_compress(c.threshold_tokens + 1) is True


def test_state_is_lazy_for_new_constructed_instances():
    """Parts of the suite build compressors via __new__ and set only what they use."""
    c = ContextCompressor.__new__(ContextCompressor)
    c.quiet_mode = True
    c._record_outcome(True, "test")
    assert len(c._outcomes) == 1
