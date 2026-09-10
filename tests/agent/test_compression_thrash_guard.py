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
    _BACKOFF_OVERRIDE_RATIO,
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
    c._pending_effectiveness = (80_000, 50.0)
    # Provider reports a LARGER prompt than before the compaction.
    c.update_from_response({"prompt_tokens": 90_000})
    assert c._outcomes[-1] is True
    assert c._last_compression_savings_pct < 0


def test_real_usage_confirms_a_genuinely_effective_pass():
    c = _cc()
    c._pending_effectiveness = (80_000, 50.0)
    c.update_from_response({"prompt_tokens": 40_000})
    assert c._outcomes[-1] is False
    assert c._last_compression_savings_pct == pytest.approx(50.0)


def test_pending_check_is_settled_exactly_once():
    c = _cc()
    c._pending_effectiveness = (80_000, 50.0)
    c.update_from_response({"prompt_tokens": 40_000})
    assert c._pending_effectiveness is None
    n = len(c._outcomes)
    c.update_from_response({"prompt_tokens": 40_000})   # no pass in between
    assert len(c._outcomes) == n


def test_zero_prompt_tokens_leaves_the_check_pending():
    """A response with no usage must not be scored as a 100% saving."""
    c = _cc()
    c._pending_effectiveness = (80_000, 50.0)
    c.update_from_response({"prompt_tokens": 0})
    assert c._pending_effectiveness == (80_000, 50.0)
    assert not c._outcomes


# ── sliding window ────────────────────────────────────────────────────


def test_alternating_outcomes_still_trigger_backoff():
    """The exact pattern the old consecutive counter could never catch."""
    c = _cc()
    for pct in (0, 50, 0):          # ineffective, effective, ineffective
        c._pending_effectiveness = (100_000, 50.0)
        c.update_from_response({"prompt_tokens": int(100_000 * (1 - pct / 100))})
    assert sum(c._outcomes) >= _OUTCOME_TRIGGER
    assert c.should_compress(c.threshold_tokens + 1) is False


def test_consecutive_counter_alone_would_not_have_fired():
    """Documents the old behaviour on the same sequence."""
    c = _cc()
    for pct in (0, 50, 0):
        c._pending_effectiveness = (100_000, 50.0)
        c.update_from_response({"prompt_tokens": int(100_000 * (1 - pct / 100))})
    assert c._ineffective_compression_count < 2      # old guard: still silent
    assert c.should_compress(c.threshold_tokens + 1) is False   # new guard: stops


def test_effective_passes_never_block_compression():
    c = _cc()
    for _ in range(_OUTCOME_WINDOW * 2):
        c._pending_effectiveness = (100_000, 50.0)
        c.update_from_response({"prompt_tokens": 40_000})
    assert c.should_compress(c.threshold_tokens + 1) is True


def test_window_forgets_old_failures():
    """Back-off must lift once recent passes are effective again."""
    c = _cc()
    for _ in range(_OUTCOME_TRIGGER):
        c._pending_effectiveness = (100_000, 50.0)
        c.update_from_response({"prompt_tokens": 100_000})
    assert c.should_compress(c.threshold_tokens + 1) is False
    for _ in range(_OUTCOME_WINDOW):
        c._pending_effectiveness = (100_000, 50.0)
        c.update_from_response({"prompt_tokens": 40_000})
    assert c.should_compress(c.threshold_tokens + 1) is True


def test_below_threshold_never_compresses_regardless_of_window():
    c = _cc()
    assert c.should_compress(1) is False


# ── fallback + lifecycle ──────────────────────────────────────────────


def test_estimate_settles_the_check_when_real_usage_never_arrives():
    """Unsettled outcomes would silently shrink the window and resume thrash."""
    c = _cc()
    c._pending_effectiveness = (100_000, 1.0)      # the pass itself saved ~1%
    c._settle_pending_on_estimate()
    assert c._outcomes[-1] is True
    assert c._pending_effectiveness is None


def test_estimate_fallback_respects_the_savings_floor():
    c = _cc()
    c._pending_effectiveness = (100_000, _MIN_SAVINGS_PCT + 5)
    c._settle_pending_on_estimate()
    assert c._outcomes[-1] is False


def test_session_reset_clears_the_window():
    c = _cc()
    for _ in range(_OUTCOME_TRIGGER):
        c._pending_effectiveness = (100_000, 50.0)
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


def test_legacy_counter_still_triggers_backoff():
    """`hermes_cli/context_switch_guard.py` reads this counter directly.

    Honouring only the new window would silently disable back-off for any
    caller that sets the legacy counter without going through
    ``_record_outcome()``.
    """
    c = _cc()
    c._ineffective_compression_count = _OUTCOME_TRIGGER
    assert c.should_compress(c.threshold_tokens + 1) is False


def test_legacy_counter_below_trigger_still_allows():
    c = _cc()
    c._ineffective_compression_count = _OUTCOME_TRIGGER - 1
    assert c.should_compress(c.threshold_tokens + 1) is True


# ── regressions found by adversarial review (2026-09-10) ──────────────


def test_estimate_fallback_uses_the_passs_own_savings_not_a_later_context():
    """C1: the fallback must not recompute from the *next* pass's size.

    ``_settle_pending_on_estimate`` runs at the top of the following
    ``compress()``, where the only size available is that pass's PRE-compaction
    total. Both figures sit just above the threshold by construction, so a
    recomputed ratio is ~0% for every pass however well it worked -- which
    latched the guard off permanently on providers that omit ``usage``.
    """
    c = _cc()
    c._pending_effectiveness = (95_000, 75.0)   # the pass really saved 75%
    c._settle_pending_on_estimate()
    assert c._outcomes[-1] is False


def test_no_usage_provider_does_not_latch_the_guard_off():
    """C1 end-to-end: five effective passes, no update_from_response()."""
    c = _cc()
    for _ in range(5):
        c._pending_effectiveness = (95_000, 72.0)
        c._settle_pending_on_estimate()
        assert c.should_compress(c.threshold_tokens + 1) is True


def test_backoff_yields_to_context_pressure():
    """C2: an overflow is certain; an ineffective compaction is merely likely."""
    c = _cc()
    for _ in range(_OUTCOME_TRIGGER):
        c._pending_effectiveness = (100_000, 0.0)
        c._settle_pending_on_estimate()
    assert c.should_compress(c.threshold_tokens + 1) is False          # backing off
    near_limit = int(c.context_length * _BACKOFF_OVERRIDE_RATIO) + 1
    assert c.should_compress(near_limit) is True                       # overridden


def test_post_compaction_growth_is_not_charged_against_the_pass():
    """C2: a big tool result landing after a compaction must not score it bad."""
    c = _cc()
    c.last_compression_rough_tokens = 30_000       # what the pass left behind
    c._pending_effectiveness = (100_000, 70.0)
    c.update_from_response({"prompt_tokens": 95_000})   # regrown by a tool result
    assert c._outcomes[-1] is False


def test_backoff_property_agrees_with_should_compress():
    """C3: context_switch_guard reads the property; it must not disagree."""
    c = _cc()
    for pct in (0.0, 50.0, 0.0):                   # the alternating pattern
        c._pending_effectiveness = (100_000, pct)
        c._settle_pending_on_estimate()
    assert c._ineffective_compression_count < _OUTCOME_TRIGGER   # legacy says "fine"
    assert c.compression_backoff_active is True
    assert c.should_compress(c.threshold_tokens + 1) is False


def test_negative_display_tokens_do_not_poison_the_window():
    """C4: -1 is a live sentinel forwarded by conversation_loop."""
    c = _cc()
    c.last_real_prompt_tokens = 0
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": "x" * 4000} for _ in range(12)
    ]
    c.compress(msgs, current_tokens=-1)
    pending = c._pending_effectiveness
    assert pending is None or pending[0] > 0


def test_outcomes_rejects_a_non_deque_window():
    """A plain list would lose maxlen and could never forget a failure."""
    c = _cc()
    c._recent_outcomes = [True, True, True, True]
    assert len(c._outcomes) <= _OUTCOME_WINDOW
