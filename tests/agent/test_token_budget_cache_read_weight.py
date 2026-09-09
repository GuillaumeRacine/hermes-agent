"""Cache-read weighting in the token budget (hermes-agent#62).

Providers report a re-sent cached prompt prefix inside ``total_tokens`` at
full weight, but bill it at roughly a tenth of a fresh input token.  Charging
it raw made the budget measure prompt volume re-sent rather than spend: on
425 recorded turns, 56 breached ``per_turn=1.5M`` on raw totals while only 1
breached on a cost-weighted total.

The numbers in the "real turn" tests below are copied verbatim from
``~/.hermes/logs/adaptive-routing.jsonl`` so the regression is pinned to
observed production traffic rather than invented figures.
"""

import pytest

from agent.token_budget import DEFAULT_CACHE_READ_WEIGHT, TokenBudget
from hermes_cli.config import resolve_token_budget


def test_cache_heavy_and_fresh_heavy_turns_of_equal_raw_total_differ():
    """Same ``total_tokens``, opposite composition -> very different charge."""
    cache_heavy = TokenBudget(per_turn=1_000_000)
    cache_heavy.record(
        {"input_tokens": 100_000, "output_tokens": 0,
         "cache_read_tokens": 900_000, "total_tokens": 1_000_000}
    )

    fresh_heavy = TokenBudget(per_turn=1_000_000)
    fresh_heavy.record(
        {"input_tokens": 1_000_000, "output_tokens": 0,
         "cache_read_tokens": 0, "total_tokens": 1_000_000}
    )

    assert cache_heavy.turn_tokens == 100_000 + 90_000   # 0.1 * 900k
    assert fresh_heavy.turn_tokens == 1_000_000
    # Raw totals are identical; only the charge differs.
    assert cache_heavy.turn_raw_tokens == fresh_heavy.turn_raw_tokens == 1_000_000
    assert cache_heavy.breach() is None
    assert fresh_heavy.breach() is None      # equal to the limit, not over
    fresh_heavy.record({"input_tokens": 1, "output_tokens": 0})
    assert fresh_heavy.breach() == "per_turn"


def test_openai_convention_subtracts_cached_from_prompt_tokens():
    """``prompt_tokens`` includes the cached prefix; it must not double-count."""
    tb = TokenBudget()
    tb.record(
        {"prompt_tokens": 100_000, "completion_tokens": 500,
         "cached_tokens": 90_000, "total_tokens": 100_500}
    )
    # fresh = 100_000 - 90_000 = 10_000; + 500 completion; + 0.1 * 90_000
    assert tb.turn_tokens == 10_000 + 500 + 9_000


def test_anthropic_convention_uses_input_tokens_as_is():
    """``input_tokens`` already excludes cache; subtracting would undercount."""
    tb = TokenBudget()
    tb.record(
        {"input_tokens": 10_000, "output_tokens": 500,
         "cache_read_input_tokens": 90_000}
    )
    assert tb.turn_tokens == 10_000 + 500 + 9_000


def test_reasoning_tokens_are_not_double_counted():
    """Reasoning is a breakdown of output, not an extra charge.

    Verified across 261/261 production turns carrying reasoning tokens:
    ``total == input + output + cache_read + cache_write`` exactly.
    """
    tb = TokenBudget()
    tb.record(
        {"input_tokens": 1_000, "output_tokens": 400,
         "reasoning_tokens": 300, "cache_read_tokens": 0}
    )
    assert tb.turn_tokens == 1_400          # not 1_700


def test_total_only_usage_is_charged_in_full():
    """No component fields -> nothing to weight; charge raw, never zero.

    Undercounting to 0 would silently disable the budget, which is worse
    than the overcounting this change exists to remove.
    """
    tb = TokenBudget(per_turn=1_000)
    assert tb.record({"total_tokens": 5_000}) == 5_000
    assert tb.turn_tokens == 5_000
    assert tb.breach() == "per_turn"


def test_context_soft_limit_counts_the_cached_prefix():
    """A 100k context is 100k whether or not the provider served it cached."""
    tb = TokenBudget(context_soft_limit=50_000)
    tb.record(
        {"input_tokens": 5_000, "output_tokens": 100,
         "cache_read_tokens": 90_000}
    )
    assert tb.last_prompt_tokens == 95_000
    assert tb.consume_compression_request() is True


@pytest.mark.parametrize(
    "given,expected",
    [(0, 0.0), (1, 1.0), (0.25, 0.25), (-1, 0.0), (5, 1.0),
     ("bad", DEFAULT_CACHE_READ_WEIGHT), (None, DEFAULT_CACHE_READ_WEIGHT)],
)
def test_weight_is_clamped_and_falls_back(given, expected):
    assert TokenBudget(cache_read_weight=given).cache_read_weight == expected


def test_weight_of_one_restores_previous_behaviour():
    tb = TokenBudget(cache_read_weight=1.0)
    tb.record(
        {"input_tokens": 100_000, "output_tokens": 0,
         "cache_read_tokens": 900_000, "total_tokens": 1_000_000}
    )
    assert tb.turn_tokens == tb.turn_raw_tokens == 1_000_000


def test_config_plumbs_the_weight_through():
    resolved = resolve_token_budget(
        {"agent": {"token_budget": {"per_turn": 10, "cache_read_weight": 0.5}}}, None
    )
    assert resolved["cache_read_weight"] == 0.5
    assert TokenBudget.from_resolved(resolved).cache_read_weight == 0.5


# ── real production turns ────────────────────────────────────────────


def test_the_wink_turn_no_longer_stops():
    """2026-09-08T22:21 gateway turn, decision bca239bd4db5801c7dabc55b.

    Raw 1,536,639 tripped ``per_turn=1,500,000``.  Two thirds of that was a
    re-read cached prefix; the real cost is 43% of the cap.
    """
    tb = TokenBudget(per_turn=1_500_000, per_session=4_000_000)
    charged = tb.record(
        {"input_tokens": 529_791, "output_tokens": 19_840,
         "cache_read_tokens": 987_008, "reasoning_tokens": 18_962,
         "total_tokens": 1_536_639}
    )
    assert charged == 648_332
    assert tb.turn_raw_tokens == 1_536_639
    assert tb.breach() is None


def test_a_genuinely_runaway_turn_still_stops():
    """2026-08-26T16:26 C4 turn: 60 API calls, 3.27M *fresh* input tokens.

    Guards against the fix turning into a blanket amnesty -- this one is
    expensive on any accounting and must still be caught.
    """
    tb = TokenBudget(per_turn=1_500_000, per_session=4_000_000)
    tb.record(
        {"input_tokens": 3_273_471, "output_tokens": 6_381,
         "cache_read_tokens": 4_975_104, "total_tokens": 8_254_956}
    )
    assert tb.turn_tokens == 3_777_362
    # Under the 4M session cap but 2.5x over the 1.5M turn cap: it still
    # stops, on the turn limit.
    assert tb.breach() == "per_turn"
    assert tb.turn_tokens > 1_500_000


def test_stop_message_shows_both_numbers():
    tb = TokenBudget(per_turn=1_500_000, per_session=1_000_000)
    tb.record(
        {"input_tokens": 529_791, "output_tokens": 19_840,
         "cache_read_tokens": 987_008, "total_tokens": 1_536_639}
    )
    msg = tb.stop_message()
    assert "648,332" in msg
    assert "raw 1,536,639" in msg
    assert "0.1x" in msg
