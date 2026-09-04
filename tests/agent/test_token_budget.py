"""Unit tests for agent/token_budget.py and hermes_cli.config.resolve_token_budget.

hermes-home #233 P1-4: hard per-session / per-turn token budget.
"""

from types import SimpleNamespace

from agent.token_budget import TokenBudget, is_continue_message
from hermes_cli.config import DEFAULT_CONFIG, resolve_token_budget


# ── resolve_token_budget ──────────────────────────────────────────────


def test_defaults_are_unlimited_and_stop():
    assert DEFAULT_CONFIG["agent"]["token_budget"] == {
        "per_session": 0,
        "per_turn": 0,
        "context_soft_limit": 0,
        "action": "stop",
        "platforms": {},
    }
    resolved = resolve_token_budget({}, "cli")
    assert resolved == {
        "per_session": 0,
        "per_turn": 0,
        "context_soft_limit": 0,
        "action": "stop",
        "platform": None,
    }
    assert resolve_token_budget(None, None)["action"] == "stop"


def test_top_level_values_and_coercion():
    cfg = {"agent": {"token_budget": {
        "per_session": "600000",     # string -> int
        "per_turn": -5,              # negative -> 0 (unlimited)
        "context_soft_limit": 80000.0,
        "action": "WARN",            # case-insensitive
    }}}
    resolved = resolve_token_budget(cfg, "cli")
    assert resolved["per_session"] == 600000
    assert resolved["per_turn"] == 0
    assert resolved["context_soft_limit"] == 80000
    assert resolved["action"] == "warn"

    bad = {"agent": {"token_budget": {"action": "explode", "per_session": "lots"}}}
    resolved = resolve_token_budget(bad, "cli")
    assert resolved["action"] == "stop"
    assert resolved["per_session"] == 0


def test_platform_override_merges_over_top_level():
    cfg = {"agent": {"token_budget": {
        "per_session": 1_000_000,
        "per_turn": 200_000,
        "action": "warn",
        "platforms": {
            "Slack": {"per_session": 600_000, "per_turn": 150_000, "action": "stop"},
            "cron": {"per_turn": 50_000},
        },
    }}}
    slack = resolve_token_budget(cfg, "slack")  # case-insensitive match
    assert slack["per_session"] == 600_000
    assert slack["per_turn"] == 150_000
    assert slack["action"] == "stop"
    assert slack["platform"] == "Slack"

    cron = resolve_token_budget(cfg, "cron")
    assert cron["per_session"] == 1_000_000   # inherited from top level
    assert cron["per_turn"] == 50_000
    assert cron["action"] == "warn"            # inherited from top level

    cli = resolve_token_budget(cfg, "cli")     # no override
    assert cli["per_session"] == 1_000_000
    assert cli["per_turn"] == 200_000
    assert cli["platform"] is None


def test_from_config_builds_budget_for_platform():
    cfg = {"agent": {"token_budget": {
        "per_session": 100, "platforms": {"slack": {"per_turn": 40}},
    }}}
    tb = TokenBudget.from_config(cfg, "slack")
    assert (tb.per_session, tb.per_turn, tb.action, tb.platform) == (100, 40, "stop", "slack")
    assert tb.enabled
    assert not TokenBudget.from_config({}, "cli").enabled


# ── counters ──────────────────────────────────────────────────────────


def test_record_accumulates_turn_and_session_and_last_prompt():
    tb = TokenBudget(per_session=10_000, per_turn=1_000)
    assert tb.record({"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}) == 600
    assert (tb.turn_tokens, tb.session_tokens, tb.last_prompt_tokens) == (600, 600, 500)

    # Anthropic-style object without total_tokens: total is derived.
    tb.record(SimpleNamespace(input_tokens=300, output_tokens=50))
    assert (tb.turn_tokens, tb.session_tokens, tb.last_prompt_tokens) == (950, 950, 300)

    tb.reset_turn()
    assert tb.turn_tokens == 0
    assert tb.session_tokens == 950          # session survives the turn reset
    assert tb.record(None) == 0

    tb.reset_session()
    assert (tb.turn_tokens, tb.session_tokens, tb.api_calls) == (0, 0, 0)


def test_breach_precedence_and_exceeded_state():
    tb = TokenBudget(per_session=1_500, per_turn=1_000)
    tb.record({"total_tokens": 600})
    assert tb.breach() is None
    tb.record({"total_tokens": 600})          # turn 1200 > 1000, session 1200 <= 1500
    assert tb.breach() == "per_turn"
    assert tb.exceeded is False               # per_turn breach is not sticky

    tb.reset_turn()
    tb.record({"total_tokens": 400})          # session 1600 > 1500
    assert tb.breach() == "per_session"
    assert tb.exceeded is True
    assert tb.session_exceeded()

    assert tb.stop_message() == (
        "Stopped: this session has used 1,600 tokens of its 1,500 budget "
        "(turn: 400/1,000). Reply `continue` to allow one more turn, or start a new session."
    )
    assert tb.summary() == "400/1600/1500"


def test_warn_mode_never_sets_exceeded_and_warns_once_per_breach():
    tb = TokenBudget(per_session=1_000, per_turn=500, action="warn")
    assert not tb.stops
    tb.record({"total_tokens": 600})
    assert tb.breach() == "per_turn"
    assert tb.should_warn("per_turn") is True
    assert tb.should_warn("per_turn") is False      # once per turn
    tb.reset_turn()
    tb.record({"total_tokens": 600})                # session 1200 > 1000
    assert tb.breach() == "per_session"
    assert tb.exceeded is False
    assert tb.should_warn("per_session") is True
    assert tb.should_warn("per_session") is False   # once per session
    assert tb.should_warn(None) is False


def test_unlimited_budget_never_breaches():
    tb = TokenBudget()
    tb.record({"total_tokens": 10_000_000})
    assert tb.breach() is None
    assert tb.summary() == "10000000/10000000"
    assert "unlimited" in tb.stop_message()


# ── continue override arithmetic ──────────────────────────────────────


def test_grant_extension_uses_per_turn_when_set():
    tb = TokenBudget(per_session=5_000, per_turn=1_000)
    tb.record({"total_tokens": 5_200})
    assert tb.breach() == "per_session" and tb.exceeded
    assert tb.extension_amount() == 1_000
    assert tb.grant_extension() == 1_000
    # limit = max(old limit, spent) + per_turn -> the extra turn is real headroom
    assert tb.per_session == 6_200
    assert tb.exceeded is False
    assert tb.extensions_granted == 1
    assert not tb.session_exceeded()


def test_grant_extension_uses_20pct_when_per_turn_unset():
    tb = TokenBudget(per_session=10_000)
    tb.record({"total_tokens": 10_500})
    assert tb.breach() == "per_session"
    assert tb.extension_amount() == 2_000
    assert tb.grant_extension() == 2_000
    assert tb.per_session == 12_500


def test_grant_extension_noop_when_no_session_limit():
    tb = TokenBudget(per_turn=1_000)
    assert tb.grant_extension() == 1_000   # amount reported, but no session cap to raise
    assert tb.per_session == 0


def test_is_continue_message():
    assert is_continue_message("continue")
    assert is_continue_message("  Continue \n")
    assert is_continue_message("CONTINUE")
    assert not is_continue_message("continue please")
    assert not is_continue_message("")
    assert not is_continue_message(["continue"])
    assert not is_continue_message(None)


# ── context soft limit ────────────────────────────────────────────────


def test_soft_limit_requests_compression_once():
    tb = TokenBudget(context_soft_limit=4_000)
    tb.record({"prompt_tokens": 3_000, "completion_tokens": 10})
    assert tb.consume_compression_request() is False
    tb.record({"prompt_tokens": 5_000, "completion_tokens": 10})
    assert tb.compression_requested()
    assert tb.consume_compression_request() is True
    assert tb.consume_compression_request() is False     # consumed
    assert tb.should_warn_soft_limit_unavailable() is True
    assert tb.should_warn_soft_limit_unavailable() is False  # once per turn
    tb.reset_turn()
    assert tb.should_warn_soft_limit_unavailable() is True
