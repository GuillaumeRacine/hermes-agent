"""Tests for the Slack plugin's interactive_setup wizard.

These cover the home-channel save logic that previously lived in
``hermes_cli/setup.py::_setup_slack`` before the Slack adapter migrated to a
bundled plugin (#41112). ``interactive_setup`` lazy-imports its CLI helpers
from ``hermes_cli.config`` (get_env_value / save_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), so we patch those
source modules.
"""
import os

import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.slack.adapter import _apply_yaml_config, interactive_setup


def _patch_setup_io(monkeypatch, prompts, saved):
    """Wire interactive_setup's lazy-imported CLI helpers to test doubles."""
    prompt_iter = iter(prompts)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: "")
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)
    # Manifest writing reaches out to hermes_cli.slack_cli + filesystem; stub it.
    import hermes_cli.slack_cli as slack_cli_mod
    monkeypatch.setattr(slack_cli_mod, "_build_full_manifest", lambda **_kw: {"display_information": {}})


def test_interactive_setup_saves_home_channel(monkeypatch, tmp_path):
    """interactive_setup() saves SLACK_HOME_CHANNEL when the user provides one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = {}
    # prompts: bot token, app token, allowed users (empty), home channel
    _patch_setup_io(
        monkeypatch,
        ["xoxb-test-token", "xapp-test-token", "", "C01ABC2DE3F"],
        saved,
    )

    interactive_setup()

    assert saved.get("SLACK_HOME_CHANNEL") == "C01ABC2DE3F"


def test_interactive_setup_home_channel_empty_not_saved(monkeypatch, tmp_path):
    """interactive_setup() does not save SLACK_HOME_CHANNEL when left blank."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = {}
    _patch_setup_io(
        monkeypatch,
        ["xoxb-test-token", "xapp-test-token", "", ""],
        saved,
    )

    interactive_setup()

    assert "SLACK_HOME_CHANNEL" not in saved


def test_yaml_bridge_exports_reaction_feedback_settings(monkeypatch, tmp_path):
    for name in (
        "SLACK_REACTION_FEEDBACK_CHANNELS",
        "SLACK_REACTION_FEEDBACK_USERS",
        "SLACK_REACTION_FEEDBACK_STATE",
        "SLACK_REACTION_FEEDBACK_MARKER",
    ):
        monkeypatch.delenv(name, raising=False)

    state_path = tmp_path / "feedback.json"
    _apply_yaml_config(
        {},
        {
            "reaction_feedback_channels": ["C_TRENDS"],
            "reaction_feedback_users": ["U_GUI"],
            "reaction_feedback_state": str(state_path),
            "reaction_feedback_marker": "React here.",
        },
    )

    assert os.environ["SLACK_REACTION_FEEDBACK_CHANNELS"] == "C_TRENDS"
    assert os.environ["SLACK_REACTION_FEEDBACK_USERS"] == "U_GUI"
    assert os.environ["SLACK_REACTION_FEEDBACK_STATE"] == str(state_path)
    assert os.environ["SLACK_REACTION_FEEDBACK_MARKER"] == "React here."
