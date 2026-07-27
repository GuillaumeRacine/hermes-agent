import sys
import types

import cli


def test_deferred_startup_syncs_skills_before_extension_discovery(monkeypatch):
    events = []

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.setattr(cli, "_deferred_agent_startup_done", False)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.main",
        types.SimpleNamespace(
            _sync_bundled_skills_for_startup=lambda: events.append("skills")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: events.append("plugins")),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        types.SimpleNamespace(
            start_background_mcp_discovery=lambda **_kwargs: events.append("mcp")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(
            register_from_config=lambda _config, accept_hooks=False: events.append(
                ("hooks", accept_hooks)
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(load_config=lambda: {}),
    )

    cli._prepare_deferred_agent_startup()
    cli._prepare_deferred_agent_startup()

    assert events == ["skills", "plugins", "mcp", ("hooks", False)]
