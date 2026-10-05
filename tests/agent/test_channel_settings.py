"""Per-channel toolset and skill-focus settings (hermes-home#330)."""

from types import SimpleNamespace

from agent.channel_settings import (
    apply_channel_toolset_filter,
    channel_list,
    channel_skill_focus,
    channel_skips_context_files,
)
from agent.prompt_builder import build_skills_system_prompt

CFG = {
    "slack": {
        "channel_disabled_toolsets": {"C1": ["delegation", "tts", "delegation"]},
        "channel_skill_focus": {"C1": "productivity"},
    }
}


class TestChannelList:
    def test_exact_match_dedupes(self):
        assert channel_list(CFG, "slack", "channel_disabled_toolsets", ["C1"]) == [
            "delegation",
            "tts",
        ]

    def test_string_value_becomes_list(self):
        assert channel_list(CFG, "slack", "channel_skill_focus", ["C1"]) == ["productivity"]

    def test_parent_fallback(self):
        assert channel_list(CFG, "slack", "channel_skill_focus", ["T9", "C1"]) == ["productivity"]

    def test_no_match_or_bad_config_is_noop(self):
        assert channel_list(CFG, "slack", "channel_skill_focus", ["C2"]) == []
        assert channel_list(CFG, "discord", "channel_skill_focus", ["C1"]) == []
        assert channel_list({"slack": {"channel_skill_focus": ["C1"]}}, "slack", "channel_skill_focus", ["C1"]) == []
        assert channel_list(None, "slack", "channel_skill_focus", ["C1"]) == []
        assert channel_list(CFG, "slack", "channel_skill_focus", [None]) == []


class TestToolsetFilter:
    def test_removes_from_enabled_and_adds_to_disabled(self):
        enabled, disabled = apply_channel_toolset_filter(
            CFG, "slack", "C1", ["delegation", "file", "terminal", "tts"], ["moa"]
        )
        assert enabled == ["file", "terminal"]
        assert disabled == ["delegation", "moa", "tts"]

    def test_other_channels_unchanged(self):
        enabled = ["delegation", "file"]
        assert apply_channel_toolset_filter(CFG, "slack", "C2", enabled, None) == (enabled, None)


class TestSkillFocus:
    def _skills(self, tmp_path):
        for cat, name in (("productivity", "ea-memory"), ("gstack", "ship"), ("social-media/x", "post")):
            d = tmp_path / "skills" / cat / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Does {name} things\n---\n")

    def test_focus_demotes_everything_else_to_names_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._skills(tmp_path)
        focused = build_skills_system_prompt(focus_categories=frozenset({"productivity"}))
        assert "Does ea-memory things" in focused
        # Names stay visible and loadable; descriptions dropped.
        assert "gstack [names only]: ship" in focused
        assert "social-media/x [names only]: post" in focused
        assert "Does ship things" not in focused
        assert "this channel's focus" in focused
        # Unfocused call is not served from the focused cache entry.
        full = build_skills_system_prompt()
        assert "Does ship things" in full

    def test_agent_focus_reads_raw_config(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: CFG)
        agent = SimpleNamespace(platform="slack", _chat_id="C1")
        assert channel_skill_focus(agent) == frozenset({"productivity"})
        assert channel_skill_focus(SimpleNamespace(platform="slack", _chat_id=None)) == frozenset()

    def test_skip_context_files_by_channel(self, monkeypatch):
        cfg = {"slack": {"channel_skip_context_files": ["C1"]}}
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: cfg)
        assert channel_skips_context_files(SimpleNamespace(platform="slack", _chat_id="C1"))
        assert not channel_skips_context_files(SimpleNamespace(platform="slack", _chat_id="C2"))
        assert not channel_skips_context_files(SimpleNamespace(platform="discord", _chat_id="C1"))
