"""The live provider-quota notice (state/provider-quota.md) reaches the system prompt."""

import os
import time
from pathlib import Path

from agent import prompt_builder


def _write_notice(hermes_home: Path, text: str, age_seconds: float = 0) -> Path:
    state = hermes_home / "state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / prompt_builder.PROVIDER_QUOTA_NOTICE_FILE
    path.write_text(text, encoding="utf-8")
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
    return path


def test_notice_loaded_when_fresh(monkeypatch):
    home = prompt_builder.get_hermes_home()
    _write_notice(home, "- openai-codex: weekly window 97% used, resets Sun 22:28 EDT")
    notice = prompt_builder.load_provider_quota_notice()
    assert notice and notice.startswith("# Provider quota (live)")
    assert "97% used" in notice


def test_notice_ignored_when_stale():
    home = prompt_builder.get_hermes_home()
    _write_notice(home, "- openai-codex: 97%", age_seconds=prompt_builder.PROVIDER_QUOTA_NOTICE_MAX_AGE_SECONDS + 60)
    assert prompt_builder.load_provider_quota_notice() is None


def test_notice_absent_or_empty_is_none():
    home = prompt_builder.get_hermes_home()
    assert prompt_builder.load_provider_quota_notice() is None
    _write_notice(home, "   \n")
    assert prompt_builder.load_provider_quota_notice() is None


def test_notice_is_capped():
    home = prompt_builder.get_hermes_home()
    _write_notice(home, "x" * 5000)
    notice = prompt_builder.load_provider_quota_notice()
    assert notice and len(notice) < 2000 and notice.endswith("…")
