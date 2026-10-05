"""Per-channel agent settings read from a platform's config section.

Lets one channel run leaner than the rest of its platform without a
separate profile. Keys live next to ``channel_prompts`` and map a channel id
(Slack/Discord channel, Telegram chat) to a list::

    slack:
      channel_disabled_toolsets:
        C0123: [delegation, image_gen, tts]
      channel_skill_focus:
        C0123: [productivity, email, apple]
      channel_skip_context_files: [C0123]   # plain list of channel ids

``channel_disabled_toolsets`` removes whole toolsets for that channel only.
``channel_skill_focus`` keeps full skill-index entries for the listed
categories and demotes every other category to a names-only line (never
hidden: every skill stays loadable with ``skill_view``).
``channel_skip_context_files`` stops cwd project files (AGENTS.md etc.) from
being injected for those channels; SOUL.md identity still loads.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

CHANNEL_DISABLED_TOOLSETS = "channel_disabled_toolsets"
CHANNEL_SKILL_FOCUS = "channel_skill_focus"
CHANNEL_SKIP_CONTEXT_FILES = "channel_skip_context_files"


def channel_list(
    config: Optional[dict],
    platform: Optional[str],
    key: str,
    channel_ids: Iterable[Optional[str]],
) -> list[str]:
    """Return the list configured under ``config[platform][key]`` for the
    first of *channel_ids* that has an entry (callers pass the exact chat id;
    on Slack that is the bare channel id, threads included).

    Accepts a single string or a list of strings. Returns ``[]`` when nothing
    matches or the config is malformed, so callers can treat it as a no-op.
    """
    if not isinstance(config, dict) or not platform:
        return []
    section = config.get(platform)
    if not isinstance(section, dict):
        return []
    mapping = section.get(key)
    if not isinstance(mapping, dict) or not mapping:
        return []
    normalized = {str(k): v for k, v in mapping.items()}
    for channel_id in channel_ids:
        if not channel_id:
            continue
        value = normalized.get(str(channel_id))
        if value is None:
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        seen: list[str] = []
        for item in value:
            name = str(item).strip()
            if name and name not in seen:
                seen.append(name)
        return seen
    return []


def apply_channel_toolset_filter(
    config: Optional[dict],
    platform: Optional[str],
    channel_id: Optional[str],
    enabled_toolsets: list[str],
    disabled_toolsets: Optional[list[str]],
) -> tuple[list[str], Optional[list[str]]]:
    """Drop the channel's disabled toolsets from the enabled list and add them
    to the disabled list. Returns the inputs unchanged when none apply."""
    blocked = channel_list(config, platform, CHANNEL_DISABLED_TOOLSETS, [channel_id])
    if not blocked:
        return enabled_toolsets, disabled_toolsets
    enabled = [ts for ts in enabled_toolsets if ts not in blocked]
    disabled = sorted(set(disabled_toolsets or []) | set(blocked))
    return enabled, disabled


def _agent_channel(agent: Any) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    platform = getattr(agent, "platform", None)
    chat_id = getattr(agent, "_chat_id", None)
    if not platform or not chat_id:
        return None, None, None
    try:
        from hermes_cli.config import read_raw_config

        return platform, str(chat_id), read_raw_config()
    except Exception:
        return None, None, None


def channel_skill_focus(agent: Any) -> frozenset[str]:
    """Skill categories to keep in full for the agent's channel, or empty."""
    platform, chat_id, config = _agent_channel(agent)
    if not config:
        return frozenset()
    return frozenset(channel_list(config, platform, CHANNEL_SKILL_FOCUS, [chat_id]))


def channel_skips_context_files(agent: Any) -> bool:
    """True when the agent's channel is listed in ``channel_skip_context_files``."""
    platform, chat_id, config = _agent_channel(agent)
    if not config:
        return False
    section = config.get(platform)
    listed = section.get(CHANNEL_SKIP_CONTEXT_FILES) if isinstance(section, dict) else None
    if isinstance(listed, str):
        listed = [listed]
    if not isinstance(listed, list):
        return False
    return chat_id in {str(x).strip() for x in listed}
