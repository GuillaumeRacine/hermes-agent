"""Regression: the test suite must never write the LIVE ~/.hermes/logs files.

``run_agent._hermes_home`` and ``gateway.run._hermes_home`` are resolved at
IMPORT time. Test modules import them at collection time — before the autouse
``_hermetic_environment`` fixture points HERMES_HOME at a tempdir — so both
caches held the real ``~/.hermes``. ``agent/agent_init.py`` then called
``setup_logging(hermes_home=run_agent._hermes_home)`` and every AIAgent built in
a test attached a RotatingFileHandler to the real agent.log / errors.log
(2026-10-05: 630 agent.log + 1,566 errors.log lines from one gateway run).

The module-level imports below are deliberate: they reproduce the
collection-time import that created the stale cache.
"""

import logging
import os
import uuid
from pathlib import Path

import sys

import gateway.run  # noqa: F401  (collection-time import, see docstring)
import hermes_logging
import run_agent  # noqa: F401
from tests.conftest import _REAL_HERMES_HOMES, _strip_real_home_log_handlers


def _real_log_tails(n_bytes: int = 256 * 1024) -> str:
    # The live gateway keeps appending to these files, so size comparison is
    # racy; a unique marker that must never appear is not.
    chunks = []
    for home in _REAL_HERMES_HOMES:
        for name in ("agent.log", "errors.log", "gateway.log"):
            p = home / "logs" / name
            if p.exists():
                with p.open("rb") as fh:
                    fh.seek(max(0, p.stat().st_size - n_bytes))
                    chunks.append(fh.read().decode("utf-8", "replace"))
    return "\n".join(chunks)


def _live(name: str):
    # Resolve through sys.modules like agent_init._ra() does: some suites
    # drop and re-import run_agent, so the collection-time binding can be stale.
    return sys.modules[name]


def _under_real_home(path: str) -> bool:
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(h) for h in _REAL_HERMES_HOMES)


def test_import_time_hermes_home_caches_point_at_test_home():
    fake = Path(os.environ["HERMES_HOME"])
    assert _live("run_agent")._hermes_home == fake
    assert _live("gateway.run")._hermes_home == fake


def test_logging_setup_from_cached_home_never_targets_real_logs():
    marker = f"log-isolation-marker-{uuid.uuid4().hex}"
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        # Exactly what AIAgent.__init__ and gateway startup do.
        hermes_logging.setup_logging(hermes_home=_live("run_agent")._hermes_home, force=True)
        hermes_logging.setup_logging(
            hermes_home=_live("gateway.run")._hermes_home, mode="gateway", force=True
        )
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers, "setup_logging attached no file handlers"
        leaked = [h.baseFilename for h in file_handlers if _under_real_home(h.baseFilename)]
        assert leaked == [], f"log handlers point at the live runtime: {leaked}"

        logging.getLogger("gateway.test_log_isolation").warning(marker)
        for h in file_handlers:
            h.flush()
        assert marker not in _real_log_tails()
        own_log = Path(os.environ["HERMES_HOME"]) / "logs" / "agent.log"
        assert marker in own_log.read_text()
    finally:
        for h in list(root.handlers):
            if h not in original:
                root.removeHandler(h)
                h.close()
        hermes_logging._logging_initialized = False


def test_guard_strips_handlers_aimed_at_real_home():
    root = logging.getLogger()
    real_log = next(iter(_REAL_HERMES_HOMES)) / "logs" / "agent.log"
    # delay=True: the file is never opened, so nothing touches the live log.
    stray = logging.FileHandler(real_log, delay=True)
    root.addHandler(stray)
    try:
        _strip_real_home_log_handlers()
        assert stray not in root.handlers
    finally:
        if stray in root.handlers:
            root.removeHandler(stray)
        stray.close()
