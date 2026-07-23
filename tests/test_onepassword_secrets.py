"""Offline tests for the 1Password CLI secret source."""

from __future__ import annotations

import os
from unittest import mock

from agent.secret_sources import onepassword as op_source


def test_validate_references_accepts_op_reference():
    assert op_source.validate_references(
        {"OPENAI_API_KEY": "op://Hermes/OpenAI/credential"}
    ) == []


def test_validate_references_rejects_bad_env_and_reference():
    warnings = op_source.validate_references(
        {
            "BAD-NAME": "op://Hermes/OpenAI/credential",
            "OPENAI_API_KEY": "plaintext-secret",
        }
    )
    assert len(warnings) == 2
    assert all("plaintext-secret" not in warning for warning in warnings)


def test_read_reference_uses_op_without_printing_secret(tmp_path, monkeypatch):
    binary = tmp_path / "op"
    binary.write_text("")
    completed = mock.Mock(returncode=0, stdout="super-secret", stderr="")
    run = mock.Mock(return_value=completed)
    monkeypatch.setattr(op_source.subprocess, "run", run)

    value = op_source.read_reference(
        "op://Hermes/OpenAI/credential",
        account="work",
        binary=binary,
    )

    assert value == "super-secret"
    command = run.call_args.args[0]
    assert command == [
        str(binary),
        "read",
        "op://Hermes/OpenAI/credential",
        "--no-newline",
        "--account",
        "work",
    ]


def test_read_reference_redacts_cli_error(tmp_path, monkeypatch):
    binary = tmp_path / "op"
    binary.write_text("")
    monkeypatch.setattr(
        op_source.subprocess,
        "run",
        mock.Mock(
            return_value=mock.Mock(
                returncode=1,
                stdout="",
                stderr="could not find op://Private/Secret/API",
            )
        ),
    )

    try:
        op_source.read_reference(
            "op://Hermes/OpenAI/credential",
            binary=binary,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "Private" not in message
    assert "reference was not found" in message


def test_apply_injects_values_and_tracks_only_names(tmp_path, monkeypatch):
    binary = tmp_path / "op"
    binary.write_text("")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        op_source,
        "read_reference",
        mock.Mock(return_value="super-secret"),
    )

    result = op_source.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Hermes/OpenAI/credential"},
        binary=binary,
    )

    assert os.environ["OPENAI_API_KEY"] == "super-secret"
    assert result.applied == ["OPENAI_API_KEY"]
    assert "super-secret" not in repr(result)


def test_apply_preserves_existing_value_when_override_disabled(tmp_path, monkeypatch):
    binary = tmp_path / "op"
    binary.write_text("")
    monkeypatch.setenv("OPENAI_API_KEY", "existing")
    read = mock.Mock(return_value="replacement")
    monkeypatch.setattr(op_source, "read_reference", read)

    result = op_source.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Hermes/OpenAI/credential"},
        override_existing=False,
        binary=binary,
    )

    assert os.environ["OPENAI_API_KEY"] == "existing"
    assert result.skipped == ["OPENAI_API_KEY"]
    read.assert_not_called()


def test_apply_fails_open_when_cli_missing(monkeypatch):
    monkeypatch.setattr(op_source, "find_op", mock.Mock(return_value=None))
    result = op_source.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Hermes/OpenAI/credential"},
    )
    assert result.applied == []
    assert result.error
