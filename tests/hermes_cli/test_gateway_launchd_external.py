import sys
from pathlib import Path

import hermes_cli.gateway as gateway_cli


def test_launchd_resolves_interpreter_when_parent_crosses_symlink(
    tmp_path,
    monkeypatch,
):
    """External-volume compatibility avoids launchd posix_spawn EPERM."""
    external = tmp_path / "external"
    venv = external / "venv"
    bin_dir = venv / "bin"
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    bin_dir.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)
    linked_venv = linked / "venv"

    monkeypatch.setattr(gateway_cli, "_detect_venv_dir", lambda: linked_venv)
    monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", linked)

    program, pythonpath = gateway_cli._launchd_python_invocation()

    assert program == str(Path(sys.executable).resolve())
    assert str(linked) in pythonpath
    assert str(site_packages).replace(str(venv), str(linked_venv)) in pythonpath


def test_launchd_plist_uses_real_user_home_as_working_directory(
    tmp_path,
    monkeypatch,
):
    external_home = tmp_path / "external-hermes"
    external_home.mkdir()
    user_home = tmp_path / "user"
    user_home.mkdir()
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: external_home)
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: user_home)

    plist = gateway_cli.generate_launchd_plist()

    assert f"<string>{user_home}</string>" in plist
    assert (
        f"<key>WorkingDirectory</key>\n    <string>{external_home}</string>"
        not in plist
    )
    assert f"<string>{user_home}/Library/Logs/gateway.log</string>" in plist
    assert f"<string>{user_home}/Library/Logs/gateway.error.log</string>" in plist
