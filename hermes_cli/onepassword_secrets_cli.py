"""CLI handlers for ``hermes secrets onepassword ...``."""

from __future__ import annotations

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import onepassword as op_source
from hermes_cli.config import load_config, save_config


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    sub = parent_parser.add_subparsers(dest="secrets_op_command")

    setup = sub.add_parser(
        "setup",
        help="Configure op:// references (secret values never enter config)",
    )
    setup.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="ENV_VAR=op://vault/item/field",
        help="Add or replace a secret reference; repeat for multiple keys",
    )
    setup.add_argument(
        "--account",
        default=None,
        help="Optional 1Password account shorthand or sign-in address",
    )
    setup.add_argument(
        "--no-override",
        action="store_true",
        help="Keep existing environment values instead of replacing them",
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show configuration and CLI availability")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Resolve configured references now")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Inject resolved values into this Hermes process (default: verify only)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Disable the 1Password integration")
    disable.set_defaults(func=cmd_disable)


def _parse_reference_args(values: list[str]) -> tuple[dict[str, str], list[str]]:
    references: dict[str, str] = {}
    errors: list[str] = []
    for raw in values:
        if "=" not in raw:
            errors.append(f"{raw!r}: expected ENV_VAR=op://vault/item/field")
            continue
        env_name, reference = raw.split("=", 1)
        env_name = env_name.strip()
        reference = reference.strip()
        warnings = op_source.validate_references({env_name: reference})
        if warnings:
            errors.extend(warnings)
            continue
        references[env_name] = reference
    return references, errors


def _config() -> tuple[dict, dict]:
    cfg = load_config()
    op_cfg = (cfg.setdefault("secrets", {}).setdefault("onepassword", {}))
    return cfg, op_cfg


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    references, errors = _parse_reference_args(list(args.reference or []))
    if errors:
        for error in errors:
            console.print(f"[red]{error}[/red]")
        return 1

    cfg, op_cfg = _config()
    existing = op_cfg.get("references")
    if not isinstance(existing, dict):
        existing = {}
    if references:
        existing.update(references)
    if not existing:
        console.print(
            "[red]No references configured.[/red]\n"
            "Use: hermes secrets onepassword setup "
            "--reference 'OPENAI_API_KEY=op://vault/item/field'"
        )
        return 1

    op_cfg["enabled"] = True
    op_cfg["references"] = existing
    op_cfg["override_existing"] = not bool(args.no_override)
    if args.account is not None:
        op_cfg["account"] = str(args.account).strip()
    op_cfg.setdefault("account", "")
    op_cfg.setdefault("binary", "")
    save_config(cfg)

    console.print(
        Panel.fit(
            "[green]1Password secret source enabled.[/green]\n"
            f"Configured references: {len(existing)}\n\n"
            "Only op:// references were saved. Secret values remain in 1Password.",
            title="1Password",
        )
    )
    return cmd_sync(argparse.Namespace(apply=False))


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    _cfg, op_cfg = _config()
    references = op_cfg.get("references")
    if not isinstance(references, dict):
        references = {}
    binary = op_source.find_op(op_cfg.get("binary") or None)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled", "yes" if op_cfg.get("enabled") else "no")
    table.add_row("op CLI", str(binary) if binary else "[yellow]not found[/yellow]")
    table.add_row("Account", str(op_cfg.get("account") or "[dim](active account)[/dim]"))
    table.add_row("References", str(len(references)))
    table.add_row(
        "Override existing",
        "yes" if op_cfg.get("override_existing", True) else "no",
    )
    console.print(Panel(table, title="1Password", border_style="cyan"))
    if references:
        names = Table(show_header=True, header_style="bold")
        names.add_column("Environment variable")
        names.add_column("Reference configured")
        for env_name in sorted(references):
            names.add_row(env_name, "yes")
        console.print(names)
    return 0

def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    _cfg, op_cfg = _config()
    if not op_cfg.get("enabled"):
        console.print(
            "[yellow]1Password integration is disabled. Run "
            "`hermes secrets onepassword setup` first.[/yellow]"
        )
        return 1

    references = op_cfg.get("references") or {}
    original: dict[str, str | None] = {
        name: os.environ.get(name)
        for name in references
        if isinstance(name, str)
    }
    result = op_source.apply_onepassword_secrets(
        enabled=True,
        references=references,
        override_existing=bool(op_cfg.get("override_existing", True)),
        account=str(op_cfg.get("account", "") or "").strip(),
        binary=str(op_cfg.get("binary", "") or "").strip() or None,
    )

    if not args.apply:
        for name, old_value in original.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

    table = Table(show_header=True, header_style="bold")
    table.add_column("Environment variable")
    table.add_column("Result")
    for name in sorted(result.applied):
        table.add_row(name, "[green]applied[/green]" if args.apply else "[green]verified[/green]")
    for name in sorted(result.skipped):
        table.add_row(name, "[dim]skipped (already set)[/dim]")
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if result.error:
        console.print(f"[red]{result.error}[/red]")
        return 1
    if not result.applied and not result.skipped:
        console.print("[yellow]No references were resolved.[/yellow]")
        return 1
    if not args.apply:
        console.print(
            "\nVerified without retaining values in this process. "
            "Use [cyan]--apply[/cyan] or restart Hermes to load them."
        )
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg, op_cfg = _config()
    op_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green] Configured op:// references were retained; "
        "no secret values are stored by Hermes."
    )
    return 0
