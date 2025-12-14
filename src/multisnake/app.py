# ruff: noqa: E402
import subprocess
from enum import Enum
from functools import wraps
from typing import Annotated

import typer
from rich import print

from multisnake.settings import ConfigName


class DaemonAction(str, Enum):
    starting = "start"
    stopping = "stop"
    restarting = "restart"
    checking = "status"


app = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    invoke_without_command=True,
)


@wraps(print)
def pprint(string, **kwargs) -> None:
    string = f"[bold magenta]multisnake[/]: {string}"
    print(string, **kwargs)


@app.command(
    help="Manage multiple DiscoSnake instances",
    no_args_is_help=True,
    rich_help_panel="multisnake help",
)
def cli(
    action: Annotated[
        DaemonAction,
        typer.Argument(help="Action to perform"),
    ],
    configs: Annotated[
        list[ConfigName] | None,  # type: ignore
        typer.Argument(
            help="List of instances to act on",
            show_choices=True,
            case_sensitive=False,
        ),
    ] = None,
) -> None:
    """Manage multiple DiscoSnake instances"""
    if configs is None or len(configs) == 0:
        pprint(":warning:  [bold yellow]no config specified[/] :warning:")
        raise typer.Exit(1)

    if configs[0] == ConfigName.all:
        configs = [config for config in ConfigName if config != ConfigName.all]

    config: Enum
    for config in configs:
        pprint(f"[lime]{action.name}[/] [bold cyan]{config.value}[/]...")
        cli_args = [action.value] if config.name == "default" else ["--config", config.name, action.value]
        proc = subprocess.Popen(["disco-snake", *cli_args])
        proc.communicate()
    pprint("[bold green]Done![/]")
    pass
