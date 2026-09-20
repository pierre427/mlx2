"""mlx2 command line: connect a coding agent to a running mlx2 server.

    mlx2 [--url URL] [--model ID] [--api-key-file PATH] {claude,codex,opencode} [agent args...]

Everything after the agent name is passed to the agent unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import clients


def parse_args(argv=None, environment=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if environment is None else environment
    # Split at the first agent name so the agent's own flags (``--model`` for
    # Claude, ``-c`` for Codex) are never parsed as launcher options.
    boundary = next(
        (index for index, item in enumerate(argv) if item in clients.INSTALL_URLS),
        len(argv),
    )
    argv, client_args = argv[: boundary + 1], argv[boundary + 1 :]
    if client_args[:1] == ["--"]:
        client_args = client_args[1:]
    parser = argparse.ArgumentParser(
        prog="mlx2",
        description="Launch a coding agent against a running mlx2 server.",
    )
    parser.add_argument(
        "--url",
        default=environment.get("MLX2_URL") or clients.DEFAULT_URL,
        help=f"server URL (default: MLX2_URL or {clients.DEFAULT_URL})",
    )
    parser.add_argument(
        "--model", help="served model id (default: the one /v1/models reports)"
    )
    parser.add_argument(
        "--api-key-file",
        help="owner-only 0600 file holding the server API key (default: MLX2_API_KEY)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in clients.INSTALL_URLS:
        commands.add_parser(
            name,
            help=f"run {name} against the server; later arguments go to {name}",
            add_help=False,
        )
    args = parser.parse_args(argv)
    args.client_args = client_args
    return args


def main(argv=None, *, environment=None, execvpe=os.execvpe):
    environment = dict(os.environ if environment is None else environment)
    args = parse_args(argv, environment)
    try:
        path = clients.find_executable(args.command)
        api_key = clients.resolve_api_key(environment, args.api_key_file)
        model, context, vision = clients.discover(args.url, api_key, args.model)
        argv, child_environment = clients.command(
            args.command,
            path,
            args.url,
            model,
            context,
            api_key,
            environment,
            vision=vision,
            client_args=args.client_args,
        )
    except clients.ClientError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Starting {args.command}: {model} · {context:,} context tokens", flush=True)
    if args.command in ("claude", "codex"):
        print(
            f"{args.command} hosted web search is disabled: mlx2 does not provide "
            "it. Local tools and MCP are unchanged.",
            flush=True,
        )
    execvpe(path, argv, child_environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
