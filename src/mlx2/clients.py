"""Launch an installed coding agent against a running mlx2 server.

Connection and model settings travel as environment variables and command-line
overrides for one process; nothing is written to the agent's global
configuration. Tool inventories, prompts and permissions stay the agent's own.
Adapted from Splash install/clients.py (Apache-2.0); see
provenance/splash-14-agent-launcher.json.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8285"  # server.build_parser --host/--port
PROVIDER = "mlx2"

INSTALL_URLS = {
    "claude": "https://code.claude.com/docs/en/overview",
    "codex": "https://developers.openai.com/codex/cli/",
    "opencode": "https://opencode.ai/docs/",
}

# Provider switches Claude Code reads (checked against 2.1.269). Any one of
# them routes requests to a cloud provider instead of ANTHROPIC_BASE_URL.
CLAUDE_PROVIDER_FLAGS = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_GATEWAY",
    "CLAUDE_CODE_USE_MANTLE",
)


class ClientError(RuntimeError):
    pass


def output_budget(context):
    return min(32768, max(1, context // 4))


def find_executable(name):
    path = shutil.which(name)
    if path is None:
        raise ClientError(
            f"{name} is not installed or is not on PATH. "
            f"Install it first: {INSTALL_URLS[name]}"
        )
    return path


def _codex_config_args(arguments):
    """Keep config overrides together: nested Clap levels can replace the root list."""
    config, remaining = [], []
    arguments = iter(arguments)
    for argument in arguments:
        if argument == "--":
            remaining.extend([argument, *arguments])
            break
        if argument in ("-c", "--config"):
            value = next(arguments, None)
            if value is None:
                raise ClientError(f"{argument} requires a config override")
            config.extend(["-c", value])
        elif argument.startswith("--config="):
            config.extend(["-c", argument.split("=", 1)[1]])
        elif argument.startswith("-c") and len(argument) > 2:
            config.extend(["-c", argument[2:].removeprefix("=")])
        else:
            remaining.append(argument)
    return config, remaining


def command(
    name,
    path,
    base_url,
    model,
    context,
    api_key,
    environment=None,
    *,
    vision=False,
    client_args=(),
):
    """Return argv and a private environment; never mutate the caller's env."""
    if not isinstance(model, str) or not model:
        raise ClientError("the server did not report a model name")
    if type(context) is not int or context <= 0:
        raise ClientError("the server did not report a valid context limit")
    if not isinstance(api_key, str) or not api_key:
        raise ClientError("API key must be a nonempty string")
    environment = dict(os.environ if environment is None else environment)
    base_url = base_url.rstrip("/")
    endpoint = base_url + "/v1"

    if name == "claude":
        environment.update(
            ANTHROPIC_BASE_URL=base_url,
            ANTHROPIC_AUTH_TOKEN=api_key,
            ANTHROPIC_MODEL=model,
            ANTHROPIC_DEFAULT_OPUS_MODEL=model,
            ANTHROPIC_DEFAULT_SONNET_MODEL=model,
            ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
            ANTHROPIC_SMALL_FAST_MODEL=model,
            CLAUDE_CODE_SUBAGENT_MODEL=model,
            CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(context),
            CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(context),
        )
        # Select the direct endpoint even in a shell configured for a cloud
        # provider; ANTHROPIC_API_KEY would be sent as x-api-key instead of
        # the local key.
        environment.pop("ANTHROPIC_API_KEY", None)
        for key in CLAUDE_PROVIDER_FLAGS:
            environment[key] = "0"
        # WebSearch is a hosted Anthropic server tool mlx2 does not provide.
        # Start in the normal approval mode even if the user's default is auto.
        return [
            path,
            "--disallowedTools",
            "WebSearch",
            "--model",
            model,
            "--permission-mode",
            "default",
            *client_args,
        ], environment

    if name == "opencode":
        output = output_budget(context)
        reference = f"{PROVIDER}/{model}"
        try:
            config = json.loads(environment.get("OPENCODE_CONFIG_CONTENT", "{}"))
            config.update(model=reference, small_model=reference)
            # An agent-level model outranks the top-level one; point the
            # built-in agents at the served model, keeping their other settings.
            for agent in ("build", "plan", "general", "explore", "title", "compaction"):
                config.setdefault("agent", {}).setdefault(agent, {})["model"] = reference
            variants = (
                config.get("provider", {})
                .get(PROVIDER, {})
                .get("models", {})
                .get(model, {})
                .get("variants", {})
            )
            config.setdefault("provider", {})[PROVIDER] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "mlx2",
                "options": {"baseURL": endpoint, "apiKey": api_key},
                "models": {
                    model: {
                        "name": model,
                        "reasoning": True,
                        "variants": {
                            "none": {"reasoningEffort": "none"},
                            "low": {"reasoningEffort": "low"},
                            "medium": {"reasoningEffort": "medium"},
                            "high": {"reasoningEffort": "high"},
                            "xhigh": {"reasoningEffort": "xhigh"},
                            **variants,
                        },
                        "attachment": bool(vision),
                        "modalities": {
                            "input": ["text", "image"] if vision else ["text"],
                            "output": ["text"],
                        },
                        # The window includes the output allowance; an explicit
                        # input budget keeps OpenCode's compaction reserve.
                        "limit": {
                            "context": context,
                            "input": max(1, context - output),
                            "output": output,
                        },
                    }
                },
            }
        except (ValueError, TypeError, AttributeError) as error:
            raise ClientError("OPENCODE_CONFIG_CONTENT must be a JSON object") from error
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        return [path, *client_args], environment

    if name == "codex":
        # Codex reads the provider key from the variable named by env_key.
        environment["MLX2_API_KEY"] = api_key
        settings = {
            "model": json.dumps(model),
            "web_search": json.dumps("disabled"),
            "model_provider": json.dumps(PROVIDER),
            f"model_providers.{PROVIDER}": (
                '{name="mlx2",base_url='
                + json.dumps(endpoint)
                + ',env_key="MLX2_API_KEY",wire_api="responses"}'
            ),
            "model_context_window": str(context),
            # Override a threshold inherited from the user's other model; the
            # remaining 10% is room for completion and compaction itself.
            "model_auto_compact_token_limit": str(context * 9 // 10),
        }
        argv = [path]
        for key, value in settings.items():
            argv.extend(["-c", f"{key}={value}"])
        # User overrides keep their order and win over these defaults. Config
        # is global in Codex: keep the complete list at the root so a
        # subcommand's overrides cannot replace the connection settings.
        overrides, arguments = _codex_config_args(client_args)
        return [*argv, *overrides, *arguments], environment

    raise ClientError(f"unknown coding client: {name}")


def _get_json(base_url, path, api_key, *, timeout=5.0):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise ClientError(
            f"{base_url}{path} returned HTTP {error.code}"
            + (" (check MLX2_API_KEY / --api-key-file)" if error.code == 401 else "")
        ) from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise ClientError(
            f"no ready mlx2 server at {base_url} ({error}). "
            "Start one with 'mlx2-serve --model ...' first."
        ) from None


def discover(base_url, api_key, model=None, *, fetch=_get_json):
    """Return (model, context, vision) reported by the running server."""
    catalog = fetch(base_url, "/v1/models", api_key)
    entries = catalog.get("data") if isinstance(catalog, dict) else None
    entries = [
        entry
        for entry in entries or ()
        if isinstance(entry, dict) and entry.get("owned_by") == "mlx2"
    ]
    if not entries:
        raise ClientError(f"could not identify an mlx2 server at {base_url}")
    if model is None:
        model = entries[0].get("id")
    selected = next((entry for entry in entries if entry.get("id") == model), {})
    status = fetch(base_url, "/v1/status", api_key)
    context = status.get("max_context") if isinstance(status, dict) else None
    if type(context) is not int or context <= 0:
        raise ClientError(
            "mlx2 is running but its context limit is not available yet; wait and retry"
        )
    vision = "vision" in (selected.get("capabilities") or ())
    return model, context, vision


def resolve_api_key(environment, api_key_file=None):
    if api_key_file is not None:
        from .server import load_api_key_file

        try:
            return load_api_key_file(api_key_file)
        except (OSError, ValueError) as error:
            raise ClientError(str(error)) from None
    # Clients require a nonempty key even when the server checks none.
    return environment.get("MLX2_API_KEY") or "local"
