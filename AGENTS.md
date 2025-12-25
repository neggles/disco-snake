# AGENTS

## Overview
- Discord bot built on disnake; `disco-snake` CLI boots `DiscoSnake` and loads cogs from `src/cogs` (src/disco_snake/cli.py, src/disco_snake/bot.py).
- AI functionality is a cog: stub loader `src/cogs/aicog.py`, main logic in `src/ai/core.py`.

## Architecture & data flow
- `Ai.on_message` filters/guards, builds context via `get_message_context`, renders Jinja2 prompts, and sends via `AiResponse` (src/ai/core.py, src/ai/settings.py, src/ai/types.py).
- Responses are logged to Postgres in `ai_message_logs` with JSON/ARRAY columns (src/db/ai/logs.py).
- Image generation uses `Imagen` to call Stable Diffusion WebUI `/sdapi/v1/txt2img`, writes outputs under `data/ai/...` (src/ai/imagen.py, src/ai/constants.py).

## Config & data layout
- Bot config is JSON merged from `data/config.json` + `data/config-{name}.json`; AI config merges `data/ai/config.json` + `data/ai/config-{name}.json` (src/disco_snake/settings.py, src/ai/settings.py).
- Per-instance paths come from `config_suffix()`/`per_config_name()`; logs/data are namespaced by config name (src/disco_snake/__init__.py).
- Chat templates live in `data/ai/chat_templates/` and hot-reload on mtime via `Prompt.template_str` (src/ai/settings.py).

## Workflows
- Run single instance: `disco-snake start` (uses `data/config.json`).
- Run named instance: `disco-snake --config aiki start` (uses suffixed configs).
- Multi-instance orchestration: `multisnake start aiki disco nya` (src/multisnake/app.py).
- DB migrations read `db_uri` from bot config; optional port override: `alembic -x port=2027 upgrade head` (alembic/env.py).
- Lint/format uses Ruff: `ruff check --fix` and `ruff format` (pyproject.toml).
- Pre-commit hooks are configured in `.pre-commit-config.yaml` (run `pre-commit run --all-files`).

## Conventions & gotchas
- Async-first DB access: use `Session` from `src/db/engine.py` in app code; `SyncSession` is for alembic only.
- Disnake-specific APIs: use `disnake` decorators for slash commands; cogs must expose `setup(bot)` to be loadable (src/cogs/).
- Typing style avoids `typing.List`/`Optional`; prefer `list[str]` and `X | None` (see .github/instructions/python.instructions.md).

## Integrations
- LLM clients in `src/ai/client/` talk to OpenAI-compatible endpoints configured by `LMApiConfig` (src/ai/settings.py).
- Vision captioning uses `DiscoEyes` to call the caption API (`VisionConfig.host`/`route`) and caches captions in `image_captions` (src/ai/eyes.py, src/db/ai/vision.py).
