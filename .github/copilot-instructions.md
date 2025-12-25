# Copilot Instructions for disco-snake

## Project Overview
Discord bot using [disnake](https://disnake.dev) with AI-powered chat and image generation. Built around a multi-bot architecture where one codebase can run multiple bot instances with separate configurations and data directories.

**Core Architecture:**
- **Bot Core**: `disco_snake/bot.py` - Main `DiscoSnake(commands.Bot)` class, handles cog loading, status tasks, guild/user persistence
- **AI Cog**: `ai/core.py` - The `Ai` cog (1400+ lines) handles all LLM interactions, message context, response generation
- **Database**: PostgreSQL with pgvector, async SQLAlchemy 2.0+, Alembic migrations in `alembic/versions/`
- **Multi-instance**: `multisnake/` orchestrates multiple bot instances via subprocess, each with separate config

## Key Patterns & Conventions

### Configuration Management
Configurations live in `data/` using JSON files with Pydantic settings:
- `data/config-{name}.json` for bot settings (BotSettings in `disco_snake/settings.py`)
- `data/ai/config-{name}.json` for AI settings (get_ai_settings in `ai/settings.py`)
- `data/ai/imagen-{name}.json` for image generation configs
- Use `per_config_name()` helper for instance-specific paths

### Database Patterns
- **Async-only**: Use `Session` (async) not `SyncSession` except for Alembic migrations
- **Models**: Located in `db/{module}/` (e.g., `db/discord/user.py`, `db/ai/`)
- **Access**: `async with Session() as session:` pattern throughout
- **Migrations**: Run `alembic revision --autogenerate -m "description"` then review generated file

### Disnake Cog Structure
Cogs in `src/cogs/` are loaded dynamically via `load_extension(f"cogs.{cog}")`:
- Must have `setup(bot)` function returning the Cog instance
- AI cog is special: defined in `ai/core.py`, stub loader at `cogs/aicog.py`
- Use `@commands.Cog.listener("on_ready")` for event handlers
- Use `@commands.slash_command()` for slash commands with disnake's decorator

### AI Message Processing Flow
1. `on_message` listener checks guild/channel settings (ResponseMode, BotMode in `ai/settings.py`)
2. `get_message_context()` builds conversation history with token limits
3. Messages processed through Jinja2 prompt templates (see Prompt model in `ai/settings.py`)
4. Response sent via `AiResponse` dataclass (in `ai/types.py`) wrapping discord messages
5. All interactions logged to `db/ai/response_log.py` (AiResponseLog table)

### Async Patterns
- Extensive use of `asyncio`, all bot methods are async
- Use `bot.executor` ThreadPoolExecutor for blocking operations
- `async_lru` for caching (see `ai/eyes.py` for examples)
- Never use sync DB sessions in async context

## Development Workflows

### Running the Bot
```bash
# Single instance (uses data/config.json)
disco-snake start

# Named instance (uses data/config-{name}.json)
disco-snake --config aiki start

# Multi-instance management
multisnake start aiki disco nya  # start multiple
multisnake status all            # check all
```

### Database Operations
```bash
# Create migration after modifying models
alembic revision --autogenerate -m "add new field"

# Review and edit in alembic/versions/, then apply
alembic upgrade head

# Use -x port=2027 to override port from command line
alembic -x port=2027 upgrade head
```

### Code Quality
```bash
# Format and lint (follows pyproject.toml [tool.ruff])
ruff check --fix
ruff format

# Run pre-commit hooks
pre-commit run --all-files
```

## Critical Integration Points

### External Services
- **LLM APIs**: OpenAI-compatible endpoints configured in `ai/client/` (openai.py, common.py)
  - Base classes: `LLMApiClientAsyncBase`, `LLMStreamingApiClientAsyncBase`
  - Configured via LMApiConfig in AI settings
- **Image Captioning**: `ai/eyes.py` DiscoEyes class for attachment perception
- **Image Generation**: `ai/imagen.py` Imagen class talks to Stable Diffusion WebUI API
- **Gradio UI**: `ai/web/gradio.py` provides web interface for testing (optional)

### Cog Communication
- Cogs access bot instance: `self.bot` inside Cog methods
- Shared state via `bot.config` (BotSettings) and cog attributes
- Database is global via `Session()` context manager
- No inter-cog dependencies except via bot instance

## Project-Specific Quirks

### Type Hints
- Modern Python 3.12+ syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`
- Avoid `typing` module for stdlib types per python.instructions.md
- Pydantic v2 models everywhere (BaseModel, Field, field_validator)

### Disnake Specifics
- Use `disnake` not `discord.py` - similar but different APIs
- Function defaults in decorators are allowed (ignore B008 ruff warning)
- Slash commands use InteractionContextTypes for context awareness

### Settings Architecture
- `GuildSettings` contains per-guild config with nested `ChannelSettings`
- `ResponseMode` enum: NoRespond, Mentioned, IdleAuto, FullAuto, Unlimited
- `BotMode` enum: Strip, Ignore, Siblings, All (controls bot message visibility)

### Token Management
- Tokenizers cached in `ai/tokenizers/` with `extract_tokenizer()` helper
- Context window management in `get_message_context()` respects model token limits
- Uses transformers' AutoTokenizer with custom PreTrainedTokenizerBase wrapper

## File Reference Examples
- Bot initialization: [disco_snake/bot.py](src/disco_snake/bot.py#L58)
- AI message handler: [ai/core.py](src/ai/core.py#L405)
- Database models: [db/discord/user.py](src/db/discord/user.py), [db/ai/response_log.py](src/db/ai/response_log.py)
- Settings definitions: [ai/settings.py](src/ai/settings.py#L33), [disco_snake/settings.py](src/disco_snake/settings.py#L66)

## Common Gotchas
- Don't forget `await` on async methods - this is async-first codebase
- Config paths are instance-specific; use `per_config_name()` from `disco_snake/__init__.py`
- Alembic env.py reads bot config for DB connection; ensure config exists before migrations
- DiscoEyes image captions are cached in DB; check `db/ai/image_caption.py` before re-requesting
- Guild/channel settings cascade: channel settings override guild defaults (see `channel_respond_mode()`)
