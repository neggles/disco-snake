# pyright: reportArgumentType=false
import logging
from collections.abc import Iterator
from copy import deepcopy
from enum import Enum
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, RootModel, field_validator
from pydantic_settings import SettingsConfigDict

from disco_snake import LOG_FORMAT, LOGDIR_PATH, PACKAGE_ROOT, per_config_name
from disco_snake.settings import JsonSettings
from shimeji.model_provider import OobaGenRequest

AI_DATA_DIR = PACKAGE_ROOT.parent.joinpath("data", "ai")
AI_LOG_DIR = LOGDIR_PATH
AI_LOG_FORMAT = LOG_FORMAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGES_DIR = AI_DATA_DIR.joinpath(per_config_name("images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class ResponseMode(str, Enum):
    """Configures how the bot will trigger responses in a channel/server"""

    NoRespond = "no"  # Completely ignore all messages from this channel
    Mentioned = "mentions"  # Only respond to messages from this channel with a mention
    IdleAuto = "idleauto"  # enable idle messaging
    FullAuto = "fullauto"  # Respond to mentions and allow auto-triggering
    Unlimited = "unlimited"  # Respond to mentions and auto-trigger with no cooldown. Use with caution.


class BotMode(str, Enum):
    """Configures how the bot will handle other bots in a channel/server"""

    Strip = "strip"  # Ignore all other bots entirely, strip them from the context
    Ignore = "ignore"  # Won't be triggered by bot messages but can still see them
    Siblings = "siblings"  # Will strip all bots except for listed siblings from the context
    All = "all"  # Will not strip any bots from the context (not recommended)


class NamedSnowflake(BaseModel):
    """A reference to a Discord object, with name and note for config file clarity"""

    id: int = Field(...)
    name: str = Field("")  # not actually used, just here so it can be in config
    note: str | None = None


class SnowflakeList(RootModel):
    """A list of NamedSnowflake objects. Used for storing lists of users and roles."""

    root: list[NamedSnowflake]

    def __iter__(self) -> Iterator[NamedSnowflake]:  # type: ignore
        return self.root.__iter__()

    def __getitem__(self, key) -> NamedSnowflake:
        return self.root.__getitem__(key)

    @property
    def ids(self) -> list[int]:
        return [x.id for x in self.root]

    def get_id(self, id: int) -> NamedSnowflake | None:
        for item in self.root:
            if item.id == id:
                return item
        return None


class PermissionList(BaseModel):
    """A list of users and roles used for permission gating"""

    users: Annotated[SnowflakeList, Field(default_factory=SnowflakeList)]
    roles: Annotated[SnowflakeList, Field(default_factory=SnowflakeList)]

    @property
    def user_ids(self) -> list[int]:
        return [x.id for x in self.users]

    @property
    def role_ids(self) -> list[int]:
        return [x.id for x in self.roles]


class ChannelSettings(NamedSnowflake):
    """Settings for a specific channel in a guild. Overrides guild settings."""

    respond: ResponseMode | None = None
    bot_action: BotMode | None = None
    imagen: bool = True
    idle_enable: bool = False
    idle_interval: int = Field(300)


class GuildSettings(NamedSnowflake):
    """Settings for a specific guild."""

    enabled: bool = True
    imagen: bool = True
    respond: ResponseMode = Field(ResponseMode.Mentioned)
    bot_action: BotMode = Field(BotMode.Siblings)
    mention_role: int | None = None
    idle_enable: bool = False
    idle_interval: int = Field(300)
    channels: list[ChannelSettings] = Field([])

    def channel_enabled(self, channel_id: int) -> bool:
        """
        Returns whether the bot should respond to messages in this channel,
        based on the guild's default setting and the channel's settings.
        """
        if self.enabled is False:
            return False  # guild is disabled, don't respond
        if channel_id in [x.id for x in self.channels if x.respond != ResponseMode.NoRespond]:
            return True  # channel is explicitly enabled
        return self.respond != ResponseMode.NoRespond  # guild default

    def channel_respond_mode(self, channel_id: int) -> ResponseMode:
        respond_mode = next((x.respond for x in self.channels if x.id == channel_id), self.respond)
        return respond_mode if respond_mode is not None else self.respond

    def channel_bot_mode(self, channel_id: int) -> BotMode:
        bot_mode = next((x.bot_action for x in self.channels if x.id == channel_id), self.bot_action)
        return bot_mode if bot_mode is not None else self.bot_action

    def channel_imagen(self, channel_id: int) -> bool:
        imagen_enable = next((x.imagen for x in self.channels if x.id == channel_id), self.imagen)
        return imagen_enable if imagen_enable is not None else self.imagen

    def channel_idle_mode(self, channel_id: int) -> tuple[bool, int]:
        return next(
            ((x.idle_enable, x.idle_interval) for x in self.channels if x.id == channel_id),
            (self.idle_enable, self.idle_interval),
        )


class GuildSettingsList(RootModel):
    """A list of guild settings. Used to store settings for multiple guilds."""

    root: list[GuildSettings]

    def __iter__(self) -> Iterator[GuildSettings]:  # type: ignore # pylance doesn't like RootModels
        return iter(self.root)

    def __getitem__(self, key) -> GuildSettings:
        return self.root[key]

    @property
    def guild_ids(self) -> list[int]:
        return [x.id for x in self]

    def get_id(self, guild_id: int) -> GuildSettings | None:
        for guild in self.root:
            if guild.id == guild_id:
                return guild
        return None


# configuration dataclasses
class BotParameters(BaseModel):
    autoresponse: bool = False
    idle_enable: bool = False
    force_lowercase: bool = False
    nicknames: list[str] = Field(default_factory=list)
    nicknames_quiet: list[str] = Field(default_factory=list)
    context_size: int = -1
    context_messages: int = 100
    logging_channel_id: int | None = None
    debug: bool = False
    memory_enable: bool = False
    max_retries: int = 3
    ctxbreak_restrict: bool = True
    ctxbreak: Annotated[PermissionList, Field(default_factory=PermissionList)]
    guilds: Annotated[GuildSettingsList, Field(default_factory=GuildSettingsList)]
    dm_users: Annotated[SnowflakeList, Field(default_factory=SnowflakeList)]
    siblings: Annotated[SnowflakeList, Field(default_factory=SnowflakeList)]

    @property
    def guild_ids(self) -> list[int]:
        return self.guilds.guild_ids

    @property
    def sibling_ids(self) -> list[int]:
        return self.siblings.ids

    @property
    def dm_user_ids(self) -> list[int]:
        return self.dm_users.ids

    def get_guild_settings(self, guild_id: int) -> GuildSettings | None:
        return self.guilds.get_id(guild_id)

    def get_idle_channels(self) -> list[int]:
        return [
            channel.id
            for guild in self.guilds
            if guild.idle_enable
            for channel in guild.channels
            if channel.idle_enable
        ]


class ChatMessage(BaseModel):
    role: str = Field("user")
    content: str = Field(...)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: str | list[str]) -> str:
        if isinstance(v, list):
            return "\n".join(v)
        return v

    def template_dump(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def is_chat_msg(obj: object) -> bool:
    if not isinstance(obj, ChatMessage) and not isinstance(obj, dict):
        return False
    if isinstance(obj, ChatMessage):
        return True
    if isinstance(obj, dict) and "role" in obj and "content" in obj:
        return True
    return False


def is_chat_msg_list(obj: object) -> bool:
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    if isinstance(obj[0], ChatMessage):
        return True
    if isinstance(obj[0], dict) and "role" in obj[0] and "content" in obj[0]:
        return True
    return False


class PromptElement(BaseModel):
    prompt: list[ChatMessage] | ChatMessage | str | None = None
    prefix: list[ChatMessage] | ChatMessage | None = None
    suffix: list[ChatMessage] | ChatMessage | None = None
    prompt_path: PathLike[str] | None = None

    _prompt_str: str | None = None  # cache for prompt string if file loaded
    _prompt_mtime: float | None = None  # cache for prompt file mtime

    @cached_property
    def _prompt_path(self) -> Path | None:
        if self.prompt_path is not None:
            return AI_DATA_DIR.joinpath(self.prompt_path)
        return None

    def _load_prompt_file(self) -> None:
        if self._prompt_path is not None:
            contents = self._prompt_path.read_text(encoding="utf-8")
            self._prompt_str = contents
            self._prompt_mtime = self._prompt_path.stat().st_mtime

    @property
    def full(self) -> str | list[dict[str, str]]:
        if self._prompt_path is not None:
            prompt_mtime = self._prompt_path.stat().st_mtime
            if self._prompt_mtime is None or prompt_mtime > self._prompt_mtime:
                logger.debug(f"Prompt file {self._prompt_path} has changed, reloading")
                self._load_prompt_file()
            if self._prompt_str is None:
                logger.warning(f"Prompt file {self._prompt_path} does not exist or is not a file")
            return [{"role": "system", "content": self._prompt_str}]

        if isinstance(self.prompt, str) or self.prompt is None:
            return self.prompt  # short-circuit for simple string/null prompts

        if isinstance(self.prompt, list):
            messages = self.prompt.copy()
        elif isinstance(self.prompt, ChatMessage):
            messages = [self.prompt]
        else:
            messages = []

        if self.prefix is not None:
            if isinstance(self.prefix, list):
                messages = self.prefix.copy() + messages
            else:
                messages = [self.prefix] + messages

        if self.suffix is not None:
            if isinstance(self.suffix, list):
                messages.extend(self.suffix)
            else:
                messages.append(self.suffix)  # type: ignore

        if not messages:
            # no messages collected
            return []

        # convert to list of dicts
        return [message.model_dump() for message in messages]


class Prompt(BaseModel):
    disco_mode: bool = False
    thinking: bool = False
    with_date: bool = False
    add_generation_prompt: bool = True

    system: PromptElement = Field(...)
    character: PromptElement = Field(...)

    think_start: str = Field("<think>")
    think_end: str = Field("</think>")
    template_file: str = Field("chat_template.j2")

    @property
    def template_str(self) -> str | None:
        tpl_path = AI_DATA_DIR.joinpath(self.template_file)
        if tpl_path.is_file():
            return tpl_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Template file {tpl_path} does not exist or is not a file")


class GradioConfig(BaseModel):
    enabled: bool = False
    bind_host: str = "127.0.0.1"
    bind_port: int = 7863
    enable_queue: bool = True
    width: str = "100%"
    theme: str | None = None
    root_path: str | None = None


class WebuiConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 7850
    secret: str = Field(...)
    title: str | None = Field(None, exclude=True)  # n.b. this is set dynamically, not in config


class LMApiConfig(BaseModel):
    endpoint: str
    api_key: Annotated[str | None, None]
    auth_header: Annotated[str, Field("X-Api-Key")]
    provider: str = "ooba"
    modeltype: str
    gensettings: OobaGenRequest
    api_v2: bool = False
    username: str | None = None
    password: str | None = None


class VisionConfig(BaseModel):
    enabled: bool = False  # whether to caption images or not
    host: str = "http://localhost:7862"  # host for vision api
    route: str = "/api/v1/caption"  # route for vision api
    token: str | None = None  # bearer token for api
    background: bool = False  # whether to caption images proactively in the background
    channel_ttl: int = 90  # monitor channel for this many seconds after last response


class AiSettings(JsonSettings):
    name: str
    prompt: Prompt
    params: BotParameters
    model_provider: LMApiConfig
    gradio: GradioConfig
    vision: VisionConfig | None = None
    analysis_header: str = "-# Analysis:"
    empty_react: str = "🤷‍♀️"
    strip: list[str] = Field([])
    bad_words: list[str] = Field([])

    model_config = SettingsConfigDict(
        json_file=[
            AI_DATA_DIR.joinpath("config.json"),
            AI_DATA_DIR.joinpath(per_config_name("config.json")),
        ],
        json_file_encoding="utf-8",
        nested_model_default_partial_update=True,
    )


def get_ai_settings() -> AiSettings:
    return AiSettings()  # type: ignore


## Imagen settings
class ImagenParams(BaseModel):
    enabled: bool
    api_host: str
    timezone: str
    chartype: str = "girl"


class ImagenApiParams(BaseModel):
    steps: int = 21
    cfg_scale: float = 7.5
    seed: int = -1
    default_width: int = 576
    default_height: int = 768
    sampler_name: str = "DPM++ 2M Karras"
    enable_hr: bool = False
    hr_steps: int = 7
    hr_denoise: float = 0.62
    hr_scale: float = 1.5
    hr_upscaler: str = "Latent"
    checkpoint: str | None = None
    vae: str | None = None
    clip_skip: int = 2
    overrides: dict | None = None
    alwayson_scripts: dict | None = None

    def get_request(self, prompt: str, negative: str, width: int = -1, height: int = -1):
        request_obj = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "seed_enable_extras": False,
            "seed_resize_from_h": 0,
            "seed_resize_from_w": 0,
            "width": width if width > 0 else self.default_width,
            "height": height if height > 0 else self.default_height,
            "batch_size": 1,
            "n_iter": 1,
            "send_images": True,
            "save_images": True,
            "sampler_name": self.sampler_name,
            "enable_hr": int(self.enable_hr),
            "hr_scale": self.hr_scale,
            "hr_second_pass_steps": self.hr_steps,
            "denoising_strength": self.hr_denoise,
            "hr_upscaler": self.hr_upscaler,
        }
        # set alwayson scripts if present
        if self.alwayson_scripts is not None:
            request_obj["alwayson_scripts"] = self.alwayson_scripts

        # copy the overrides dict so we don't modify the original
        overrides = self.overrides.copy() if self.overrides is not None else {}

        # set the checkpoint and VAE if provided
        if self.checkpoint is not None:
            overrides["sd_model_checkpoint"] = self.checkpoint
        if self.vae is not None:
            overrides["sd_vae"] = self.vae
        # set clip skip
        overrides["CLIP_stop_at_last_layers"] = self.clip_skip

        # if we have overrides, set them in the request object
        if len(overrides.keys()) > 0:
            request_obj["override_settings"] = overrides
            request_obj["override_settings_restore_afterwards"] = True

        # return the request object
        return request_obj


class ImagenLMPrompt(BaseModel):
    tags: list[str] = Field(...)
    template: list[str] = Field(...)
    gensettings: OobaGenRequest = Field(...)

    @property
    def tag_string(self) -> str:
        return ", ".join(self.tags)

    @property
    def default_prompt(self) -> str:
        prompt = self.gensettings.prompt
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        prompt = prompt.strip()
        return prompt if len(prompt) > 0 else "a cute girl looking out her apartment window"

    def wrap_prompt(self, user_message: str | None = None) -> str:
        if user_message is None or len(user_message) == 0:
            user_message = self.default_prompt
        return "\n".join(self.template).format(prompt_tags=self.tag_string, user_message=user_message.strip())

    def clean_tags(self, prompt: str) -> str:
        return prompt.replace(f"{self.tag_string},", "").strip()

    def get_request(self, prompt: str | None = None) -> OobaGenRequest:
        gensettings = deepcopy(self.gensettings)
        if prompt is not None and prompt != "":
            gensettings.prompt = prompt
        return OobaGenRequest.model_validate(gensettings)


class ImagenSDPrompt(BaseModel):
    lm_weight: float = 1.15
    tag_sep: str = ","
    word_sep: str = " "
    cleanup_desc: bool = True
    leading: list[str] = Field(...)
    trailing: list[str] = Field(...)
    negative: list[str] = Field(...)
    banned_tags: list[str] = Field(...)

    def wrap_prompt(self, prompt: str | list[str]) -> str:
        prompt = prompt if isinstance(prompt, list) else [prompt]
        prompt = self.tag_sep.join(self.get_leading(join=False) + prompt + self.get_trailing(join=False))  # type: ignore
        if self.word_sep != " ":
            prompt = prompt.replace(f",{self.word_sep}", self.tag_sep)
        return prompt

    def get_negative(self, join: bool = True) -> str | list[str]:
        if self.word_sep != " ":
            negative = [x.replace(" ", self.word_sep) for x in self.negative]
        else:
            negative = self.negative
        return self.tag_sep.join(negative) if join else negative

    def get_leading(self, join: bool = True) -> str | list[str]:
        if self.word_sep != " ":
            leading = [x.replace(" ", self.word_sep) for x in self.leading]
        else:
            leading = self.leading
        return self.tag_sep.join(leading) if join else leading

    def get_trailing(self, join: bool = True) -> str | list[str]:
        if self.word_sep != " ":
            trailing = [x.replace(" ", self.word_sep) for x in self.trailing]
        else:
            trailing = self.trailing
        return self.tag_sep.join(trailing) if join else trailing


class ImagenSettings(JsonSettings):
    params: ImagenParams = Field(...)
    api_params: ImagenApiParams = Field(...)
    lm_prompt: ImagenLMPrompt = Field(...)
    sd_prompt: ImagenSDPrompt = Field(...)

    model_config = SettingsConfigDict(
        json_file=[
            AI_DATA_DIR.joinpath("imagen.json"),
            AI_DATA_DIR.joinpath(per_config_name("imagen.json")),
        ],
        json_file_encoding="utf-8",
    )


def get_imagen_settings() -> ImagenSettings:
    return ImagenSettings()  # type: ignore
