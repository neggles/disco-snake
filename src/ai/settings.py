import logging
import re
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from pandas import NA
from pydantic import BaseModel, BaseSettings, Field
from shimeji.model_provider import OobaGenRequest

from disco_snake import LOG_FORMAT, LOGDIR_PATH, PACKAGE_ROOT, per_config_name
from disco_snake.settings import JsonConfig

AI_DATA_DIR = PACKAGE_ROOT.parent.joinpath("data", "ai")
AI_LOG_DIR = LOGDIR_PATH
AI_LOG_FORMAT = LOG_FORMAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGEN_CFG_PATH = AI_DATA_DIR.joinpath(per_config_name("imagen.json"))
IMAGES_DIR = AI_DATA_DIR.joinpath(per_config_name("images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class ResponseMode(str, Enum):
    """Bitflag field for response modes"""

    NoRespond = "no"  # Completely ignore all messages from this channel
    Mentioned = "mentions"  # Only respond to messages from this channel with a mention
    IdleAuto = "idleauto"  # enable idle messaging
    FullAuto = "fullauto"  # Respond to mentions and allow auto-triggering


class BotMode(str, Enum):
    Strip = "strip"  # Ignore all other bots entirely, strip them from the context
    Ignore = "ignore"  # Won't be triggered by bot messages but can still see them
    Siblings = "siblings"  # Will strip all bots except for listed siblings from the context
    All = "all"  # Will not strip any bots from the context (not recommended)


class NamedSnowflake(BaseModel):
    id: int = Field(...)
    name: str = Field("")  # not actually used, just here so it can be in config
    note: Optional[str] = Field(None)


class PermissionList(BaseModel):
    users: List[NamedSnowflake] = Field([])
    roles: List[NamedSnowflake] = Field([])

    @property
    def user_ids(self) -> List[int]:
        return [x.id for x in self.users]

    @property
    def role_ids(self) -> List[int]:
        return [x.id for x in self.roles]


class ChannelSettings(NamedSnowflake):
    respond: Optional[ResponseMode] = Field(None)
    bot_action: Optional[BotMode] = Field(None)
    imagen: Optional[bool] = Field(True)


class GuildSettings(NamedSnowflake):
    enabled: bool = Field(True)
    respond: ResponseMode = Field(ResponseMode.Mentioned)
    bot_action: BotMode = Field(BotMode.Siblings)
    channels: List[ChannelSettings] = Field(default_factory=list)

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
        if channel_id in [x.id for x in self.channels if x.respond is not None]:
            return next(x.respond for x in self.channels if x.id == channel_id)
        return self.respond

    def channel_bot_mode(self, channel_id: int) -> BotMode:
        if channel_id in [x.id for x in self.channels if x.bot_action is not None]:
            return next(x.bot_action for x in self.channels if x.id == channel_id)
        return self.bot_action


# configuration dataclasses
class BotParameters(BaseModel):
    conditional_response: bool
    idle_messaging: bool
    idle_messaging_interval: int
    nicknames: List[str]
    context_size: int = 1024
    context_messages: int = 50
    logging_channel_id: Optional[int] = None
    activity_channels: List[int] = Field([])
    debug: bool = False
    memory_enable: bool = False
    max_retries: int = 3
    ctxbreak: PermissionList = Field(default_factory=PermissionList)
    guilds: List[GuildSettings] = Field([])
    dm_users: List[NamedSnowflake] = Field([])
    siblings: List[NamedSnowflake] = Field([])

    @property
    def guild_ids(self) -> List[int]:
        return [x.id for x in self.guilds]

    def get_guild_settings(self, guild_id: int) -> Optional[GuildSettings]:
        return next((x for x in self.guilds if x.id == guild_id), None)


class PromptElement(BaseModel):
    prefix: str = Field(...)
    prompt: Union[str, List[str]] = Field(...)
    suffix: str = Field(...)
    concat: str = Field("\n")

    @property
    def full(self) -> str:
        if isinstance(self.prompt, list):
            prompt = self.concat.join(self.prompt)
        else:
            prompt = self.prompt
        return self.concat.join([self.prefix, prompt, self.suffix]).strip()


class Prompt(BaseModel):
    instruct: Optional[bool] = Field(True)
    character: PromptElement = Field(...)
    system: PromptElement = Field(...)
    model: PromptElement = Field(...)
    prefix_bot: str = Field("\n")
    prefix_user: str = Field("\n")
    prefix_sep: str = Field("\n")


class GradioConfig(BaseModel):
    enabled: bool = False
    bind_host: str = "127.0.0.1"
    bind_port: int = 7863
    enable_queue: bool = True
    width: str = "100%"
    theme: Optional[str] = None


class LMApiConfig(BaseModel):
    endpoint: str = Field(...)
    provider: str = Field("ooba")
    modeltype: str = Field(...)
    gensettings: OobaGenRequest = Field(...)
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)


class VisionConfig(BaseModel):
    enabled: bool = False
    modeltype: str = "clip"
    api_host: str = "http://localhost:7862"
    api_token: Optional[str] = None


class AiSettings(BaseSettings):
    name: str
    prompt: Prompt
    gradio: GradioConfig
    params: BotParameters
    model_provider: LMApiConfig
    bad_words: List[str] = Field([])
    vision: Optional[VisionConfig] = None

    class Config(JsonConfig):
        json_config_path = AI_DATA_DIR.joinpath("config.json")


@lru_cache(maxsize=2)
def get_ai_settings(config_path: Optional[Path] = None) -> AiSettings:
    if config_path is None:
        config_path = AI_DATA_DIR.joinpath(per_config_name("config.json"))
    settings = AiSettings(json_config_path=config_path)
    return settings


## Imagen settings
class ImagenParams(BaseModel):
    enabled: bool
    api_host: str
    timezone: str


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
    checkpoint: Optional[str] = None
    vae: Optional[str] = None
    clip_skip: int = 2
    overrides: Optional[dict] = None

    def get_request(self, prompt: str, negative: str, width: int = -1, height: int = -1):
        request_obj = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
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
    tags: List[str] = Field(...)
    header: List[str] = Field(...)
    trailer: List[str] = Field(...)
    gensettings: OobaGenRequest = Field(...)

    def __post_init__(self):
        self.re_subject = re.compile(r".* of")
        self.default_prompt = (
            self.gensettings.prompt
            if len(self.gensettings.prompt) > 0
            else "a cute girl looking out her apartment window"
        )

    def get_tags(self) -> str:
        return ", ".join(self.tags)

    def get_header(self) -> str:
        return "\n".join(self.header).replace("{prompt_tags}", self.get_tags())

    def get_trailer(self) -> str:
        return "\n".join(self.trailer).replace("{prompt_tags}", self.get_tags())

    def prompt(self, user_message: str) -> str:
        if len(user_message) == 0:
            user_message = self.gensettings.prompt
        return f"{self.get_header()}{user_message}{self.get_trailer()}"

    def clean_tags(self, prompt: str) -> str:
        return prompt.replace(self.get_tags() + ", ", "")

    def get_request(self, prompt: Optional[str] = None) -> OobaGenRequest:
        gensettings = deepcopy(self.gensettings)
        if prompt is not None and prompt != "":
            gensettings.prompt = prompt
        return OobaGenRequest.parse_obj(gensettings)


class ImagenSDPrompt(BaseModel):
    lm_weight: float = 1.15
    leading: List[str] = Field(...)
    trailing: List[str] = Field(...)
    negative: List[str] = Field(...)
    banned_tags: List[str] = Field(...)

    def prompt(self, prompt: str) -> str:
        leading_tags = ", ".join(self.leading)
        trailing_tags = ", ".join(self.trailing)
        return f"{leading_tags}, {prompt}, {trailing_tags}"

    def negative_prompt(self) -> str:
        return ", ".join(self.negative)

    def get_leading(self) -> str:
        return ", ".join(self.leading)

    def get_trailing(self) -> str:
        return ", ".join(self.trailing)


class ImagenSettings(BaseSettings):
    params: ImagenParams = Field(...)
    api_params: ImagenApiParams = Field(...)
    lm_prompt: ImagenLMPrompt = Field(...)
    sd_prompt: ImagenSDPrompt = Field(...)

    class Config(JsonConfig):
        json_config_path = AI_DATA_DIR.joinpath("imagen.json")


@lru_cache(maxsize=2)
def get_imagen_settings(config_path: Optional[Path] = None) -> ImagenSettings:
    if config_path is None:
        config_path = AI_DATA_DIR.joinpath(per_config_name("imagen.json"))
    settings = ImagenSettings(json_config_path=config_path)
    return settings
