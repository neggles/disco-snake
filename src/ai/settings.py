import logging
import re
from copy import deepcopy
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import BaseModel, BaseSettings, Field
from shimeji.model_provider import OobaGenRequest

from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.settings import JsonConfig

AI_DATA_DIR = DATADIR_PATH.joinpath("ai")
AI_LOG_DIR = LOGDIR_PATH
AI_LOG_FORMAT = LOG_FORMAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AI_CFG_PATH = AI_DATA_DIR.joinpath("config.json")

IMAGEN_CFG_PATH = AI_DATA_DIR.joinpath("imagen.json")
IMAGES_DIR = AI_DATA_DIR.joinpath("images")


class ModelProviderConfig(BaseModel):
    endpoint: str = Field(...)
    type: str = Field("ooba")
    gensettings: OobaGenRequest = Field(...)
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)


# configuration dataclasses
class MemoryStoreConfig(BaseModel):
    database_uri: str
    model: str
    model_layer: int
    short_term_amount: int
    long_term_amount: int


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
    ctxbreak_users: List[int] = Field([])
    ctxbreak_roles: List[int] = Field([])


class VisionConfig(BaseModel):
    enabled: bool = False
    model_name: str = "clip"
    api_host: str = "http://localhost:7862"
    api_token: Optional[str] = None


class AiSettings(BaseSettings):
    name: str
    guilds: List[int]
    prompt: Union[str, List[str]]
    params: BotParameters
    model_provider: ModelProviderConfig
    bad_words: List[str] = Field([])
    memory_store: Optional[MemoryStoreConfig] = None
    vision: Optional[VisionConfig] = None

    class Config(JsonConfig):
        json_config_path = AI_CFG_PATH


@lru_cache(maxsize=1)
def get_ai_settings() -> AiSettings:
    settings = AiSettings()
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
    trailer: str = Field(...)
    gensettings: OobaGenRequest = Field(...)

    def __post_init__(self):
        self.re_subject = re.compile(r".* of")
        self.default_prompt = (
            self.gensettings["prompt"]
            if len(self.gensettings["prompt"]) > 0
            else "a cute girl looking out her apartment window"
        )

    def get_tags(self) -> str:
        return ", ".join(self.tags)

    def get_header(self) -> str:
        return "\n".join(self.header).replace("{prompt_tags}", self.get_tags())

    def get_trailer(self) -> str:
        return "\n\n" + self.trailer

    def prompt(self, user_message: str) -> str:
        if len(user_message) == 0:
            user_message = self.gensettings["prompt"]
        return f"{self.get_header()}{user_message}{self.get_trailer()}"

    def clean_tags(self, prompt: str) -> str:
        return prompt.replace(self.get_tags() + ", ", "")

    def get_request(self, prompt: Optional[str] = None) -> OobaGenRequest:
        gensettings = deepcopy(self.gensettings)
        if prompt is not None and prompt != "":
            gensettings["prompt"] = prompt
        return OobaGenRequest.parse_obj(gensettings)


class ImagenSDPrompt(BaseModel):
    leading: List[str]
    trailing: List[str]
    negative: List[str]
    lm_weight: float = 1.15

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
        json_config_path = IMAGEN_CFG_PATH


@lru_cache(maxsize=1)
def get_imagen_settings() -> ImagenSettings:
    settings = ImagenSettings()
    return settings