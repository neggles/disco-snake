import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from shimeji.model_provider import OobaGenRequest


class AsDictMixin:
    def asdict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ModelProviderConfig:
    endpoint: str
    gensettings: dict
    type: str = "sukima"
    username: Optional[str] = None
    password: Optional[str] = None


# configuration dataclasses
@dataclass
class MemoryStoreConfig:
    database_uri: str
    model: str
    model_layer: int
    short_term_amount: int
    long_term_amount: int


@dataclass
class BotParameters:
    conditional_response: bool
    idle_messaging: bool
    idle_messaging_interval: int
    nicknames: List[str]
    context_size: int
    logging_channel_id: int
    activity_channels: List[int]
    debug: bool
    memory_enable: bool
    max_retries: int
    ctxbreak_users: List[int]
    ctxbreak_roles: List[int]


@dataclass
class VisionConfig:
    api_host: str
    api_token: Optional[str] = None


@dataclass
class ChatbotConfig:
    name: str
    prompt: Union[str, List[str]]
    params: BotParameters
    memory_store: MemoryStoreConfig
    model_provider: ModelProviderConfig
    vision: Optional[VisionConfig]
    bad_words: List[str] = None


@dataclass
class ImagenParams:
    enabled: bool
    api_host: str
    timezone: str


@dataclass
class ImagenApiParams:
    steps: int = 25
    cfg_scale: float = 7.75
    seed: int = -1
    default_width: int = 576
    default_height: int = 768
    sampler_name: str = "Euler a"
    enable_hr: bool = False
    hr_steps: int = 0
    hr_denoise: float = 0.55
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


@dataclass
class ImagenLMPrompt:
    tags: List[str]
    header: List[str]
    trailer: str
    gensettings: Dict[str, Any]

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


@dataclass
class ImagenSDPrompt:
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


@dataclass
class ImagenConfig(AsDictMixin):
    params: ImagenParams
    api_params: ImagenApiParams
    lm_prompt: ImagenLMPrompt
    sd_prompt: ImagenSDPrompt
