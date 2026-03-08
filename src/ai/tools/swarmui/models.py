import logging
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, ClassVar

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from ai.tools.common import TOOLS_DATA_DIR
from disco_snake import per_config_name
from disco_snake.settings import JsonSettings

logger = logging.getLogger(__name__)

SWARM_CONFIG_PATHS = [TOOLS_DATA_DIR.joinpath("swarmui.json")]
_instance_config_path = TOOLS_DATA_DIR.joinpath(per_config_name("swarmui.json"))
if _instance_config_path.is_file() and _instance_config_path not in SWARM_CONFIG_PATHS:
    SWARM_CONFIG_PATHS.append(_instance_config_path)

SWARM_DATA_DIR = TOOLS_DATA_DIR.joinpath("swarmui")
SWARM_IMAGES_DIR = SWARM_DATA_DIR.joinpath("images")
if not SWARM_IMAGES_DIR.is_dir():
    SWARM_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

logger.debug(f"SwarmUI data directory: {SWARM_DATA_DIR}")
logger.debug(f"SwarmUI images directory: {SWARM_IMAGES_DIR}")
logger.debug(f"SwarmUI config paths: {SWARM_CONFIG_PATHS}")


class SwarmUIError(Exception):
    """Raised when the SwarmUI API returns an error response."""

    def __init__(
        self,
        message: str,
        error_id: str | None = None,
        status_code: int | None = None,
    ):
        """Initialize the error with SwarmUI context."""
        super().__init__(message)
        self.error_id = error_id
        self.status_code = status_code


class SwarmUIBaseModel(BaseModel):
    """Base model with relaxed parsing for SwarmUI payloads."""

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        use_enum_values=True,
    )


class SwarmUIResponse(SwarmUIBaseModel):
    """Base response type that carries SwarmUI error data."""

    error: str | None = None
    error_id: str | None = Field(None)

    def raise_for_error(self):
        """Raise SwarmUIError if the response contains an error."""
        if self.error:
            raise SwarmUIError(message=self.error, error_id=self.error_id)


class SwarmSessionInfo(SwarmUIResponse):
    """Response payload for GetNewSession."""

    session_id: str | None = None
    user_id: str | None = None
    output_append_user: bool | None = None
    version: str | None = None
    server_id: str | None = None
    permissions: list[str] | None = None

    @field_validator("permissions", mode="after")
    def validate_permissions(cls, value):
        return sorted(value) if value else []


class SwarmStatusCounters(SwarmUIBaseModel):
    """Status counters for current generation activity."""

    waiting_gens: int
    loading_models: int
    waiting_backends: int
    live_gens: int


class SwarmBackendStatus(SwarmUIBaseModel):
    """Backend status block from GetCurrentStatus."""

    status: str
    class_name: str = Field(..., alias="class")
    message: str
    any_loading: bool


class SwarmUIStatus(SwarmUIResponse):
    """Response payload for GetCurrentStatus."""

    status: SwarmStatusCounters
    backend_status: SwarmBackendStatus
    supported_features: list[str]


class SwarmGenerationResponse(SwarmUIResponse):
    """Raw image generation response from the GenerateImage endpoint."""

    images: list[str] = Field(..., alias="images")


class AspectRatio(str, Enum):
    Default = "Default"
    Custom = "Custom"

    UltraWide = "21:9"
    Wide = "16:9"
    SemiWide = "8:5"
    FilmLandscape = "3:2"
    Landscape = "4:3"
    Square = "1:1"
    Portrait = "3:4"
    FilmPortrait = "2:3"
    SemiTall = "5:8"
    Tall = "9:16"
    UltraTall = "9:21"


class LoraSection(int, Enum):
    Global = 0
    Base = 4
    Refiner = 1
    Video = 2
    VideoSwap = 3


class SwarmT2IParams(SwarmUIBaseModel):
    """Common text-to-image parameters, with support for extra fields."""

    prompt: str | None = None
    negative_prompt: str | None = Field(None, alias="negativeprompt")

    model: str | None = None
    seed: int = -1
    steps: int = Field(30, ge=1, le=1000)
    cfg_scale: float = Field(1.0, alias="cfgscale", ge=1.0, le=20.0)

    aspect_ratio: AspectRatio = Field(AspectRatio.Square, alias="aspectratio")
    side_length: int | None = Field(None, alias="sidelength")
    width: int | None = None
    height: int | None = None

    sampler: str = "euler"
    scheduler: str = "sgm_uniform"
    automaticvae: bool | None = None

    # extra optional parameters
    enable_mahiron: bool | None = Field(None, alias="enablemahiron")
    pag_scale: float | None = Field(None, alias="perturbedattentionguidancescale")

    # loras
    loras: list[str] | None = None
    lora_weights: list[float] | None = Field(None, alias="loraweights")
    lora_section_confinement: list[LoraSection] | None = Field(None, alias="lorasectionconfinement")

    # refiner control
    refiner_control_percentage: float | None = Field(None, alias="refinercontrolpercentage")
    refiner_method: str | None = Field(None, alias="refinermethod")
    refiner_upscale: float | None = Field(None, alias="refinerupscale")
    refiner_upscale_method: str | None = Field(None, alias="refinerupscalemethod")
    refiner_model: str | None = Field(None, alias="refinermodel")
    refiner_vae: str | None = Field(None, alias="refinervae")
    refiner_steps: int | None = Field(None, alias="refinersteps")
    refiner_cfg_scale: float | None = Field(None, alias="refinercfgscale")

    presets: list[str] | None = None
    extra_metadata: dict[str, str] | None = None

    DEFAULT_ASPECT_RATIO: ClassVar[AspectRatio] = AspectRatio.Portrait

    @model_validator(mode="after")
    def validate_parameters(self):
        if self.cfg_scale == 1.0 and self.negative_prompt:
            logger.info("CFG Scale is 1.0, negative prompt will have no effect so will not be used")
            self.negative_prompt = None
        if self.aspect_ratio == AspectRatio.Custom:
            if not (self.width and self.height):
                raise ValueError("Width and height must be set for custom aspect ratios")
        if self.aspect_ratio == AspectRatio.Default:
            self.width, self.height = None, None
            self.aspect_ratio = self.DEFAULT_ASPECT_RATIO

        elif self.width or self.height:
            raise ValueError("Aspect ratio must be 'Custom' when width or height is set.")
        if self.loras:
            if not self.lora_weights:
                logger.warning("Lora weights not provided; defaulting to 1.0 for all loras.")
                self.lora_weights = [1.0] * len(self.loras)
            if len(self.loras) != len(self.lora_weights):
                raise ValueError("Loras and lora_weights must have the same length.")
            if not self.lora_section_confinement:
                logger.warning("Lora section confinement not provided; defaulting to Global for all loras.")
                self.lora_section_confinement = [LoraSection.Global] * len(self.loras)
            if len(self.loras) != len(self.lora_section_confinement):
                raise ValueError("Loras and lora_section_confinement must have the same length.")
        return self

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        use_enum_values=True,
    )


class SwarmGeneratedImage(SwarmUIBaseModel):
    """Image generation result for the GenerateImage endpoint.
    This is not the raw response, since the raw response only returns the paths to the image files.
    """

    server_path: str = Field(...)
    image_bytes: bytes = Field(...)
    params: SwarmT2IParams = Field(...)

    _pil_image: Image.Image | None = None

    @computed_field
    @property
    def filename(self) -> str:
        """Get the filename of the generated image."""
        return Path(self.server_path).name

    @computed_field
    @property
    def local_path(self) -> Path:
        """Get the default save path for the image."""
        return SWARM_DATA_DIR.joinpath("images", self.server_path)

    @property
    def image(self) -> Image.Image:
        """Get the generated image as a PIL Image, loading it if necessary."""
        if not self._pil_image:
            img = Image.open(self.local_path)
            self._pil_image = img
        return self._pil_image

    def save_file(self, overwrite: bool = False) -> None:
        """Save the generated image to a file."""

        if not self.local_path.parent.is_dir():
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.is_file() or overwrite:
            logger.debug(f"Saving {self.filename} to {self.local_path}.")
            self.local_path.write_bytes(self.image_bytes)
            self.local_path.with_suffix(".json").write_text(
                self.params.model_dump_json(exclude_none=True, indent=2)
            )
        else:
            logger.debug(f"File {self.local_path} already exists, skipping save.")

    @classmethod
    def load_file(cls, local_path: PathLike) -> "SwarmGeneratedImage":
        """Load the generated image from a file."""
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Image file {local_path} does not exist.")
        json_path = local_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Metadata file {json_path} does not exist.")

        server_path = local_path.relative_to(SWARM_IMAGES_DIR).as_posix()
        image_bytes = local_path.read_bytes()
        params = SwarmT2IParams.model_validate_json(json_path.read_text())
        return cls(server_path=server_path, image_bytes=image_bytes, params=params)


class SwarmUISettings(JsonSettings):
    """Configuration settings for SwarmUI API client."""

    base_url: str
    api_token: str | None = None
    verify_ssl: bool | None = None
    user_agent: str = "disco-snake/SwarmUIClient/1.0"

    default_timeout: Annotated[float, Field(90.0, ge=1.0, le=300.0)]
    default_params: SwarmT2IParams = Field(default_factory=SwarmT2IParams)

    prompt_prefix: Annotated[str, Field("")]
    prefix_tags: Annotated[list[str], Field([])]
    prompt_suffix: Annotated[str, Field("")]

    aspect_ratio_list: list[AspectRatio] = Field(
        default_factory=lambda: [x for x in AspectRatio],
        description="List of permitted aspect ratios for image generation. Default is all.",
    )

    model_config = SettingsConfigDict(
        json_file=SWARM_CONFIG_PATHS,
        json_file_encoding="utf-8",
        nested_model_default_partial_update=True,
        extra="forbid",
    )

    @field_validator("base_url", mode="after")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        """Normalize the base URL to avoid trailing slashes."""
        return value.rstrip("/")

    def wrap_prompt(self, prompt: str) -> str:
        """Extend the given prompt with quality prefix/suffix if configured."""
        parts = []
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        parts.append(prompt)
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)
        return " ".join(parts)
