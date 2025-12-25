"""llama.cpp llama-server client-specific code"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, RootModel


from .common import GenerationSettingsBase


class Samplers(str, Enum):
    """llama.cpp sampler types"""

    PENALTIES = "penalties"
    DRY = "dry"
    TOP_N_SIGMA = "top_n_sigma"
    TOP_K = "top_k"
    TYP_P = "typ_p"
    TOP_P = "top_p"
    MIN_P = "min_p"
    XTC = "xtc"
    TEMPERATURE = "temperature"


class LlamaSamplerOrder(RootModel[list[Samplers]]):
    """Order of samplers to apply in llama.cpp"""


DEFAULT_SAMPLER_ORDER = LlamaSamplerOrder(
    [
        Samplers.PENALTIES,
        Samplers.DRY,
        Samplers.TOP_N_SIGMA,
        Samplers.TOP_K,
        Samplers.TYP_P,
        Samplers.TOP_P,
        Samplers.MIN_P,
        Samplers.XTC,
        Samplers.TEMPERATURE,
    ]
)


class LlamaGenerationSettings(GenerationSettingsBase):
    """llama.cpp generation settings"""

    seed: int | None = None
    temperature: float | None = None
    dynatemp_range: float | int = 0
    dynatemp_exponent: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    typical_p: float = 1.0
    min_p: float = 0.0
    top_n_sigma: int | float = -1
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1

    repeat_last_n: int = 64
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1

    mirostat: float = 0.0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    max_tokens: int = Field(-1, alias="n_predict")
    n_keep: int = 0
    n_discard: int = 0

    ignore_eos: bool = False
    stream: bool = False
    n_probs: int = 0
    min_keep: int = 1
    chat_format: str | None = None
    reasoning_format: str | None = None
    reasoning_in_content: bool = False
    thinking_forced_open: bool = False
    timings_per_token: bool = False
    post_sampling_probs: bool = False

    speculative_n_max: int = Field(0, alias="speculative.n_max")
    speculative_n_min: int = Field(0, alias="speculative.n_min")
    speculative_p_min: float = Field(0.0, alias="speculative.p_min")

    samplers: LlamaSamplerOrder | None = DEFAULT_SAMPLER_ORDER
    lora: list[str] = Field(default_factory=list)  # loras to load, not doing this yet

    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        serialize_by_alias=True,
    )


class LlamaServerProps(BaseModel):
    """llama-server api /props response"""

    default_generation_settings: LlamaGenerationSettings
