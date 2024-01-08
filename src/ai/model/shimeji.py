from typing import Optional

from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast

from ai.settings import LMApiConfig
from shimeji.model_provider import (
    ModelGenArgs,
    ModelSampleArgs,
    OobaModel,
)


class ModelGenSettings(BaseModel):
    gen_args: ModelGenArgs
    sample_args: ModelSampleArgs
    model: Optional[str] = None

    def __post_init__(self):
        self.gen_args = ModelGenArgs(**self.gen_args)
        self.sample_args = ModelSampleArgs(**self.sample_args)


def get_ooba_model(cfg: LMApiConfig, tokenizer: PreTrainedTokenizerFast) -> OobaModel:
    # load model provider gen_args into basemodel

    return OobaModel(
        endpoint_url=cfg.endpoint,
        default_args=cfg.gensettings,
        tokenizer=tokenizer,
        api_v2=False,
    )
