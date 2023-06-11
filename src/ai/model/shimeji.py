from typing import Optional

from pydantic import BaseModel
from shimeji.model_provider import (
    EnmaModel,
    ModelGenArgs,
    ModelGenRequest,
    ModelSampleArgs,
    OobaModel,
    SukimaModel,
)
from transformers import PreTrainedTokenizerFast

from ai.settings import LMApiConfig


class ModelGenSettings(BaseModel):
    gen_args: ModelGenArgs
    sample_args: ModelSampleArgs
    model: Optional[str] = None

    def __post_init__(self):
        self.gen_args = ModelGenArgs(**self.gen_args)
        self.sample_args = ModelSampleArgs(**self.sample_args)


def get_sukima_model(cfg: LMApiConfig, tokenizer: PreTrainedTokenizerFast) -> SukimaModel:
    # load model provider gen_args into basemodel
    gen_settings = ModelGenSettings(**cfg.gensettings)

    request = ModelGenRequest(
        model=gen_settings.model,
        prompt="",
        sample_args=gen_settings.sample_args,
        gen_args=gen_settings.gen_args,
    )

    return SukimaModel(
        endpoint_url=cfg.endpoint,
        username=cfg.username,
        password=cfg.password,
        args=request,
        tokenizer=tokenizer,
    )


def get_enma_model(cfg: LMApiConfig, tokenizer: PreTrainedTokenizerFast) -> EnmaModel:
    # load model provider gen_args into basemodel
    gen_settings = ModelGenSettings(**cfg.gensettings)

    request = ModelGenRequest(
        model=gen_settings.model,
        prompt="",
        sample_args=gen_settings.sample_args,
        gen_args=gen_settings.gen_args,
    )

    return EnmaModel(
        endpoint_url=cfg.endpoint,
        args=request,
        tokenizer=tokenizer,
    )


def get_ooba_model(cfg: LMApiConfig, tokenizer: PreTrainedTokenizerFast) -> OobaModel:
    # load model provider gen_args into basemodel

    return OobaModel(
        endpoint_url=cfg.endpoint,
        args=cfg.gensettings,
        tokenizer=tokenizer,
    )
