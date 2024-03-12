import copy
import json
import logging
from abc import ABC, abstractmethod
from functools import wraps
from os import PathLike
from typing import Any, Optional

import aiohttp
import requests
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from transformers import AutoTokenizer, BatchEncoding, LlamaTokenizerFast, PreTrainedTokenizerBase

from shimeji.tokenizers import Llama

logger = logging.getLogger(__name__.replace("shimeji", "ai"))


def default_sampler_priority() -> list[str]:
    return [
        "min_p",
        "temperature",
        "dynamic_temperature",
        "quadratic_sampling",
        "top_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
        "tfs",
        "top_a",
        "mirostat",
        "top_k",
    ]


class OobaGenParams(BaseModel):
    preset: str | None = Field(None, description="name of a file under presets/ to load settings from")
    min_p: float = 0
    dynamic_temperature: bool = False
    dynatemp_low: Optional[float] = None
    dynatemp_high: Optional[float] = None
    dynatemp_exponent: Optional[float] = None
    top_k: int = 0
    repetition_penalty: float = 1
    repetition_penalty_range: int = 1024
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    guidance_scale: float = 1
    negative_prompt: str = ""
    penalty_alpha: float = 0
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    temperature_last: bool = False
    do_sample: bool = True
    seed: int = -1
    encoder_repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    num_beams: int = 1
    length_penalty: float = 1
    early_stopping: bool = False
    truncation_length: int = 0
    max_tokens_second: int = 0
    custom_token_bans: str = ""
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    grammar_string: str = ""
    sampler_priority: list[str] = Field(default_factory=default_sampler_priority)


class OobaCompletionParams(BaseModel):
    prompt: str | list[str]
    echo: bool = False
    stream: bool = False
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Optional[str | list[str]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class OobaGenRequest(OobaGenParams, OobaCompletionParams):
    pass


class TabbyGenRequest(BaseModel):
    prompt: str | list[str]
    model: Optional[str] = None

    max_tokens: int = 128
    generate_window: Optional[int] = None

    temperature: Optional[float] = None
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    temp_exponent: Optional[float] = None
    temperature_last: Optional[bool] = None
    smoothing_factor: Optional[float] = None

    top_p: Optional[float] = None
    typical: Optional[float] = Field(None, alias="typical_p")
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    top_a: Optional[float] = None
    tfs: Optional[float] = None

    cfg_scale: Optional[float] = None
    negative_prompt: Optional[str | list[str]] = None

    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.15
    repetition_decay: Optional[int] = None
    penalty_range: Optional[int] = None

    token_healing: Optional[bool] = None
    add_bos_token: Optional[bool] = None
    ban_eos_token: Optional[bool] = None

    stop: Optional[str | list[str]] = None

    logprobs: Optional[int] = None
    logit_bias: Optional[dict[str, int]] = None

    # system-level
    echo: bool = False
    stream: bool = False


class ModelProvider(ABC):
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(
        self,
        endpoint_url: str,
        default_args: Optional[dict | BaseModel] = None,
        *,
        debug: bool = False,
        **kwargs,
    ):
        self.tokenizer: PreTrainedTokenizerBase
        self.endpoint_url = endpoint_url
        self.debug = debug
        if not hasattr(self, "default_args"):
            self.default_args = default_args or kwargs.get("args", None)
        if self.default_args is None:
            raise ValueError("default args is required")

    @abstractmethod
    def generate(self, args: dict | BaseModel, return_dict: bool = False) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    async def generate_async(self, args: dict | BaseModel, return_dict: bool = False) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def response(
        self,
        context: Optional[str],
        gensettings: Optional[dict | BaseModel] = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    def response_async(
        self,
        context: Optional[str],
        gensettings: Optional[dict | BaseModel] = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    @wraps(PreTrainedTokenizerBase.batch_encode_plus)
    def tokenize(self, text: str, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer.batch_encode_plus([text], *args, **kwargs)


class OobaModel(ModelProvider):
    def __init__(
        self,
        endpoint_url: str,
        default_args: OobaGenRequest,
        *,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        super().__init__(endpoint_url, default_args, **kwargs)
        self.default_args: OobaGenRequest

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer: Tokenizer = tokenizer
        elif isinstance(tokenizer, (str, PathLike)):
            self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer: LlamaTokenizerFast = Llama()

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "Content-Encoding": "utf-8",
            "Accept": "application/json",
            "Accept-Encoding": "utf-8",
        }

    def generate(self, args: OobaGenRequest, return_dict: bool = False) -> str:
        payload = args.dict(exclude_none=True)

        if self.debug:
            logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        resp, err_resp = None, None
        try:
            resp = requests.post(f"{self.endpoint_url}/v1/completions", headers=self.headers, json=payload)
            resp.encoding = "utf-8"
        except Exception as e:
            raise e
        if resp.status_code == 200:
            return resp.json() if return_dict else resp.json()["choices"][0]["text"]
        else:
            err_resp = resp.json() if hasattr(resp, "json") else None
            raise Exception(f"Could not generate text with text-generation-webui. Error: {err_resp}")

    async def generate_async(self, args: OobaGenRequest, return_dict: bool = False) -> str:
        payload = args.dict(exclude_none=True)

        if self.debug:
            logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        resp, err_resp = None, None
        try:
            async with aiohttp.ClientSession(base_url=self.endpoint_url, headers=self.headers) as session:
                async with session.post("/v1/completions", json=payload) as resp:
                    if resp.status == 200:
                        ret = await resp.json(encoding="utf-8")
                        return ret if return_dict else ret["choices"][0]["text"]
                    else:
                        resp.raise_for_status()
        except Exception as e:
            if resp is not None:
                err_resp = await resp.text(encoding="utf-8")
            raise Exception(f"Could not generate response. Error: {err_resp}") from e

    def response(
        self,
        context: Optional[str] = None,
        gensettings: Optional[TabbyGenRequest] = None,
        return_dict: bool = False,
    ) -> str:
        # error if neither argument is provided
        if gensettings is None and context is None:
            raise ValueError("I can't generate a response without a prompt!")
        # otherwise copy default gensettings if no gensettings are provided
        elif gensettings is None:
            gensettings = copy.deepcopy(self.default_args)

        # if context is provided, set the prompt to it
        if context is not None:
            gensettings.prompt = context
        # if we still have no prompt, error
        if gensettings.prompt is None or gensettings.prompt == "":
            raise ValueError("I can't generate a response without a prompt!")

        return self.generate(gensettings, return_dict=return_dict)

    async def response_async(
        self,
        context: Optional[str] = None,
        gensettings: Optional[TabbyGenRequest] = None,
        return_dict: bool = False,
    ) -> str:
        # error if neither argument is provided
        if gensettings is None and context is None:
            raise ValueError("I can't generate a response without a prompt!")
        # otherwise copy default gensettings if no gensettings are provided
        elif gensettings is None:
            gensettings = copy.deepcopy(self.default_args)

        # if context is provided, set the prompt to it
        if context is not None:
            gensettings.prompt = context
        # if we still have no prompt, error
        if gensettings.prompt is None or gensettings.prompt == "":
            raise ValueError("I can't generate a response without a prompt!")

        return await self.generate_async(gensettings, return_dict=return_dict)
