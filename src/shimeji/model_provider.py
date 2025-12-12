# pyright: reportIncompatibleMethodOverride=false
import json
import logging
import re
from abc import ABC, abstractmethod
from functools import wraps
from os import PathLike
from typing import Any, Optional
from warnings import warn

import aiohttp
import httpx
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from ai.types import ModelInfo

logger = logging.getLogger(__name__.replace("shimeji", "ai"))

re_ip = re.compile(r"^\w+://((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}/.*", re.I)


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
    min_p: float | None = None
    dynamic_temperature: bool = False
    dynatemp_low: float | None = None
    dynatemp_high: float | None = None
    dynatemp_exponent: float | None = None
    top_k: int = -1
    repetition_penalty: float | None = None
    repetition_penalty_range: int | None = None
    typical_p: float | None = None
    tfs: float | None = None
    top_a: float | None = None
    epsilon_cutoff: float | None = None
    eta_cutoff: float | None = None
    guidance_scale: float | None = None
    negative_prompt: str | list[str] | None = None
    penalty_alpha: float = 0
    mirostat_mode: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    temperature_last: bool = False
    do_sample: bool | None = None
    seed: int = -1
    encoder_repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    min_tokens: int | None = 0
    num_beams: int | None = None
    length_penalty: float | None = None
    early_stopping: bool | None = None
    truncation_length: int = 0
    ban_eos_token: bool | None = None
    add_bos_token: bool | None = None
    skip_special_tokens: bool = True
    grammar_string: str | None = None
    sampler_priority: list[str] | None = Field(None)


class OobaCompletionParams(BaseModel):
    prompt: str | list[str] = Field(default_factory=list)
    echo: bool = False
    stream: bool = False
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None


class OobaGenRequest(OobaGenParams, OobaCompletionParams):
    model_config = ConfigDict(
        extra="allow",
    )

    pass


class TabbyGenRequest(BaseModel):
    prompt: str | list[str]
    model: str | None = None

    max_tokens: int | None = None
    generate_window: int | None = None

    temperature: float | None = None
    min_temp: float | None = None
    max_temp: float | None = None
    temp_exponent: float | None = None
    temperature_last: bool | None = None
    smoothing_factor: float | None = None

    top_p: float | None = None
    typical: float | None = Field(None, alias="typical_p")
    min_p: float | None = None
    top_k: int | None = None
    top_a: float | None = None
    tfs: float | None = None

    cfg_scale: float | None = None
    negative_prompt: str | None = None
    speculative_ngram: bool | None = None

    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float | None = None
    penalty_range: int | None = None
    repetition_decay: int | None = None

    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_range: int | None = None
    dry_sequence_breakers: str | list[str] | None = None

    token_healing: bool | None = None
    add_bos_token: bool | None = None
    ban_eos_token: bool | None = None

    stop: str | list[str] | None = None

    logprobs: int | None = None
    logit_bias: dict[str, int] | None = None

    # system-level
    echo: bool = False
    stream: bool = False


class ModelProvider(ABC):
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(
        self,
        endpoint_url: str,
        default_args: dict | BaseModel | None = None,
        api_key: str | None = None,
        auth_header: str = "X-Api-Key",
        *,
        debug: bool = False,
        **kwargs,
    ):
        self.tokenizer: PreTrainedTokenizerBase
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.auth_header = auth_header
        self.debug = debug
        if not hasattr(self, "default_args"):
            self.default_args = default_args or kwargs.get("args", None)
        if self.default_args is None:
            raise ValueError("default args is required")

    @abstractmethod
    def generate(
        self,
        args: dict | BaseModel,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    async def generate_async(
        self,
        args: dict | BaseModel,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        warn("Using synchronous response method in async context.", RuntimeWarning, stacklevel=2)
        return self.generate(args, return_dict)

    @abstractmethod
    def response(
        self,
        context: Optional[str],
        gensettings: BaseModel | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    async def response_async(
        self,
        context: Optional[str],
        gensettings: dict | BaseModel | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        warn("Using synchronous response method in async context.", RuntimeWarning, stacklevel=2)
        return self.response(context, gensettings, return_dict)

    @wraps(PreTrainedTokenizerBase.batch_encode_plus)
    def tokenize(self, text: str, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer.batch_encode_plus([text], *args, **kwargs)


class OobaModel(ModelProvider):
    def __init__(
        self,
        endpoint_url: str,
        default_args: OobaGenRequest,
        api_key: str | None = None,
        auth_header: str = "X-Api-Key",
        ssl_verify: bool | None = None,
        *,
        tokenizer: PreTrainedTokenizerBase | PathLike | None = None,
        **kwargs,
    ):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        super().__init__(endpoint_url, default_args, api_key, auth_header, **kwargs)
        self.default_args: OobaGenRequest
        self.ssl_verify = ssl_verify
        if ssl_verify is None:
            self.ssl_verify = (
                False
                if "localhost" in endpoint_url or ".local" in endpoint_url or re_ip.match(endpoint_url)
                else True
            )

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer: PreTrainedTokenizerBase = tokenizer
        elif isinstance(tokenizer, (str, PathLike)):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            warn("No tokenizer provided; defaulting to Llama tokenizer.", UserWarning, stacklevel=2)
            self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

        self.client = httpx.Client(base_url=self.endpoint_url, headers=self.headers, verify=False)
        self._model_info: ModelInfo | None = None

    @property
    def model_info(self) -> ModelInfo:
        if self._model_info is None:
            info_response = self._fetch_model_info()
            self._model_info = ModelInfo.model_validate(info_response, strict=False)
        return self._model_info

    def _fetch_model_info(self) -> dict[str, Any]:
        resp = self.client.get("/v1/model", headers=self.headers)
        try:
            resp.raise_for_status()
            if resp.status_code == 200:
                return resp.json()
            else:
                msg = f"Could not retrieve model info from text-generation-webui. Error: {resp.text}"
                logger.warning(msg)
                raise RuntimeError(msg)
        except httpx.HTTPError as e:
            msg = f"Could not retrieve model info from text-generation-webui. Error: {resp.text}"
            logger.warning(msg)
            raise RuntimeError(msg) from e

    @property
    def headers(self):
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "utf-8",
            "Accept": "application/json",
            "Accept-Encoding": "utf-8",
        }
        if self.api_key is not None:
            headers[self.auth_header] = self.api_key
        return headers

    def generate(
        self,
        args: OobaGenRequest | TabbyGenRequest,
        return_dict: bool = False,
    ) -> str:
        payload = args.model_dump(exclude_none=True)

        if self.debug:
            logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        resp, err_resp = None, None
        try:
            resp = httpx.post(f"{self.endpoint_url}/v1/completions", headers=self.headers, json=payload)
            resp.encoding = "utf-8"
        except Exception as e:
            raise e
        if resp.status_code == 200:
            return resp.json() if return_dict else resp.json()["choices"][0]["text"]
        else:
            err_resp = resp.json() if hasattr(resp, "json") else None
            raise Exception(f"Could not generate text with text-generation-webui. Error: {err_resp}")

    async def generate_async(
        self,
        args: OobaGenRequest | TabbyGenRequest,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        payload = args.model_dump(exclude_none=True)

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
                        if resp is not None:
                            err_resp = await resp.text(encoding="utf-8")
                        resp.raise_for_status()
                        raise SystemError()  # raise_for_status will have raised an error
        except Exception as e:
            raise Exception(f"Could not generate response. Error: {err_resp}") from e

    def response(
        self,
        context: str | None = None,
        gensettings: TabbyGenRequest | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        # error if neither argument is provided
        if gensettings is None and context is None:
            raise ValueError("I can't generate a response without a prompt!")
        # otherwise copy default gensettings if no gensettings are provided
        elif gensettings is None:
            gensettings = TabbyGenRequest.model_validate(
                self.default_args.model_dump(exclude_none=True), strict=False
            )

        # if context is provided, set the prompt to it
        if context is not None:
            gensettings.prompt = context
        # if we still have no prompt, error
        if gensettings.prompt is None or gensettings.prompt == "":
            raise ValueError("I can't generate a response without a prompt!")

        return self.generate(gensettings, return_dict=return_dict)

    async def response_async(
        self,
        context: str | None = None,
        gensettings: TabbyGenRequest | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, Any]:
        # error if neither argument is provided
        if gensettings is None and context is None:
            raise ValueError("I can't generate a response without a prompt!")
        # otherwise copy default gensettings if no gensettings are provided
        elif gensettings is None:
            gensettings = TabbyGenRequest.model_validate(
                self.default_args.model_dump(exclude_none=True), strict=False
            )

        # if context is provided, set the prompt to it
        if context is not None:
            gensettings.prompt = context
        # if we still have no prompt, error
        if gensettings.prompt is None or gensettings.prompt == "":
            raise ValueError("I can't generate a response without a prompt!")

        return await self.generate_async(gensettings, return_dict=return_dict)
