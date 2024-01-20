import copy
import json
import logging
from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional

import aiohttp
import requests
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaTokenizerFast

from shimeji.tokenizers import Llama

logger = logging.getLogger(__name__.replace("shimeji", "ai"))


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


class OobaCompletionParams(BaseModel):
    prompt: str | list[str]
    echo: bool = False
    stream: bool = False
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0
    stop: Optional[str | list[str]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1


class OobaGenRequest(OobaGenParams, OobaCompletionParams):
    pass


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
        self.endpoint_url = endpoint_url
        self.debug = debug
        if not hasattr(self, "default_args"):
            self.default_args = default_args or kwargs.get("args", None)
        if self.default_args is None:
            raise ValueError("default args is required")

    @abstractmethod
    def generate(self, args: dict | BaseModel) -> str:
        """Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    async def generate_async(self, args: dict | BaseModel) -> str:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def should_respond(self, context: str, name: str) -> str:
        """Determine if the ModelProvider predicts that the name should respond to the given context.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :raises NotImplementedError: If the should_respond method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    def should_respond_async(self, context: str, name: str) -> str:
        """Determine if the ModelProvider predicts that the name should respond to the given context asynchronously.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :raises NotImplementedError: If the should_respond method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def response(self, context: str) -> str:
        """Generate a response from the ModelProvider's endpoint.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    def response_async(self, context: str) -> str:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")

    def tokenize(self, text: str, return_ids: bool = False) -> list[int | str]:
        """Tokenize a string and return it.

        :param text: The text to tokenize.
        :type text: str
        :return: The tokenized text.
        :raises NotImplementedError: If the tokenize method is not implemented.
        """
        raise NotImplementedError("Abstract base class was called ;_;")


class OobaModel(ModelProvider):
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(
        self,
        endpoint_url: str,
        default_args: OobaGenRequest,
        *,
        tokenizer: Optional[Tokenizer] = None,
        api_v2: bool = True,
        **kwargs,
    ):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        super().__init__(endpoint_url, default_args, **kwargs)
        self.default_args: OobaGenRequest
        self.api_v2 = api_v2

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer: Tokenizer = tokenizer
        elif isinstance(tokenizer, (str, PathLike)):
            self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer: LlamaTokenizerFast = Llama()

    @property
    def _api_path(self):
        if self.api_v2:
            return "v1/completions"
        return "api/v1/generate"

    @property
    def _result_key(self):
        if self.api_v2:
            return "choices"
        return "results"

    def generate(self, args: OobaGenRequest) -> str:
        payload = args.dict(exclude_none=True)
        if self.api_v2 and "max_tokens" not in payload:
            payload["max_tokens"] = payload.pop("max_new_tokens", 1024)

        if self.debug:
            logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        resp, err_resp = None, None
        try:
            resp = requests.post(
                f"{self.endpoint_url}/{self._api_path}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            resp.encoding = "utf-8"
        except Exception as e:
            raise e
        if resp.status_code == 200:
            return resp.json()[self._result_key][0]["text"]
        else:
            err_resp = resp.json() if hasattr(resp, "json") else None
            raise Exception(f"Could not generate text with text-generation-webui. Error: {err_resp}")

    async def generate_async(self, args: OobaGenRequest) -> str:
        payload = args.dict(exclude_none=True)
        if self.api_v2 and "max_tokens" not in payload:
            payload["max_tokens"] = payload.pop("max_new_tokens", 1024)

        if self.debug:
            logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        resp, err_resp = None, None
        try:
            async with aiohttp.ClientSession(base_url=self.endpoint_url) as session:
                async with session.post(f"/{self._api_path}", json=payload) as resp:
                    if resp.status == 200:
                        ret = await resp.json(encoding="utf-8")
                        return ret[self._result_key][0]["text"]
                    else:
                        resp.raise_for_status()
        except Exception as e:
            if resp is not None:
                err_resp = await resp.text(encoding="utf-8")
            raise Exception(f"Could not generate response. Error: {err_resp}") from e

    def should_respond(self, context, name: str, prefix: str = "") -> bool:
        """
        Determine if the Ooba endpoint predicts that the name should respond to the given context.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: OobaGenRequest = copy.deepcopy(self.default_args)
        args.prompt = f"{context}{prefix}"
        args.temperature = 0.7
        args.temperature_last = False
        args.top_p = 1.0
        args.min_p = 0.1
        args.top_k = 25
        args.repetition_penalty = 1.2
        args.repetition_penalty_range = 32
        args.do_sample = True
        args.max_tokens = 24
        args.min_length = 0
        response = self.generate(args)
        logger.debug(f"Response: {response.strip()}")
        if response.strip().startswith((name, prefix + name, prefix + " " + name)):
            return True
        else:
            return False

    async def should_respond_async(self, context, name: str, prefix: str = "") -> bool:
        """Determine if the Ooba endpoint predicts that the name should respond to the given context asynchronously.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: OobaGenRequest = copy.deepcopy(self.default_args)
        args.prompt = f"{context}{prefix}"
        args.temperature = 0.7
        args.temperature_last = False
        args.top_p = 1.0
        args.min_p = 0.1
        args.top_k = 25
        args.repetition_penalty = 1.2
        args.repetition_penalty_range = 32
        args.do_sample = True
        args.max_tokens = 24
        args.min_length = 0
        response = await self.generate_async(args)
        logger.debug(f"Response: {response.strip()}")
        if response.strip().startswith((name, prefix + name, prefix + ": " + name)):
            return True
        else:
            return False

    def response(self, context: str = None, gensettings: Optional[OobaGenRequest] = None) -> str:
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

        return self.generate(gensettings)

    async def response_async(self, context: str = None, gensettings: Optional[OobaGenRequest] = None) -> str:
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

        return await self.generate_async(gensettings)
