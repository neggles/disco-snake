import copy
import logging
from abc import ABC, abstractmethod
from os import PathLike
from typing import List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from transformers import AutoTokenizer, LlamaTokenizerFast

from shimeji.tokenizers import Llama

logger = logging.getLogger(__name__)


class ModelGenArgs(BaseModel):
    max_length: int = Field(256)
    max_time: Optional[float] = None
    min_length: Optional[int] = None
    eos_token_id: Optional[int] = None
    logprobs: Optional[int] = None
    best_of: Optional[int] = None

    # text-generation-webui specific
    seed: Optional[int] = None
    add_bos_token: Optional[bool] = None
    truncation_length: Optional[int] = None
    ban_eos_token: Optional[bool] = None
    skip_special_tokens: Optional[bool] = None
    stopping_strings: Optional[List[str]] = None


class ModelSampleArgs(BaseModel):
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    tfs: Optional[float] = None
    rep_p: Optional[float] = None
    rep_p_range: Optional[int] = None
    rep_p_slope: Optional[float] = None
    bad_words: Optional[List[str]] = None

    do_sample: Optional[bool] = None
    penalty_alpha: Optional[float] = None

    # text-generation-webui specific (mostly contrastive search)
    no_repeat_ngram_size: Optional[int] = None
    num_beams: Optional[int] = None
    penalty_alpha: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    encoder_rep_p: Optional[float] = None


class ModelGenRequest(BaseModel):
    model: Optional[str] = Field(None)
    prompt: str = Field("")
    softprompt: Optional[str] = Field(None)
    sample_args: ModelSampleArgs = Field(default_factory=ModelSampleArgs)
    gen_args: ModelGenArgs = Field(default_factory=ModelGenArgs)


class OobaGenRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 250
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    repetition_penalty: float = 1.1
    encoder_repetition_penalty: float = 1.0
    top_k: int = 40
    top_a: float = 0.0
    tfs: float = 1.0
    min_length: int = 0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    penalty_alpha: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    seed: int = -1
    add_bos_token: bool = True
    truncation_length: int = 2048
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    stopping_strings: List[str] = []

    @classmethod
    def from_generic(cls, req: ModelGenRequest) -> "OobaGenRequest":
        """
        Converts a ModelGenRequest to an OobaGenRequest since the format is quite a bit different.
        """
        ret = {
            "prompt": req.prompt,
            "max_new_tokens": req.gen_args.max_length,
            "do_sample": req.sample_args.do_sample,
            "temperature": req.sample_args.temp,
            "top_p": req.sample_args.top_p,
            "typical_p": req.sample_args.typical_p,
            "repetition_penalty": req.sample_args.rep_p,
            "encoder_repetition_penalty": req.sample_args.encoder_rep_p,
            "top_k": req.sample_args.top_k,
            "no_repeat_ngram_size": req.sample_args.no_repeat_ngram_size,
            "num_beams": req.sample_args.num_beams,
            "penalty_alpha": req.sample_args.penalty_alpha,
            "length_penalty": req.sample_args.length_penalty,
            "early_stopping": req.sample_args.early_stopping,
            "min_length": req.gen_args.min_length,
            "seed": req.gen_args.seed,
            "add_bos_token": req.gen_args.add_bos_token,
            "truncation_length": req.gen_args.truncation_length,
            "ban_eos_token": req.gen_args.ban_eos_token,
            "skip_special_tokens": req.gen_args.skip_special_tokens,
            "stopping_strings": req.gen_args.stopping_strings,
        }
        ret = OobaGenRequest.parse_obj(ret)
        return ret


class ModelProvider(ABC):
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(
        self,
        endpoint_url: str,
        default_args: Optional[dict | BaseModel] = None,
        **kwargs,
    ):
        self.endpoint_url = endpoint_url
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
        api_v1: bool = True,
        **kwargs,
    ):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        super().__init__(endpoint_url, default_args, **kwargs)
        self.default_args: OobaGenRequest
        self.api_v1 = api_v1

        if isinstance(tokenizer, Tokenizer):
            self.tokenizer: Tokenizer = tokenizer
        elif isinstance(tokenizer, (str, PathLike)):
            self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer: LlamaTokenizerFast = Llama()

    @property
    def _api_path(self):
        if self.api_v1:
            return "/api/v1/generate"
        return "/v1/completions"

    @property
    def _result_key(self):
        if self.api_v1:
            return "results"
        return "choices"

    def generate(self, args: Union[ModelGenRequest, OobaGenRequest]) -> str:
        """
        Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        if not isinstance(args, OobaGenRequest):
            args: OobaGenRequest = OobaGenRequest.from_generic(args)

        try:
            r = requests.post(
                f"{self.endpoint_url}{self._api_path}",
                headers={"Content-Type": "application/json"},
                json=args.dict(),
            )
            r.encoding = "utf-8"
        except Exception as e:
            raise e
        if r.status_code == 200:
            return r.json()[self._result_key][0]["text"]
        else:
            raise Exception(f"Could not generate text with text-generation-webui. Error: {r.json()}")

    async def generate_async(self, args: Union[ModelGenRequest, OobaGenRequest]) -> str:
        """
        Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        if not isinstance(args, OobaGenRequest):
            args: OobaGenRequest = OobaGenRequest.from_generic(args)
        try:
            async with aiohttp.ClientSession(base_url=self.endpoint_url) as session:
                async with session.post(f"/{self._api_path}", json=args.dict()) as resp:
                    if resp.status == 200:
                        ret = await resp.json(encoding="utf-8")
                        return ret[self._result_key][0]["text"]
                    else:
                        resp.raise_for_status()
        except Exception as e:
            raise Exception(f"Could not generate response. Error: {await resp.text(encoding='utf-8')}") from e

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
        args.temperature = 0.25
        args.top_p = 0.9
        args.top_k = 40
        args.repetition_penalty = 1.0
        args.do_sample = True
        args.max_new_tokens = 24
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
        args.temperature = 0.25
        args.top_p = 1.0
        args.top_k = 1
        args.num_beams = 3
        args.repetition_penalty = 1.2
        args.do_sample = True
        args.max_new_tokens = 10
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
