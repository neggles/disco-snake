import copy
import json
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel

from shimeji.util import tokenizer


class DictJsonMixin:
    def asdict(self, *args, **kwargs) -> Dict[str, Any]:
        return self.dict(*args, **kwargs)

    def asjson(self, *args, **kwargs):
        return json.dumps(self.dict(*args, **kwargs))


class ModelGenArgs(BaseModel, DictJsonMixin):
    max_length: int
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


class ModelLogitBiasArgs(BaseModel, DictJsonMixin):
    id: int
    bias: float


class ModelPhraseBiasArgs(BaseModel, DictJsonMixin):
    sequences: List[str]
    bias: float
    ensure_sequence_finish: bool
    generate_once: bool


class ModelSampleArgs(BaseModel, DictJsonMixin):
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
    logit_biases: Optional[List[ModelLogitBiasArgs]] = None
    phrase_biases: Optional[List[ModelPhraseBiasArgs]] = None

    # enma-api specific
    do_sample: Optional[bool] = None
    penalty_alpha: Optional[float] = None
    num_return_sequences: Optional[int] = None
    stop_sequence: Optional[str] = None

    # text-generation-webui specific (mostly contrastive search)
    no_repeat_ngram_size: Optional[int] = None
    num_beams: Optional[int] = None
    penalty_alpha: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None


class ModelGenRequest(BaseModel, DictJsonMixin):
    model: Optional[str]
    prompt: str
    softprompt: Optional[str] = None
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs

    def __init__(__pydantic_self__, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if __pydantic_self__.sample_args is None:
            __pydantic_self__.sample_args = ModelSampleArgs()
        if __pydantic_self__.gen_args is None:
            __pydantic_self__.gen_args = ModelGenArgs()


class OobaGenRequest(BaseModel, DictJsonMixin):
    prompt: str
    max_new_tokens: int = 250
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.5
    typical_p: float = 1.0
    repetition_penalty: float = 1.125
    top_k: int = 40
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


class ModelSerializer(json.JSONEncoder):
    """Class to serialize ModelGenRequest to JSON."""

    def default(self, o):
        if hasattr(o, "asjson"):
            return o.asjson()
        return json.JSONEncoder.default(self, o)


class ModelProvider:
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(self, endpoint_url: str, **kwargs):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        self.endpoint_url = endpoint_url
        self.kwargs = kwargs
        if "args" not in kwargs:
            raise Exception("default args is required")
        self.default_args = self.kwargs.get("args", None)
        self.auth()

    def auth(self):
        """Authenticate with the ModelProvider's endpoint.

        :raises NotImplementedError: If the authentication method is not implemented.
        """
        raise NotImplementedError("auth method is required")

    def generate(self, args):
        """Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("generate method is required")

    async def generate_async(self, args):
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        raise NotImplementedError("generate method is required")

    async def hidden_async(self, model, text, layer):
        """Fetch a layer's hidden states from text.

        :param model: The model to extract hidden states from.
        :type model: str
        :param text: The text to use.
        :type text: str
        :param layer: The layer to fetch the hidden states from.
        :type layer: int
        """

        raise NotImplementedError("hidden_async method is required")

    async def image_label_async(self, model, url, labels):
        """Classify an image with labels (CLIP).

        :param model: The model to use for classification.
        :type model: str
        :param url: The image URL to use.
        :type url: str
        :param labels: The labels to use.
        :type labels: list
        """

        raise NotImplementedError("image_label_async method is required")

    def should_respond(self, context, name):
        """Determine if the ModelProvider predicts that the name should respond to the given context.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :raises NotImplementedError: If the should_respond method is not implemented.
        """
        raise NotImplementedError("should_respond method is required")

    def should_respond_async(self, context, name):
        """Determine if the ModelProvider predicts that the name should respond to the given context asynchronously.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :raises NotImplementedError: If the should_respond method is not implemented.
        """
        raise NotImplementedError("should_respond method is required")

    def response(self, context):
        """Generate a response from the ModelProvider's endpoint.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("response method is required")

    def response_async(self, context):
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :raises NotImplementedError: If the response method is not implemented.
        """
        raise NotImplementedError("response method is required")


class SukimaModel(ModelProvider):
    def __init__(self, endpoint_url: str, **kwargs):
        """Constructor for SukimaModel.

        :param endpoint_url: The URL for the Sukima endpoint.
        :type endpoint_url: str
        """
        self.session = aiohttp.ClientSession()
        self.client = requests.Session()
        super().__init__(endpoint_url, **kwargs)
        self.auth()

    def auth(self):
        """Authenticate with the Sukima endpoint.

        :raises Exception: If the authentication fails.
        """

        if "username" not in self.kwargs and "password" not in self.kwargs:
            raise Exception("username, password, and or token are not in kwargs")

        try:
            r = self.client.post(
                f"{self.endpoint_url}/api/v1/users/token",
                data={"username": self.kwargs["username"], "password": self.kwargs["password"]},
            )
            if r.status_code == 200:
                self.token = r.json()["access_token"]
            else:
                raise Exception(f"Could not authenticate with Sukima. Error: {r.text}")
        except Exception as e:
            raise e

    def conv_listobj_to_listdict(self, list_objects):
        """Convert the elements of a list to a dictionary for JSON compatability.

        :param list_objects: The list.
        :type list_objects: list
        :return: A list which has it's elements converted to dictionaries.
        :rtype: list
        """

        list_dict = []
        if list_objects:
            for object in list_objects:
                list_dict.append(vars(object))
            return list_dict
        else:
            return list_objects

    def generate(self, args: ModelGenRequest):
        """Generate a response from the Sukima endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :return: The response from the endpoint.
        :rtype: str
        :raises Exception: If the request fails.
        """

        argdict = {
            "model": args.model,
            "prompt": args.prompt,
            "sample_args": {
                "temp": args.sample_args.temp,
                "top_p": args.sample_args.top_p,
                "top_a": args.sample_args.top_a,
                "top_k": args.sample_args.top_k,
                "typical_p": args.sample_args.typical_p,
                "tfs": args.sample_args.tfs,
                "rep_p": args.sample_args.rep_p,
                "rep_p_range": args.sample_args.rep_p_range,
                "rep_p_slope": args.sample_args.rep_p_slope,
                "bad_words": args.sample_args.bad_words,
                "logit_biases": self.conv_listobj_to_listdict(args.sample_args.logit_biases),
            },
            "gen_args": {
                "max_length": args.gen_args.max_length,
                "max_time": args.gen_args.max_time,
                "min_length": args.gen_args.min_length,
                "eos_token_id": args.gen_args.eos_token_id,
                "logprobs": args.gen_args.logprobs,
                "best_of": args.gen_args.best_of,
            },
        }
        try:
            r = requests.post(
                f"{self.endpoint_url}/api/v1/models/generate",
                data=json.dumps(argdict),
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except Exception as e:
            raise e
        if r.status_code == 200:
            return r.json()["output"][len(argdict["prompt"]) :]
        else:
            raise Exception(f"Could not generate text with Sukima. Error: {r.json()}")

    async def generate_async(self, args: ModelGenRequest):
        """Generate a response from the Sukima endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :return: The response from the endpoint.
        :rtype: str
        :raises Exception: If the request fails.
        """

        argdict = {
            "model": args.model,
            "prompt": args.prompt,
            "sample_args": {
                "temp": args.sample_args.temp,
                "top_p": args.sample_args.top_p,
                "top_a": args.sample_args.top_a,
                "top_k": args.sample_args.top_k,
                "typical_p": args.sample_args.typical_p,
                "tfs": args.sample_args.tfs,
                "rep_p": args.sample_args.rep_p,
                "rep_p_range": args.sample_args.rep_p_range,
                "rep_p_slope": args.sample_args.rep_p_slope,
                "bad_words": args.sample_args.bad_words,
                "logit_biases": self.conv_listobj_to_listdict(args.sample_args.logit_biases),
            },
            "gen_args": {
                "max_length": args.gen_args.max_length,
                "max_time": args.gen_args.max_time,
                "min_length": args.gen_args.min_length,
                "eos_token_id": args.gen_args.eos_token_id,
                "logprobs": args.gen_args.logprobs,
                "best_of": args.gen_args.best_of,
            },
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.endpoint_url}/api/v1/models/generate",
                    json=argdict,
                    headers={"Authorization": f"Bearer {self.token}"},
                ) as resp:
                    if resp.status == 200:
                        js = await resp.json()
                        return js["output"][len(argdict["prompt"]) :]
                    else:
                        raise Exception(f"Could not generate response. Error: {await resp.text()}")
            except Exception as e:
                raise e

    async def hidden_async(self, model, text, layer):
        """Fetch a layer's hidden states from text.

        :param model: The model to extract hidden states from.
        :type model: str
        :param text: The text to use.
        :type text: str
        :param layer: The layer to fetch the hidden states from.
        :type layer: int
        """

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.endpoint_url}/api/v1/models/hidden",
                    json={"model": model, "prompt": text, "layers": [layer]},
                    headers={"Authorization": f"Bearer {self.token}"},
                ) as resp:
                    if resp.status == 200:
                        return (await resp.json())[f"{layer}"][0]
                    else:
                        raise Exception(f"Could not fetch hidden states. Error: {await resp.text()}")
            except Exception as e:
                raise e

    async def image_label_async(self, model, url, labels):
        """Classify an image with labels (CLIP).

        :param model: The model to use for classification.
        :type model: str
        :param url: The image URL to use.
        :type url: str
        :param labels: The labels to use.
        :type labels: list
        """

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.endpoint_url}/api/v1/models/classify",
                    json={"model": model, "prompt": url, "labels": labels},
                    headers={"Authorization": f"Bearer {self.token}"},
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        raise Exception(f"Could not classify image. Error: {await resp.text()}")
            except Exception as e:
                raise e

    def should_respond(self, context, name):
        """Determine if the Sukima endpoint predicts that the name should respond to the given context.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        phrase_bias = ModelPhraseBiasArgs
        phrase_bias.sequences = [name]
        phrase_bias.bias = 1.5
        phrase_bias.ensure_sequence_finish = True
        phrase_bias.generate_once = True

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.max_length = 10
        args.gen_args.eos_token_id = 25
        args.gen_args.best_of = None
        args.sample_args.temp = 0.25
        args.sample_args.rep_p = None
        args.sample_args.rep_p_range = None
        args.sample_args.rep_p_slope = None
        args.sample_args.phrase_biases = phrase_bias
        response = self.generate(args)
        if name in response:
            return True
        else:
            return False

    async def should_respond_async(self, context, name):
        """Determine if the Sukima endpoint predicts that the name should respond to the given context asynchronously.

        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """
        phrase_bias = ModelPhraseBiasArgs
        phrase_bias.sequences = [name]
        phrase_bias.bias = 1.5
        phrase_bias.ensure_sequence_finish = True
        phrase_bias.generate_once = True

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.max_length = 10
        args.gen_args.eos_token_id = 25
        args.gen_args.best_of = None
        args.sample_args.temp = 0.25
        args.sample_args.rep_p = None
        args.sample_args.rep_p_range = None
        args.sample_args.rep_p_slope = None
        args.sample_args.phrase_biases = phrase_bias
        response = await self.generate_async(args)
        if response.startswith(name):
            return True
        else:
            return False

    def response(self, context):
        """Generate a response from the Sukima endpoint.

        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.eos_token_id = 198
        args.gen_args.min_length = 1
        response = self.generate(args)
        return response

    async def response_async(self, context):
        """Generate a response from the Sukima endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.eos_token_id = 198
        args.gen_args.min_length = 1
        response = await self.generate_async(args)
        return response


class EnmaModel(ModelProvider):
    def __init__(self, endpoint_url: str, **kwargs):
        """Constructor for Enma_ModelProvider.
        :param endpoint_url: The URL for the Enma endpoint. (this is the completion endpoint on the gateway!)
        :type endpoint_url: str
        """
        self.session = aiohttp.ClientSession()
        self.client = requests.Session()
        super().__init__(endpoint_url, **kwargs)
        self.auth()

    def auth(self):
        """
        :drollwide:
        enma doesnt have authentication (at least on fab8e60) so this just returns true
        """
        return True

    def conv_listobj_to_listdict(self, list_objects):
        """Convert the elements of a list to a dictionary for JSON compatability.
        :param list_objects: The list.
        :type list_objects: list
        :return: A list which has it's elements converted to dictionaries.
        :rtype: list
        """

        list_dict = []
        if list_objects:
            for object in list_objects:
                list_dict.append(vars(object))
            return list_dict
        else:
            return list_objects

    def generate(self, args: ModelGenRequest):
        """Generate a response from the Enma endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :return: The response from the endpoint.
        :rtype: str
        :raises Exception: If the request fails.
        """
        argdict = {
            "engine": args.model,  # enma uses engine instead of model
            "prompt": args.prompt,
            "temperature": args.sample_args.temp,
            "top_p": args.sample_args.top_p,
            "top_k": args.sample_args.top_k,
            "repetition_penalty": args.sample_args.rep_p,
            "do_sample": args.sample_args.do_sample,
            "penalty_alpha": args.sample_args.penalty_alpha,
            "num_return_sequences": args.sample_args.num_return_sequences,
            "stop_sequence": args.sample_args.stop_sequence,
        }

        for arg in argdict.values():
            if arg is None:
                raise ValueError("Missing required argument: " + arg)
        try:
            r = requests.post(f"{self.endpoint_url}", data=json.dumps(argdict))
        except Exception as e:
            raise e
        if r.status_code == 200:
            return r.json()[0]["generated_text"][len(argdict["prompt"]) :]
        else:
            raise Exception(f"Could not generate text with Enma. Error: {r.json()}")

    async def generate_async(self, args: ModelGenRequest):
        """Generate a response from the Enma endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :return: The response from the endpoint.
        :rtype: str
        :raises Exception: If the request fails.
        """

        argdict = {
            "engine": args.model,  # enma uses engine instead of model
            "prompt": args.prompt,
            "temperature": args.sample_args.temp,
            "top_p": args.sample_args.top_p,
            "top_k": args.sample_args.top_k,
            "repetition_penalty": args.sample_args.rep_p,
            "do_sample": args.sample_args.do_sample,
            "penalty_alpha": args.sample_args.penalty_alpha,
            "num_return_sequences": args.sample_args.num_return_sequences,
            "stop_sequence": args.sample_args.stop_sequence,
        }
        for arg in argdict.values():
            if arg is None:
                raise ValueError("Missing required argument: " + arg)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.endpoint_url}", json=argdict) as resp:
                    if resp.status == 200:
                        js = await resp.json()
                        return js[0]["generated_text"][len(argdict["prompt"]) :]
                    else:
                        raise Exception(f"Could not generate response. Error: {await resp.text()}")
            except Exception as e:
                raise e

    def should_respond(self, context, name):
        """Determine if the Enma endpoint predicts that the name should respond to the given context.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.sample_args.temp = 0.25
        args.sample_args.top_p = 0.9
        args.sample_args.top_k = 40
        args.sample_args.rep_p = None
        args.sample_args.do_sample = True
        args.sample_args.penalty_alpha = None
        args.sample_args.num_return_sequences = 1  # i have no idea what these should be
        args.sample_args.stop_sequence = None
        response = self.generate(args)
        if name in response:
            return True
        else:
            return False

    async def should_respond_async(self, context, name):
        """Determine if the Enma endpoint predicts that the name should respond to the given context asynchronously.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.sample_args.temp = 0.25
        args.sample_args.top_p = 0.9
        args.sample_args.top_k = 40
        args.sample_args.rep_p = None
        args.sample_args.do_sample = True
        args.sample_args.penalty_alpha = None
        args.sample_args.num_return_sequences = 1  # i have no idea what these should be
        args.sample_args.stop_sequence = None
        response = await self.generate_async(args)
        if response.startswith(name):
            return True
        else:
            return False

    def response(self, context):
        """Generate a response from the Enma endpoint.
        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.eos_token_id = 198
        args.gen_args.min_length = 1
        response = self.generate(args)
        return response

    async def response_async(self, context):
        """Generate a response from the Enma endpoint asynchronously.
        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.eos_token_id = 198
        args.gen_args.min_length = 1
        response = await self.generate_async(args)
        return response


class TextSynthModel(ModelProvider):
    def __init__(self, endpoint_url: str = "https://api.textsynth.com", **kwargs):
        """Constructor for TextSynth_ModelProvider.

        :param endpoint_url: The URL for the TextSynth endpoint.
        :type endpoint_url: str
        :param token: The API token for the TextSynth endpoint.
        :type token: str
        """
        super().__init__(endpoint_url, **kwargs)
        self.auth()

    def auth(self):
        """Authenticate with the TextSynth endpoint.

        :raises Exception: If the authentication fails.
        """
        if "token" not in self.kwargs:
            raise Exception("token is not in kwargs")
        self.token = self.kwargs["token"]

    async def generate_async(self, args: ModelGenRequest) -> str:
        """Generate a response from the TextSynth endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :return: The response from the endpoint.
        :rtype: str
        :raises Exception: If the request fails.
        """
        model = args.model
        args = {
            "prompt": args.prompt,
            "max_tokens": args.gen_args.max_length,
            "temperature": args.sample_args.temp,
            "top_p": args.sample_args.top_p,
            "top_k": args.sample_args.top_k,
            "stop": tokenizer.decode(args.gen_args.eos_token_id),
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.endpoint_url}/v1/engines/{model}/completions",
                    json=args,
                    headers={"Authorization": f"Bearer {self.token}"},
                ) as resp:
                    if resp.status == 200:
                        js = await resp.json()
                        return js["text"]
                    else:
                        raise Exception(f"Could not generate response. Error: {resp.text()}")
            except Exception as e:
                raise e

    async def should_respond_async(self, context, name):
        """Determine if the TextSynth endpoint predicts that the name should respond to the given context asynchronously.

        :param context: The context to use.
        :type context: str
        :param name:
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.max_length = 10
        args.sample_args.temp = 0.25
        response = await self.generate_async(args)
        if response.startswith(name):
            return True
        else:
            return False

    async def response_async(self, context):
        """Generate a response from the TextSynth endpoint asynchronously.

        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.eos_token_id = 198
        args.gen_args.min_length = 1
        response = await self.generate_async(args)
        return response


class OobaModel(ModelProvider):
    """Abstract class for model providers that provide access to generative AI models."""

    def __init__(self, endpoint_url: str, **kwargs):
        """Constructor for ModelProvider.

        :param endpoint_url: The URL of the endpoint.
        :type endpoint_url: str
        """
        self.session = aiohttp.ClientSession()
        self.client = requests.Session()
        super().__init__(endpoint_url, **kwargs)
        self.auth()

    def auth(self) -> bool:
        """
        :drollwide: (hi yoinked!)
        text-generation-webui doesn't have authentication so this just returns true
        """
        return True

    def convert_gen_request(self, req: Union[Dict[str, Any], ModelGenRequest]) -> str:
        if isinstance(req, ModelGenRequest):
            if req.sample_args.logit_biases is not None:
                raise ValueError("logit_biases is not supported by this model provider")
            if req.sample_args.phrase_biases is not None:
                raise ValueError("phrase_biases is not supported by this model provider")

            if req.sample_args.stop_sequence is not None:
                if req.gen_args.stopping_strings is None:
                    req.gen_args.stopping_strings = [req.sample_args.stop_sequence]
                else:
                    req.gen_args.stopping_strings = req.gen_args.stopping_strings.append(
                        req.sample_args.stop_sequence
                    )

            ret = OobaGenRequest(
                prompt=req.prompt,
                do_sample=req.sample_args.do_sample,
                temperature=req.sample_args.temp,
                top_p=req.sample_args.top_p,
                typical_p=req.sample_args.typical_p,
                repetition_penalty=req.sample_args.rep_p,
                top_k=req.sample_args.top_k,
                no_repeat_ngram_size=req.sample_args.no_repeat_ngram_size,
                num_beams=req.sample_args.num_beams,
                penalty_alpha=req.sample_args.penalty_alpha,
                length_penalty=req.sample_args.length_penalty,
                early_stopping=req.sample_args.early_stopping,
                max_new_tokens=req.gen_args.max_length,
                min_length=req.gen_args.min_length,
                seed=req.gen_args.seed,
                add_bos_token=req.gen_args.add_bos_token,
                truncation_length=req.gen_args.truncation_length,
                ban_eos_token=req.gen_args.ban_eos_token,
                skip_special_tokens=req.gen_args.skip_special_tokens,
                stopping_strings=req.gen_args.stopping_strings,
            )
        else:
            ret = OobaGenRequest(**req)
        return ret

    def generate(self, args: Union[Dict[str, Any], ModelGenRequest, OobaGenRequest]) -> str:
        """
        Generate a response from the ModelProvider's endpoint.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        if isinstance(args, (ModelGenRequest, Dict[str, Any])):
            args: ModelGenRequest = self.convert_gen_request(args)

        try:
            r = requests.post(f"{self.endpoint_url}/api/v1/generate", data=args.asjson())
            r.raise_for_status()
        except Exception as e:
            raise e
        if r.status_code == 200:
            return r.json()["results"]["text"]
        else:
            raise Exception(f"Could not generate text with text-generation-webui. Error: {r.json()}")

    async def generate_async(self, args: Union[Dict[str, Any], ModelGenRequest, OobaGenRequest]) -> str:
        """Generate a response from the ModelProvider's endpoint asynchronously.

        :param args: The arguments to pass to the endpoint.
        :type args: dict
        :raises NotImplementedError: If the generate method is not implemented.
        """
        if isinstance(args, (ModelGenRequest, Dict[str, Any])):
            args: ModelGenRequest = self.convert_gen_request(args)
            try:
                async with self.session.post(
                    f"{self.endpoint_url}/api/v1/generate", json=args.asjson()
                ) as resp:
                    if resp.status == 200:
                        ret = await resp.json()
                        return ret["results"]["text"]
                    else:
                        raise Exception(f"Could not generate response. Error: {await resp.text()}")
            except Exception as e:
                raise e

    async def hidden_async(self, model, text, layer):
        """Fetch a layer's hidden states from text.

        :param model: The model to extract hidden states from.
        :type model: str
        :param text: The text to use.
        :type text: str
        :param layer: The layer to fetch the hidden states from.
        :type layer: int
        """

        raise NotImplementedError("hidden_async method is not implemented for this model provider")

    async def image_label_async(self, model, url, labels):
        """Classify an image with labels (CLIP).

        :param model: The model to use for classification.
        :type model: str
        :param url: The image URL to use.
        :type url: str
        :param labels: The labels to use.
        :type labels: list
        """

        raise NotImplementedError("image_label_async method is not implemented for this model provider")

    def should_respond(self, context, name):
        """Determine if the Enma endpoint predicts that the name should respond to the given context.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.sample_args.temp = 0.25
        args.sample_args.top_p = 0.9
        args.sample_args.top_k = 40
        args.sample_args.rep_p = None
        args.sample_args.do_sample = True
        args.sample_args.penalty_alpha = None
        args.sample_args.num_return_sequences = 1  # i have no idea what these should be
        args.sample_args.stop_sequence = None
        response = self.generate(args)
        if name in response:
            return True
        else:
            return False

    async def should_respond_async(self, context, name):
        """Determine if the Enma endpoint predicts that the name should respond to the given context asynchronously.
        :param context: The context to use.
        :type context: str
        :param name: The name to check.
        :type name: str
        :return: Whether or not the name should respond to the given context.
        :rtype: bool
        """

        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.sample_args.temp = 0.25
        args.sample_args.top_p = 0.9
        args.sample_args.top_k = 40
        args.sample_args.rep_p = None
        args.sample_args.do_sample = True
        args.sample_args.penalty_alpha = None
        args.sample_args.num_return_sequences = 1  # i have no idea what these should be
        args.sample_args.stop_sequence = None
        response = await self.generate_async(args)
        if response.startswith(name):
            return True
        else:
            return False

    def response(self, context):
        """Generate a response from the Enma endpoint.
        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.min_length = 1
        response = self.generate(args)
        return response

    async def response_async(self, context):
        """Generate a response from the Enma endpoint asynchronously.
        :param context: The context to use.
        :type context: str
        :return: The response from the endpoint.
        :rtype: str
        """
        args: ModelGenRequest = copy.deepcopy(self.default_args)
        args.prompt = context
        args.gen_args.min_length = 1
        response = await self.generate_async(args)
        return response