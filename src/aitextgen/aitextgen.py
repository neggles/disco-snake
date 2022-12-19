import logging
import os
import re
from typing import Optional, Union

import torch
from pkg_resources import resource_filename
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)
from .utils import model_max_length, reset_seed, set_seed

logger = logging.getLogger("aitextgen")
logger.setLevel(logging.DEBUG)

STATIC_PATH = resource_filename(__name__, "static")


class aitextgen:
    """
    Class that serves as the main aitextgen object for training and generation.
    """

    openai_tf_gpt2 = None

    # default values for GPT2Tokenizer
    tokenizer = None
    vocab_file = os.path.join(STATIC_PATH, "gpt2_vocab.json")
    merges_file = os.path.join(STATIC_PATH, "gpt2_merges.txt")
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"

    def __init__(
        self,
        model: str = None,
        model_folder: str = None,
        config: Union[str, PretrainedConfig] = None,
        tokenizer: Union[str, PreTrainedTokenizerFast] = None,
        cache_dir: str = "aitextgen",
        torch_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        to_fp16: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        if model:
            assert not os.path.isfile(model), (
                "As of aitextgen 0.5.0, you must " + "use `model_folder` to load an existing model."
            )

        if not verbose:
            for module in [
                "transformers.file_utils",
                "transformers.configuration_utils",
                "transformers.tokenization_utils",
                "filelock",
                "transformers.modeling_gpt2",
            ]:
                logging.getLogger(module).setLevel(logging.WARN)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        if model_folder:
            logger.info(f"Loading model from provided weights and config in /{model_folder}.")
            self.model = AutoModelForCausalLM.from_pretrained(model_folder, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=True)
        elif config:
            # Manually construct a model from scratch
            logger.info("Constructing model from provided config.")
            if isinstance(config, str):
                config = AutoConfig.from_pretrained(config)
            self.model = AutoModelForCausalLM.from_config(config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(config=config, use_fast=True)
        else:
            # Download and cache model from Huggingface
            if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
                logger.info(f"Loading {model or 'gpt2'} model from /{cache_dir}.")
            else:
                logger.info(f"Downloading {model or 'gpt2'} model to /{cache_dir}.")
            self.model = AutoModelForCausalLM.from_pretrained(model or "gpt2", cache_dir=cache_dir)
            if model and "gpt2" not in model:
                logger.info(f"Using the tokenizer for {model}.")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model,
                    cache_dir=cache_dir,
                    use_fast=True,
                )

        logger.info(self)

        if self.tokenizer is None:
            # Update tokenizer settings (if not set already)
            args = locals()
            custom_tokenizer = False
            for attr in [
                "vocab_file",
                "merges_file",
                "tokenizer_file",
                "tokenizer",
                "bos_token",
                "eos_token",
                "unk_token",
            ]:
                if args[attr] is not None:
                    custom_tokenizer = True
                    setattr(self, attr, args[attr])

            if custom_tokenizer:
                logger.info("Using a custom tokenizer.")
            else:
                logger.info("Using the default GPT-2 Tokenizer.")

            if isinstance(tokenizer, str):
                # load the custom GPT-2 tokenizer from a serialized tokenizer.
                # GPT-Neo uses the GPT-2 tokenizer.
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer,
                    bos_token=self.bos_token,
                    eos_token=self.eos_token,
                    unk_token=self.unk_token,
                    pad_token=self.pad_token,
                )
            elif isinstance(tokenizer, PreTrainedTokenizerFast):
                # use the provided tokenizer
                self.tokenizer = tokenizer
            else:
                self.tokenizer = GPT2TokenizerFast(
                    vocab_file=self.vocab_file,
                    merges_file=self.merges_file,
                    bos_token=self.bos_token,
                    eos_token=self.eos_token,
                    unk_token=self.unk_token,
                    pad_token=self.pad_token,
                    verbose=False,
                )
                if not custom_tokenizer:
                    # https://github.com/huggingface/transformers/issues/10202
                    self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})

        self.tokenizer.padding_side = "left"

        if to_fp16 and torch_device != "cpu":
            self.to_fp16()

        if torch_device != "cpu":
            self.to(torch_device)

    def generate(
        self,
        n: int = 1,
        prompt: str = "",
        prepend_bos: bool = None,
        min_length: int = None,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        seed: int = None,
        pad_token_id: str = None,
        use_cache: bool = True,
        lstrip: bool = True,
        nonempty_output: bool = True,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Optional[str]:
        """
        Generates texts using the stored Transformers model.
        """

        prompt_tensors = self.tokenizer(text=prompt, return_tensors="pt", padding=True).to(self.get_device())

        input_ids = prompt_tensors["input_ids"].to(self.get_device()) if prompt else None

        if prepend_bos is None:
            prepend_bos = getattr(self.model.config, "line_by_line", None)

        if prepend_bos:
            bos = self.tokenizer(self.tokenizer.bos_token, return_tensors="pt").to("cuda:0").input_ids
            input_ids = torch.cat((bos, input_ids), dim=1)

        if seed:
            set_seed(seed)

        return_as_list = kwargs.pop("return_as_list", None)
        base_length = kwargs.pop("base_length", None)

        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.eos_token_id

        # prevent an error from using a length greater than the model
        gen_max_length = model_max_length(self.model.config)
        max_length = min(gen_max_length, max_length)

        while True:
            outputs = self.model.generate(
                input_ids=input_ids,
                min_length=min_length,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=n,
                pad_token_id=pad_token_id,
                use_cache=use_cache,
                **kwargs,
            )

            gen_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)

            # Handle stripping tokenization spaces w/ regex
            if lstrip:
                gen_texts = [re.sub(r"^\s+", "", text) for text in gen_texts]

            if nonempty_output:
                if min_length:
                    gen_texts = list(filter(lambda x: len(x) > min_length, gen_texts))
                else:
                    gen_texts = list(filter(lambda x: len(x) > 0, gen_texts))

            # if there is no generated text after cleanup, try again.
            if len(gen_texts) == 0:
                continue

            # Reset seed if used
            if seed:
                reset_seed()

            return gen_texts

    def generate_one(self, **kwargs) -> None:
        """
        Generates a single text, and returns it as a string. Useful for
        returning a generated text within an API.

        See generate() for more parameters.
        """

        return self.generate(n=1, **kwargs)[0]

    def save(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        self.model.save_pretrained(target_folder)
        self.tokenizer.save_pretrained(target_folder)

    def save_for_upload(self, target_folder: str = "my-model"):
        """
        Saves the model + tokenizerinto the specified directory.

        This generates the 6 files needed to upload the model to
        Huggingface's S3 bucket.
        """
        self.model.save_pretrained(target_folder)
        self.tokenizer.save_pretrained(target_folder)

    def export(
        self,
        quantize: bool = True,
    ) -> None:
        """
        Exports the model, with optional quantization
        """

    def to(self, device: str, index: int = None) -> None:
        """Moves the model to the specified device."""
        if index is not None:
            self.model.to(torch.device(device, index))
        else:
            self.model.to(torch.device(device))

    def to_gpu(self, index: int = 0) -> None:
        """Moves the model to the specified GPU."""

        assert torch.cuda.is_available(), "CUDA is not installed."

        self.to(torch.device("cuda", index))

    def to_cpu(self, index: int = 0) -> None:
        """Moves the model to the specified CPU."""

        self.to(torch.device("cpu", index))

    def to_fp16(self) -> None:
        """
        Converts the model to a FP16 representation.
        Should only be used to generate on a supported GPU.
        """

        self.model = self.model.half()

    def get_device(self) -> str:
        """Getter for the current device where the model is located."""
        return self.model.device.type

    def __repr__(self) -> str:
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/24
        num_params_m = int(sum(p.numel() for p in self.model.parameters()) / 10**6)
        model_name = type(self.model.config).__name__.replace("Config", "")
        return f"{model_name} loaded with {num_params_m}M parameters."
