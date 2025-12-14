from os import PathLike
from typing import Any

from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.tokenization_utils import AddedToken
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "tokenizer_file": {},
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class YiTokenizer(LlamaTokenizer):
    """
    Construct a Yi tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: PathLike,
        unk_token: str | AddedToken = "<unk>",
        bos_token: str | AddedToken = "<|startoftext|>",
        eos_token: str | AddedToken = "<|endoftext|>",
        pad_token: str | AddedToken = "<unk>",
        sp_model_kwargs: dict[str, Any] | None = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
