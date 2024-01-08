from functools import partial
from pathlib import Path

from transformers import LlamaTokenizerFast

tokenizer: LlamaTokenizerFast = partial(
    LlamaTokenizerFast.from_pretrained,
    str(Path(__file__).parent),
)
