from functools import partial
from pathlib import Path
from typing import Callable

from transformers import LlamaTokenizerFast

tokenizer_path: Path = Path(__file__).parent

tokenizer: Callable[..., LlamaTokenizerFast] = partial(
    LlamaTokenizerFast.from_pretrained,
    str(tokenizer_path),
)
