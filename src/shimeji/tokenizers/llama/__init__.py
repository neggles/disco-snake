from pathlib import Path

from transformers import LlamaTokenizerFast

tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(Path(__file__).parent)
