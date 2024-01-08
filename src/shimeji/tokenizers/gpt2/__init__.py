from pathlib import Path

from transformers import GPT2TokenizerFast

tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(Path(__file__).parent)
