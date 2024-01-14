import logging
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def extract_tokenizer(name: str, target_dir: Path):
    try:
        data_dir = resources.files(f"{__name__.split('.')[0]}.tokenizers.{name}")
        for f in data_dir.iterdir():
            target_dir.joinpath(f.name).write_bytes(f.read_bytes())
    except Exception as e:
        logger.exception(f"Failed to extract tokenizer: {e}")
        raise e


def get_tokenizer(name: str = "llama", **kwargs) -> PreTrainedTokenizerBase:
    """Get a tokenizer by extracting it from package data files and loading it"""
    # pop these kwargs if present to avoid accidental overrides
    _, _ = kwargs.pop("pretrained_model_name_or_path", None), kwargs.pop("local_files_only", None)
    # extract tokenizer to temp dir
    with TemporaryDirectory(prefix="disco_snake_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        extract_tokenizer(name, temp_path)
        # load tokenizer from temp dir
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=temp_path,
            local_files_only=True,
            **kwargs,
        )
    # return tokenizer if found
    if tokenizer is None:
        raise ValueError("Tokenizer not found!")
    return tokenizer
