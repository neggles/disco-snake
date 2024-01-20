import re
from enum import Enum
from typing import Any

from transformers import PreTrainedTokenizerBase

from shimeji.tokenizers import Llama


class TrimDir(int, Enum):
    Top = 0
    Bottom = 1
    Never = 2


class BreakType(int, Enum):
    Newline = 0
    Sentence = 1
    Token = 2


def split_into_sentences(text: str) -> list[str | Any]:
    # preserve line breaks too
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)


def trim_newlines(
    tokens: list[int],
    trim_dir: TrimDir,
    limit: int,
    tokenizer: PreTrainedTokenizerBase,
):
    if (trim_dir == TrimDir.Never) or (len(tokens) <= limit):
        return tokens

    lines = tokenizer.decode(tokens).split("\n")
    start, end, step = 0, 0, 0
    if trim_dir == TrimDir.Top:
        start = len(lines) - 1
        end = -1
        step = -1
    elif trim_dir == TrimDir.Bottom:
        start = 0
        end = len(lines)
        step = 1

    acc_tokens = []

    for idx in range(start, end, step):
        line = lines[idx]
        if trim_dir == TrimDir.Top:
            line = "\n" + line
        elif trim_dir == TrimDir.Bottom:
            line = line + "\n"
        new_tokens = tokenizer.encode(line)
        if len(new_tokens) + len(acc_tokens) > limit:
            return acc_tokens
        else:
            if trim_dir == TrimDir.Top:
                acc_tokens = new_tokens + acc_tokens
            elif trim_dir == TrimDir.Bottom:
                acc_tokens = acc_tokens + new_tokens
    return acc_tokens


def trim_sentences(
    tokens: list[int],
    trim_dir: TrimDir,
    limit: int,
    tokenizer: PreTrainedTokenizerBase,
):
    if (trim_dir == TrimDir.Never) or (len(tokens) <= limit):
        return tokens

    text = tokenizer.decode(tokens)
    sentences = split_into_sentences(text)

    start, end, step = 0, 0, 0
    text_begin, text_end = 0, 0
    sentence_idx, last_sentence_idx = 0, 0

    match trim_dir:
        case TrimDir.Top:
            start = len(sentences) - 1
            end = -1
            step = -1

        case TrimDir.Bottom:
            start = 0
            end = len(sentences)
            step = 1
        case _:
            return tokens

    text_begin = 0
    text_end = len(text)

    for idx in range(start, end, step):
        sentence = sentences[idx]
        if trim_dir == TrimDir.Top:
            sentence_idx = text.rindex(sentence) + text_begin
            if (sentence_idx > 0) and (sentence_idx < len(text)) and (text[sentence_idx] == " "):
                sentence_idx -= 1
            to_tokenize = text[sentence_idx:]
            token_count = len(tokenizer.encode(to_tokenize))
            if token_count >= limit:
                to_encode = text[text_end:]
                return tokenizer.encode(to_encode)
            text_end = sentence_idx - 1

        elif trim_dir == TrimDir.Bottom:
            sentence_idx = text.index(sentence) + text_begin
            sentence_end = sentence_idx + len(sentence)
            if (sentence_end < text_end) and (text[sentence_end : sentence_end + 1] == "\n"):
                sentence_end += 1
            to_tokenize = text[0:sentence_end]
            token_count = len(tokenizer.encode(to_tokenize))
            if token_count >= limit:
                to_encode = text[0:last_sentence_idx]
                return tokenizer.encode(to_encode)
            last_sentence_idx = sentence_end
            text_begin += len(sentence)

    return tokens


def trim_tokens(tokens: list[int], trim_dir: TrimDir, limit: int):
    overrun = len(tokens) - limit
    if overrun <= 0:
        return tokens

    match trim_dir:
        case TrimDir.Top:
            return tokens[overrun:]
        case TrimDir.Bottom:
            return tokens[:limit]
        case _:
            return tokens


class ContextEntry:
    def __init__(
        self,
        keys: list[str] = [""],
        text: str = "",
        prefix: str = "",
        suffix: str = "",
        token_budget: int = 2048,
        reserved_tokens: int = 0,
        insertion_order: int = 100,
        insertion_position: int = -1,
        trim_direction: TrimDir = TrimDir.Bottom,
        trim_type: BreakType = BreakType.Sentence,
        insertion_type: BreakType = BreakType.Sentence,
        forced_activation: bool = False,
        cascading_activation: bool = False,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        self.keys = keys  # key used to activate this context entry
        self.text = prefix + text + suffix  # text associated with this context entry
        self.token_budget = token_budget  # max amount of tokens that this context entry can use
        self.reserved_tokens = reserved_tokens  # number of tokens that are reserved for this context entry
        self.insertion_order = insertion_order  # order in which this context entry is inserted
        self.insertion_position = insertion_position  # position in the text where this context entry is inserted, 0 is the beginning, -1 is the end
        self.trim_direction = trim_direction  # direction in which to trim the text
        self.trim_type = trim_type  # type of trimming to perform
        self.insertion_type = insertion_type  # determines what units are used to insert the text
        self.forced_activation = (
            forced_activation  # if True, this context entry is activated even if it is not activated
        )
        self.cascading_activation = cascading_activation  # when activated, this context entry will search for other entries and activate them if found
        self.tokenizer = tokenizer if tokenizer is not None else Llama

    # max_length is in tokens
    def trim(self, max_length: int, token_budget: int):
        target_tokens = 0
        tokens = self.tokenizer.encode(self.text)
        token_count = len(tokens)

        projected = max_length - token_count
        if projected > token_budget:
            target_tokens = token_budget
        elif projected >= 0:
            target_tokens = token_count
        else:
            target_tokens = max_length

        match self.trim_type:
            case BreakType.Newline:
                tokens = self.trim_newlines(tokens, self.trim_direction, target_tokens)
            case BreakType.Sentence:
                tokens = self.trim_sentences(tokens, self.trim_direction, target_tokens)
            case BreakType.Token:
                tokens = self.trim_tokens(tokens, self.trim_direction, target_tokens)
            case _:
                if len(tokens) > target_tokens:
                    tokens = self.trim_tokens(tokens, self.trim_direction, target_tokens)

        return tokens

    def get_text(self, max_length: int, token_budget: int):
        return self.tokenizer.decode(self.trim(max_length, token_budget))

    def trim_newlines(self, tokens, trim_dir, limit):
        return trim_newlines(tokens, trim_dir, limit, self.tokenizer)

    def trim_sentences(self, tokens, trim_dir, limit):
        return trim_sentences(tokens, trim_dir, limit, self.tokenizer)

    def trim_tokens(self, tokens, trim_dir, limit):
        return trim_tokens(tokens, trim_dir, limit)
