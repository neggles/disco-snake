"""OpenAI client wrapper and utilities."""

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, HttpUrl, JsonValue, RootModel, Tag

logger = logging.getLogger(__name__)

type MessageRole = Literal["system", "user", "assistant", "function"]


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None


class FunctionCall(BaseModel):
    name: str | Literal["auto"] = "auto"
    arguments: str | None = None


# chat message content part types
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    url: HttpUrl | str


class ImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl | HttpUrl | str


class ToolCallPart(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


def _content_part_discriminator(v: Any) -> str:
    if isinstance(v, str):
        return "str"
    if isinstance(v, TextPart | ImagePart | ToolCallPart):
        return v.type
    if isinstance(v, RootModel):
        return _content_part_discriminator(v.root)
    if isinstance(v, dict | BaseModel):
        return v.get("type", "str")
    if isinstance(v, list):
        return
    raise ValueError("Invalid message content part")


type ContentPartType = Annotated[
    (
        Annotated[str, Tag("str")]
        | Annotated[TextPart, Tag("text")]
        | Annotated[ImagePart, Tag("image_url")]
        | Annotated[ToolCallPart, Tag("function")]
    ),
    Discriminator(_content_part_discriminator),
]


class ContentPart(RootModel[ContentPartType]):
    """A single message content part."""


class ContentPartList(RootModel[list[ContentPartType]]):
    """A list of message content parts."""


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: str | None


class UserMessage(BaseModel):
    role: Literal["user"]
    content: ContentPartList | ContentPart


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: ContentPartList | ContentPart


type ChatMessageType = Annotated[
    (SystemMessage | UserMessage | AssistantMessage),
    Discriminator("role"),
]


class ChatMessage(RootModel[ChatMessageType]):
    """A single chat message."""


class ChatMessageList(RootModel[list[ChatMessageType]]):
    """A list of chat messages."""

    def append(self, message: ChatMessageType) -> None:
        """Append a message to the chat log."""
        self.root.append(message)
