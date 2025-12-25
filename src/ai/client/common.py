from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from ai.models.openai import ChatMessageList


class GenerationSettingsBase(BaseModel, ABC):
    """Abstract base class for generation settings of different AI clients."""

    def to_request_dict(self, *args, **kwargs) -> dict:
        """Dump the generation settings in a format suitable for API requests."""
        return self.model_dump(
            *args,
            exclude_defaults=True,
            **kwargs,
        )

    def to_request_json(self, *args, **kwargs) -> str:
        """Dump the generation settings as JSON suitable for API requests."""
        return self.model_dump_json(
            *args,
            exclude_defaults=True,
            **kwargs,
        )


class LLMApiClientBase(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        settings: GenerationSettingsBase,
    ) -> str:
        """Generate a completion for the given prompt using the specified generation settings."""
        pass

    @abstractmethod
    def respond(
        self,
        conversation: list[dict] | ChatMessageList,
        settings: GenerationSettingsBase,
    ) -> dict:
        """Generate a chat completion for the given conversation using the specified generation settings."""
        pass

    @abstractmethod
    def tokenize(
        self,
        text: str,
    ) -> list[int]:
        """Tokenize the given text and return a list of token IDs."""
        pass

    @abstractmethod
    def embed(
        self,
        texts: str | list[str],
    ) -> list[list[float]]:
        """Generate embeddings for the given list of texts."""
        pass


class LLMApiClientAsyncBase(ABC):
    """Abstract base class for asynchronous LLM API clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        settings: GenerationSettingsBase,
    ) -> str:
        """Asynchronously generate a completion for the given prompt using the specified generation settings."""
        pass

    @abstractmethod
    async def respond(
        self,
        conversation: list[dict] | ChatMessageList,
        settings: GenerationSettingsBase,
    ) -> dict:
        """Asynchronously generate a chat completion for the given conversation using the specified generation settings."""
        pass

    @abstractmethod
    async def tokenize(
        self,
        text: str,
    ) -> list[int]:
        """Asynchronously tokenize the given text and return a list of token IDs."""
        pass

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Asynchronously generate embeddings for the given list of texts."""
        pass


class LLMStreamingApiClientAsyncBase(LLMApiClientAsyncBase, ABC):
    """Abstract base class for asynchronous LLM API clients with streaming support."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        settings: GenerationSettingsBase,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously generate a completion stream for the given prompt using the specified generation settings."""
        pass

    @abstractmethod
    async def respond_stream(
        self,
        conversation: list[dict] | ChatMessageList,
        settings: GenerationSettingsBase,
    ) -> AsyncGenerator[dict, None]:
        """Asynchronously generate a chat completion stream for the given conversation using the specified generation settings."""
        pass
