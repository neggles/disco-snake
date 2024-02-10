from typing import Optional

from shimeji.model_provider import ModelProvider
from shimeji.postprocessor import Postprocessor
from shimeji.preprocessor import Preprocessor


class ChatBot:
    def __init__(
        self,
        name: str,
        model_provider: ModelProvider,
        preprocessors: list[Preprocessor | None] = [],
        postprocessors: list[Postprocessor | None] = [],
        **kwargs,
    ):
        self.name = name
        self.model_provider: ModelProvider = model_provider

        self.preprocessors: list[Preprocessor] = preprocessors
        self.postprocessors: list[Postprocessor] = postprocessors

    def preprocess(self, text: str, is_respond: bool = True) -> str:
        for preprocessor in self.preprocessors:
            text = preprocessor(text, is_respond, name=self.name)

        return text.rstrip(" ")

    def postprocess(self, text: str) -> str:
        for postprocessor in self.postprocessors:
            text = postprocessor(text)

        return text.rstrip(" ")

    def should_respond(self, text: str):
        text = self.preprocess(text, is_respond=False)
        return self.model_provider.should_respond(text, self.name)

    async def should_respond_async(self, text: str):
        text = self.preprocess(text, is_respond=False)

        return await self.model_provider.should_respond_async(text, self.name)

    def respond(self, text: str) -> str:
        text = self.preprocess(text, is_respond=False)

        response = self.model_provider.response(text)

        text = self.postprocess(text)

        return response

    async def respond_async(self, text: str, is_respond: bool = True) -> str:
        text = self.preprocess(text, is_respond=False)

        response = await self.model_provider.response_async(text)

        text = self.postprocess(text)

        return response

    def conditional_response(self, text: str) -> Optional[str]:
        if self.should_respond(text):
            return self.respond(text, False)
        else:
            return None
