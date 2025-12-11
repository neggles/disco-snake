from shimeji.model_provider import ModelProvider
from shimeji.postprocessor import Postprocessor
from shimeji.preprocessor import Preprocessor


class ChatBot:
    def __init__(
        self,
        name: str,
        model_provider: ModelProvider,
        preprocessors: list[Preprocessor] | None = None,
        postprocessors: list[Postprocessor] | None = None,
        **kwargs,
    ):
        self.name = name
        self.model_provider: ModelProvider = model_provider

        self.preprocessors: list[Preprocessor] = preprocessors if preprocessors is not None else []
        self.postprocessors: list[Postprocessor] = postprocessors if postprocessors is not None else []

    def preprocess(self, text: str, is_respond: bool = True) -> str:
        for preprocessor in self.preprocessors:
            text = preprocessor(text, is_respond, name=self.name)

        return text.rstrip(" ")

    def postprocess(self, text: str) -> str:
        for postprocessor in self.postprocessors:
            text = postprocessor(text)

        return text.rstrip(" ")

    def respond(self, text: str) -> str:
        text = self.preprocess(text, is_respond=False)

        response: str = self.model_provider.response(text, return_dict=False)  # type: ignore

        text = self.postprocess(text)

        return response

    async def respond_async(self, text: str, is_respond: bool = True) -> str:
        text = self.preprocess(text, is_respond=False)

        response: str = await self.model_provider.response_async(text, return_dict=False)  # type: ignore

        text = self.postprocess(text)

        return response
