from typing import List

from shimeji.model_provider import ModelProvider
from shimeji.postprocessor import Postprocessor
from shimeji.preprocessor import Preprocessor


class ChatBot:
    def __init__(
        self,
        name: str,
        model_provider: ModelProvider,
        preprocessors: List[Preprocessor | None] = [],
        postprocessors: List[Postprocessor | None] = [],
        **kwargs,
    ):
        """Constructor for ChatBot.

        :param name: The name of the chatbot.
        :type name: str
        :param model_provider: The model provider to use.
        :type model_provider: ModelProvider
        """

        self.name = name
        self.model_provider: ModelProvider = model_provider

        self.preprocessors: List[Preprocessor] = preprocessors
        self.postprocessors: List[Postprocessor] = postprocessors

        self.conversation_chain: List[str] = []

    def should_respond(self, text, push_chain):
        """Determine if the chatbot should respond to the given text or conversation chain.

        :param text: The response text.
        :type text: str
        :param push_chain: Whether to push the response to the conversation chain.
        :type push_chain: bool
        :return: Whether the chatbot should respond.
        :rtype: bool
        """

        if push_chain:
            self.conversation_chain.append(text)
        if self.conversation_chain:
            text = "\n".join(self.conversation_chain)

        if self.preprocessors:
            for preprocessor in self.preprocessors:
                text = preprocessor(text, is_respond=False, name=self.name)

        return self.model_provider.should_respond(text, self.name)

    async def should_respond_async(self, text: str, push_chain: bool = False):
        """Determine if the chatbot should respond to the given text or conversation chain asynchronously.

        :param text: The response text.
        :type text: str
        :param push_chain: Whether to push the response to the conversation chain.
        :type push_chain: bool
        :return: Whether the chatbot should respond.
        :rtype: bool
        """

        if push_chain:
            self.conversation_chain.append(text)
        if self.conversation_chain:
            text = "\n".join(self.conversation_chain)

        if self.preprocessors:
            for preprocessor in self.preprocessors:
                text = preprocessor(text, is_respond=False, name=self.name)

        return await self.model_provider.should_respond_async(text, self.name)

    def respond(self, text: str, push_chain: bool = False):
        """Respond to the given text or conversation chain.

        :param text: The response text.
        :type text: str
        :param push_chain: Whether to push the response to the conversation chain.
        :type push_chain: bool
        """

        if push_chain:
            self.conversation_chain.append(text)
        if self.conversation_chain:
            text = "\n".join(self.conversation_chain)

        if self.preprocessors:
            for preprocessor in self.preprocessors:
                text = preprocessor.__call__(text, is_respond=True, name=self.name)

        response = self.model_provider.response(text)

        if self.postprocessors:
            for postprocessor in self.postprocessors:
                response = postprocessor.__call__(response)

        if push_chain:
            self.conversation_chain.append(f"{self.name}:{response}")

        return response

    async def respond_async(self, text: str, push_chain: bool = False, is_respond: bool = True):
        """Respond to the given text or conversation chain asynchronously.

        :param text: The response text.
        :type text: str
        :param push_chain: Whether to push the response to the conversation chain.
        :type push_chain: bool
        """

        if push_chain:
            self.conversation_chain.append(text)
        if self.conversation_chain:
            text = "\n".join(self.conversation_chain)

        if self.preprocessors:
            for preprocessor in self.preprocessors:
                text = preprocessor.__call__(text, is_respond=is_respond, name=self.name)

        response = await self.model_provider.response_async(text)

        if self.postprocessors:
            for postprocessor in self.postprocessors:
                response = postprocessor.__call__(response)

        if push_chain:
            self.conversation_chain.append(f"{self.name}:{response}")

        return response

    def conditional_response(self, text: str, push_chain: bool = True):
        """Respond to the given text or conversation chain if the chatbot should respond.

        :param text: The response text.
        :type text: str
        :param push_chain: Whether to push the response to the conversation chain.
        :type push_chain: bool
        :return: Whether the chatbot should respond.
        :rtype: bool
        """

        if self.should_respond(text, push_chain):
            return self.respond(text, False)
        else:
            return None
