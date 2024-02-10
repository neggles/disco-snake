from abc import ABC, abstractmethod


class Postprocessor(ABC):
    """Abstract class for postprocessors."""

    def __call__(self, context: str) -> str:
        return self.process(context)

    @abstractmethod
    def process(self, context: str) -> str:
        raise NotImplementedError("Abstract base class was called ;_;")


class NewlinePrunerPostprocessor(Postprocessor):
    """Postprocessor that removes newlines."""

    def process(self, context: str) -> str:
        """Process the given context.

        :param context: The context to process.
        :type context: str
        :return: The processed context which has no trailing newlines.
        :rtype: str
        """
        return context.rstrip("\n")
