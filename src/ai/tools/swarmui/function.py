"""OpenAI function definitions for SwarmUI tools."""

from enum import Enum


from .client import SwarmUIClient
from .models import SwarmGeneratedImage, SwarmUISettings


class ToolAspectRatio(str, Enum):
    Wide = "16:9"
    SemiWide = "8:5"
    FilmLandscape = "3:2"
    Landscape = "4:3"
    Square = "1:1"
    Portrait = "3:4"
    FilmPortrait = "2:3"
    SemiTall = "5:8"
    Tall = "9:16"


class SwarmUIGenerationFunction:
    """Function for generating images using SwarmUI."""

    def __init__(self, client: SwarmUIClient):
        self.client = client

    @property
    def settings(self) -> SwarmUISettings:
        return self.client.settings

    def generate_image(
        self,
        prompt: str,
        tags: list[str] | None = None,
        aspect_ratio: ToolAspectRatio = ToolAspectRatio.Square,
    ) -> list[SwarmGeneratedImage]:
        """Generate images using the SwarmUI API."""
        prompt_tags = (self.settings.prefix_tags or []) + (tags or [])
        if prompt_tags:
            prompt = f"{prompt}\nTags: {', '.join(prompt_tags)}"

        if self.settings.prompt_prefix:
            prompt = f"{self.settings.prompt_prefix} {prompt}"
        if self.settings.prompt_suffix:
            prompt = f"{prompt}, {self.settings.prompt_suffix}"

        params = self.settings.default_params.model_copy()
        params.prompt = prompt
        params.aspect_ratio = aspect_ratio

        batch = self.client.generate_images(params)
        if not isinstance(batch, list):
            batch = [batch]

        return batch
