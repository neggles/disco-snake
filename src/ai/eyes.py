import logging
from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from typing import Optional, TypeAlias, Union

from aiohttp import ClientSession
from disnake import Attachment
from PIL import Image

import logsnake
from ai.config import VisionConfig
from disco_snake import LOG_FORMAT, LOGDIR_PATH

# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name="disco-eyes",
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath("disco-eyes.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=2,
)

API_PATH = "/api/v1/caption"
ImageOrBytes: TypeAlias = Union[Image.Image, bytes]


class DiscoEyes:
    def __init__(self, config: VisionConfig):
        self.config = config

    @property
    def api_host(self):
        return self.config.api_host

    @property
    def api_token(self):
        return self.config.api_token

    async def caption(self, image: ImageOrBytes) -> str:
        return await self.perceive(image)

    async def perceive(self, image: ImageOrBytes) -> str:
        logger.info("Processing image")
        if not isinstance(image, Image.Image):
            logger.debug("Converting image to PIL Image")
            image = Image.open(BytesIO(image), formats=["PNG", "WEBP", "JPEG", "GIF"])

        logger.debug("Resizing image")
        buf = BytesIO()
        image = image.copy()
        image.thumbnail((512, 512))
        image.save(buf, format="PNG")

        logger.debug("Sending image to API")
        payload = {"image": b64encode(buf.getvalue()).decode()}
        async with ClientSession(
            base_url=self.api_host,
            headers={"Authorization": f"Bearer {self.api_token}"},
        ) as session:
            async with session.post(API_PATH, json=payload) as resp:
                data = await resp.json()
                resp.raise_for_status()
                caption = data["caption"]
                logger.info(f"Received caption: {caption}")
                return caption

    @lru_cache(maxsize=128)
    async def perceive_cached(self, attachment: Attachment) -> Optional[str]:
        if not attachment.content_type.startswith("image/"):
            return None
        data = await attachment.read()
        return await self.perceive(data)
