import logging
from base64 import b64encode
from io import BytesIO
from typing import Optional

from aiohttp import ClientSession
from async_lru import alru_cache
from disnake import Attachment, Embed
from PIL import Image
from requests import get as requests_get

import logsnake
from ai.config import VisionConfig
from ai.types import ImageOrBytes
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

IMAGE_MAX_BYTES = 20 * (2**20)
IMAGE_MAX_PX = 768
IMAGE_FORMATS = ["PNG", "WEBP", "JPEG", "GIF"]


class DiscoEyes:
    def __init__(self, config: VisionConfig):
        self.config = config

    @property
    def api_host(self):
        return self.config.api_host

    @property
    def api_token(self):
        return self.config.api_token

    async def perceive(self, image: ImageOrBytes) -> str:
        logger.info("Processing image")
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image), formats=IMAGE_FORMATS)

        buf = BytesIO()
        image = image.copy()
        image.thumbnail((512, 512))
        image.save(buf, format="PNG")

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

    @alru_cache(maxsize=128)
    async def perceive_attachment(self, attachment: Attachment) -> Optional[str]:
        if attachment.size > IMAGE_MAX_BYTES:
            logger.debug(f"got attachment larger than 20MB: {attachment.size}, skipping")
            return None
        if not attachment.content_type.startswith("image/"):
            logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
            return None

        if max(attachment.width, attachment.height) > IMAGE_MAX_PX:
            data = await self.get_attachment_scaled(attachment)
        else:
            data = await attachment.read()
        caption = await self.perceive(data)
        return caption

    @alru_cache(maxsize=128)
    async def perceive_image_embed(self, embed: Embed) -> Optional[str]:
        if embed.type != "image":
            raise ValueError("Embed is not an image embed")
        caption = await self.perceive(requests_get(embed.image.url).raw)
        return caption

    async def get_attachment_scaled(self, attachment: Attachment) -> Optional[Image.Image]:
        if not attachment.content_type.startswith("image/"):
            logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
            return None

        # get width and height of attachment image
        width, height = attachment.width, attachment.height
        # scale max dimension to IMAGE_MAX_PX
        is_portrait = height > width
        short, long = (width, height) if is_portrait else (height, width)
        # calculate new dimensions
        if long > IMAGE_MAX_PX:
            ratio = IMAGE_MAX_PX / long
            short = int(short * ratio)
            long = IMAGE_MAX_PX
        width, height = (short, long) if is_portrait else (long, short)

        # change cdn url to media url
        image_url = attachment.url.replace("cdn.discordapp.com", "media.discordapp.net")
        # add query params to url
        scaled_url = f"{image_url}?width={width}&height={height}"
        # fetch image
        logger.debug(f"Fetching scaled image from {image_url}")
        async with ClientSession() as session:
            async with session.get(scaled_url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                image = Image.open(BytesIO(data), formats=IMAGE_FORMATS)
                return image
