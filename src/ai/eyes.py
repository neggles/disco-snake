import asyncio
import logging
import re
from base64 import b64encode
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from aiohttp import ClientError, ClientSession
from disnake import Attachment
from PIL import Image
from pydantic import BaseModel, Field
from requests import get as requests_get

from ai.settings import VisionConfig
from ai.types import ImageOrBytes, MessageChannel, TimestampStore
from db import ImageCaption, Session, SessionType

if TYPE_CHECKING:
    from ai import Ai

# setup cog logger
logger = logging.getLogger(__name__)


IMAGE_MAX_BYTES = 20 * (2**20)
IMAGE_MAX_PX = 768
IMAGE_FORMATS = ["PNG", "WEBP", "JPEG", "GIF"]

re_image_of = re.compile(
    r"(?:This|the)\s+(?:is an image of|image is of)\s+(.+)",
    re.I + re.M,
)


class InfoResponse(BaseModel):
    model_name: str = Field(...)
    model_type: str = Field(...)
    uptime: str = Field(...)
    memory_stats: dict = Field(...)


class CaptionResponse(BaseModel):
    caption: str = Field(...)
    info: Optional[InfoResponse] = Field(None)
    error: Optional[str] = Field(None)


class DiscoEyes:
    def __init__(self, cog: "Ai") -> None:
        self.cog: "Ai" = cog
        self.config: VisionConfig = cog.config.vision  # type: ignore
        self.timezone: ZoneInfo = cog.bot.timezone
        self.web_client: ClientSession = None  # type: ignore
        self.api_client: ClientSession = None  # type: ignore
        self.db_client: SessionType = None  # type: ignore

        # this tracks which channels we've recently responded in, so the caption engine can
        # proactively caption images in those channels without waiting for a prompt
        self.attention: TimestampStore = None  # type: ignore

    async def start(self) -> None:
        self.web_client: ClientSession = ClientSession(
            loop=self.cog.bot.loop,
        )
        self.api_client = ClientSession(
            loop=self.cog.bot.loop,
            base_url=self.api_host,
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        self.db_client = Session
        self.attention = TimestampStore(ttl=self.config.channel_ttl)

        self._api_info: InfoResponse = None
        await self._fetch_api_info()

    def shutdown(self) -> None:
        logger.info("Closing web and API clients...")
        close_tasks = [
            asyncio.ensure_future(self.web_client.close()),
            asyncio.ensure_future(self.api_client.close()),
        ]
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*close_tasks))

    @property
    async def enabled(self) -> bool:
        return self.config.enabled

    @property
    async def background(self) -> bool:
        return self.config.background

    @property
    def api_host(self) -> str:
        return self.config.host

    @property
    def api_token(self) -> Optional[str]:
        return self.config.token

    @property
    def api_info(self) -> InfoResponse:
        return self._api_info

    async def _fetch_api_info(self) -> None:
        async with self.api_client.get("/api/v1/info") as resp:
            resp.raise_for_status()
            data = await resp.json(encoding="utf-8")
        self._api_info = InfoResponse.model_validate(data)
        logger.debug(f"Received API info: {self._api_info}")

    async def submit_request(self, image: ImageOrBytes, return_obj: bool = False) -> str | CaptionResponse:
        logger.info("Processing image")
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image), formats=IMAGE_FORMATS)

        buf = BytesIO()
        image = image.copy()
        image.thumbnail((512, 512))
        image.save(buf, format="PNG")

        payload = {"image": b64encode(buf.getvalue()).decode()}
        async with self.api_client.post(self.config.route, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json(encoding="utf-8")
        response = CaptionResponse.parse_obj(data)

        # Strip out "This is an image of" etc. from the caption
        caption = re_image_of.sub(r"\1", response.caption).rstrip(".")

        if len(caption) <= 4:
            raise ValueError(f"Received caption is too short: {caption=}")
        elif len(caption) > 320:
            caption = caption[:320] + "..."

        # update the object with the stripped caption
        response.caption = caption
        logger.info(f"Received caption: {caption}")
        return response if return_obj else caption

    async def perceive_url(self, url: str, id: Optional[int] = None) -> str | CaptionResponse | None:
        if id is not None:
            image_caption = await self.db_client_get_caption(id)
            if image_caption is not None:
                return image_caption.caption

        async with self.web_client.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            caption = await self.submit_request(data)
            return caption

    # Returns the caption for an image attachment from the db_client if it exists, otherwise
    # submits the image to the api_client for captioning and saves the result to the db_client.
    async def perceive_attachment(self, attachment: Attachment) -> str | CaptionResponse | None:
        if attachment.content_type is None:
            logger.debug("Attachment has no content type")
            return None

        if not attachment.content_type.startswith("image/"):
            logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
            return None
        image_caption = await self.db_client_get_caption(attachment.id)
        if image_caption is not None:
            return image_caption.caption

        logger.info(f"Captioning image {attachment.id}")
        attachment_dict = attachment.to_dict()
        _ = attachment_dict.pop("content_type")  # don't need this

        caption_text = None  # make it not complain if we get a different exception
        try:
            caption_text = await self._submit_attachment(attachment)
        except (ClientError, ValueError) as e:
            logger.error(f"Failed to caption image attachment: {e}")
        finally:
            if caption_text is None:
                raise ValueError("Failed to caption image attachment")

        image_caption = ImageCaption(
            id=attachment.id,
            filename=attachment.filename,
            description=attachment.description,
            size=attachment.size,
            url=attachment.url,
            proxy_url=attachment.proxy_url,
            height=attachment.height or 0,
            width=attachment.width or 0,
            captioned_with=f"{self.api_info.model_type} {self.api_info.model_name}",
            caption=caption_text,
            captioned_at=datetime.now(tz=self.timezone),
        )
        await self.db_client_save_caption(image_caption)
        return image_caption.caption

    # Submits the image to the api_client for captioning
    async def _submit_attachment(self, attachment: Attachment) -> str | CaptionResponse | None:
        if attachment.content_type is None:
            logger.debug("Attachment has no content type")
            return None
        if attachment.size > IMAGE_MAX_BYTES:
            logger.debug(f"got attachment larger than 20MB: {attachment.size}, skipping")
            return None
        if not attachment.content_type.startswith("image/"):
            logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
            return None

        max_edge = max(attachment.width or 0, attachment.height or 0)
        if attachment.width is None or attachment.height is None:
            if attachment.size > IMAGE_MAX_BYTES:
                logger.debug(f"got attachment larger than 20MB: {attachment.size}, skipping")
                return None
            data = requests_get(attachment.url).content
        elif max_edge > IMAGE_MAX_PX or max_edge == 0:
            data = await self._get_thumbnail(attachment)
        else:
            data = await attachment.read()
        if not isinstance(data, ImageOrBytes):
            logger.debug("Attachment is not a bytes object")
            return None
        return await self.submit_request(data)

    async def _get_thumbnail(self, attachment: Attachment) -> Optional[Image.Image]:
        if attachment.content_type is None:
            logger.debug("Attachment has no content type")
            return None
        if not attachment.content_type.startswith("image/"):
            logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
            return None

        # get width and height of attachment image
        if attachment.width is None or attachment.height is None:
            logger.debug("Attachment has no width or height")
            return None

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
        async with self.web_client.get(scaled_url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            image = Image.open(BytesIO(data), formats=IMAGE_FORMATS)
            return image

    async def db_client_save_caption(self, caption: ImageCaption) -> str:
        """Save a caption to the database.

        Returns the same object that was passed in, allowing for
        chaining of calls e.g. `return await self.db_client_save_caption(caption)`

        :param caption: The caption to save.
        :type caption: ImageCaption
        :return: The saved caption.
        :rtype: str
        """
        logger.info(f"Saving caption for image {caption.id}")
        caption_str = caption.caption
        async with self.db_client.begin() as session:
            await session.merge(caption)
        logger.debug("Caption saved successfully")
        return caption_str

    async def db_client_get_caption(self, image_id: int) -> Optional[ImageCaption]:
        async with self.db_client.begin() as session:
            caption = await session.get(ImageCaption, image_id)
        return caption

    def watch(self, channel: MessageChannel):
        self.attention.refresh(channel.id)

    def watching(self, channel: MessageChannel) -> bool:
        if self.background is False:
            return False
        return self.attention.active(channel.id)
