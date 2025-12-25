import logging
from io import BytesIO
from typing import Literal

from aiohttp import ClientSession
from httpx import Client
from PIL import Image
from PIL.Image import Resampling
from PIL.ImageFile import ImageFile

from ai.constants import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_BYTES_STR,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
)
from ai.models.openai import ImagePart, ImageUrl

from .misc import data_uri_from_bytes

logger = logging.getLogger(__name__)


class FileSizeError(Exception):
    """Raised when an image exceeds the maximum allowed size."""

    pass


def fetch_image_file(
    url: str,
    session: Client | None = None,
) -> bytes | None:
    """Fetch image bytes from a URL, ensuring it does not exceed MAX_IMAGE_BYTES."""
    close_session = False
    if session is None:
        session = Client()
        close_session = True

    try:
        with session:
            resp = session.get(url)
            resp.raise_for_status()
            if len(resp.content) > MAX_IMAGE_BYTES:
                raise FileSizeError()
            return resp.content
    except FileSizeError:
        logger.warning(f"Image at {url} is over {MAX_IMAGE_BYTES_STR}, skipping.")
        return None
    except Exception:
        logger.exception(f"Failed to fetch image from {url}")
        return None
    finally:
        if close_session:
            session.close()


async def fetch_image_file_async(
    url: str,
    session: ClientSession | None = None,
) -> bytes | None:
    """Fetch image bytes from a URL, ensuring it does not exceed MAX_IMAGE_BYTES."""
    close_session = False
    if session is None:
        session = ClientSession()
        close_session = True

    try:
        async with session.get(url) as response:
            response.raise_for_status()
            content_length = response.content_length
            if content_length and content_length > MAX_IMAGE_BYTES:
                raise FileSizeError()
            data = await response.read()
            if len(data) > MAX_IMAGE_BYTES:
                raise FileSizeError()
            return data
    except FileSizeError:
        logger.warning(f"Image at {url} is over {MAX_IMAGE_BYTES_STR}, skipping.")
        return None
    except Exception:
        logger.exception(f"Failed to fetch image from {url}")
        return None
    finally:
        if close_session:
            await session.close()


def enforce_image_resolution(
    image: bytes | Image.Image,
    max_px: tuple[int, int] = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT),
    format: Literal["JPEG", "PNG"] = "JPEG",
    bytes_only: bool = False,
) -> bytes | tuple[ImageFile, bytes]:
    """Ensure the image does not exceed max resolution, downscaling if necessary."""
    # load image if bytes provided
    if isinstance(image, bytes):
        imgbuf = Image.open(BytesIO(image))
    elif isinstance(image, Image.Image):
        imgbuf = image
    else:
        raise TypeError("image must be bytes or PIL.Image.Image")

    w, h = imgbuf.size
    if w > max_px[0] or h > max_px[1]:
        if imgbuf is image:
            imgbuf = imgbuf.copy()
        imgbuf.thumbnail(max_px, Resampling.LANCZOS)
        logger.debug(f"Resized image from {w}x{h} to {imgbuf.size[0]}x{imgbuf.size[1]}")

    if isinstance(image, bytes):
        image_bytes = image
    else:
        with BytesIO() as buf:
            imgbuf.save(buf, format=format, quality=90 if format == "JPEG" else None)
            image_bytes = buf.getvalue()

    if bytes_only:
        return image_bytes
    else:
        return imgbuf, image_bytes


def image_part_from_bytes(
    data: bytes,
    max_px: tuple[int, int] = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT),
) -> ImagePart:
    """Create an image content part from image bytes."""

    payload = enforce_image_resolution(data, max_px=max_px, format="JPEG", bytes_only=True)
    payload_uri = data_uri_from_bytes(payload, filename="image.jpg", fallback_mime="image/jpeg")

    return ImagePart(type="image_url", image_url=ImageUrl(url=payload_uri))
