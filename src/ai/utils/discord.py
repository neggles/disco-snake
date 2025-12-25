import logging
from functools import lru_cache

from aiohttp import ClientSession
from disnake import (
    DMChannel,
    Emoji,
    GroupChannel,
    Interaction,
    Member,
    Message,
    Role,
    TextChannel,
    Thread,
)

from ai.constants import MAX_IMAGE_BYTES, MAX_IMAGE_BYTES_STR, MAX_IMAGES_PER_MESSAGE
from ai.models.openai import ContentPart, ImagePart
from ai.types import MessageChannel
from disco_snake.bot import DiscoSnake

from .image import fetch_image_file_async, image_part_from_bytes

logger = logging.getLogger(__name__)


def member_in_role(member: Member, role: Role | int) -> bool:
    """Returns True if the user has the role"""
    role_id = getattr(role, "id", role)
    return role_id in [x.id for x in member.roles]


def member_in_any_role(member: Member, roles: list[Role | int]) -> bool:
    """Returns True if the user has any of the roles"""
    return any(member_in_role(member, role) for role in roles)


@lru_cache(maxsize=100, typed=True)
def _dedupe_custom_emoji(emojis: list[Emoji]) -> list[Emoji]:
    seen_names = set()
    deduped = []
    for emo in emojis:
        if not emo.is_usable():
            continue
        if emo.name not in seen_names:
            deduped.append(emo)
            seen_names.add(emo.name)
            continue
    return deduped


@lru_cache(maxsize=100, typed=True)
def get_usable_custom_emoji(context: Interaction | MessageChannel) -> list[Emoji]:
    """Get a list of available emojis for the message context"""
    bot: DiscoSnake = context.client  # type: ignore
    usable = []

    match context:
        case Interaction():
            # if we have permissions, use all emojis
            if context.app_permissions.external_emojis:
                usable = _dedupe_custom_emoji(bot.emojis)
        case TextChannel() | Thread():
            # if we have permissions, use all emojis
            if context.permissions_for(bot.user).external_emojis:
                usable = _dedupe_custom_emoji(bot.emojis)
        case DMChannel() | GroupChannel():
            # all emojis are available in DMs and group chats
            usable = _dedupe_custom_emoji(bot.emojis)
        case _:
            logger.error(f"Unknown context type for emoji retrieval: {context!r}")

    # fallback to guild emojis if we have none and we're in a guild context
    if not usable and context.guild:
        usable = _dedupe_custom_emoji(context.guild.emojis)

    return usable


async def get_user_content_parts(
    session: ClientSession,
    content: str,
    message: Message,
    author_name: str | None = None,
    vision: bool = False,
    embed_send_url: bool = True,
) -> list[ContentPart]:
    """Fetch and return the content parts of a message sent by a user."""
    parts: list[ContentPart] = []
    if not content:
        if message.content:
            content = f"{author_name}: {message.content}" if author_name else message.content
        else:
            logger.info(f"Message {message.id} has no text content, will only process images.")
    else:
        content = f"{author_name}: {content}" if author_name else content

    if content:
        parts.append({"type": "text", "text": content})

    # process attachments and embeds
    n_attached = 0
    for attachment in message.attachments:
        # TODO: handle text attachments, probably as a link to the file to avoid bloat
        if not vision:
            break
        try:
            if n_attached >= MAX_IMAGES_PER_MESSAGE:
                break
            if not attachment.content_type:
                logger.debug(f"Skipping attachment {attachment.id} (unknown content type)")
                continue
            if not attachment.content_type.startswith("image/"):
                logger.debug(
                    f"Skipping attachment {attachment.id} ({attachment.content_type} is not an image)"
                )
                continue
            if attachment.size > MAX_IMAGE_BYTES:
                logger.warning(f"Skipping image {attachment.id} (over {MAX_IMAGE_BYTES_STR})")
                continue

            # fetch attachment
            payload = await attachment.read()
            parts.append(image_part_from_bytes(payload))
            n_attached += 1

        except Exception:
            logger.exception(f"Failed to fetch attachment {attachment.url}")

    if n_attached >= MAX_IMAGES_PER_MESSAGE:
        return parts

    for embed in message.embeds:
        # TODO: handle text-only embeds properly instead of just skipping them when vision is disabled
        if not vision:
            break
        try:
            if n_attached >= MAX_IMAGES_PER_MESSAGE:
                break
            if embed.image and embed.image.url:
                if embed_send_url:
                    # just send the URL and let the backend fetch it
                    parts.append(ImagePart(image_url=embed.image.url))
                else:
                    # fetch embed image
                    payload = await fetch_image_file_async(embed.image.url, session=session)
                    if payload is None:
                        logger.debug(f"Skipping embed image from {embed.image.url} (failed to fetch)")
                        continue
                    parts.append(image_part_from_bytes(payload))
                n_attached += 1
            elif embed.description:
                parts.append(
                    {"type": "text", "text": f"[Embed, description={embed.description} url={embed.url}]"}
                )
        except Exception:
            logger.exception(f"Failed to fetch embed image from {embed.image.url}")

    return parts
