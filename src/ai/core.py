import asyncio
import json
import logging
import re
from asyncio import Lock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

from disnake import (
    ApplicationCommandInteraction,
    Colour,
    DMChannel,
    Embed,
    File,
    GroupChannel,
    Member,
    Message,
    MessageInteraction,
    TextChannel,
    Thread,
    User,
)
from disnake.ext import commands, tasks
from Levenshtein import distance as lev_distance
from shimeji import ChatBot
from shimeji.model_provider import OobaModel
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.preprocessor import ContextPreprocessor
from shimeji.util import (
    INSERTION_TYPE_NEWLINE,
    TRIM_DIR_NONE,
    TRIM_DIR_TOP,
    TRIM_TYPE_NEWLINE,
    ContextEntry,
)
from sqlalchemy import select
from sqlalchemy.orm import lazyload, load_only
from transformers import AutoTokenizer

import logsnake
from ai.eyes import DiscoEyes
from ai.imagen import Imagen
from ai.settings import (
    AI_DATA_DIR,
    AI_LOG_DIR,
    AI_LOG_FORMAT,
    BotMode,
    LMApiConfig,
    Prompt,
    ResponseMode,
    get_ai_settings,
)
from ai.tokenizers import PreTrainedTokenizerBase, extract_tokenizer
from ai.types import MessageChannel
from ai.ui import AiStatusEmbed
from ai.utils import (
    any_in_text,
    convert_mentions_emotes,
    get_full_class_name,
    get_lm_prompt_time,
    member_in_any_role,
    restore_mentions_emotes,
)
from ai.web import GradioUi
from cogs.privacy import PrivacyEmbed, PrivacyView, get_policy_text
from db import DiscordUser, Session
from disco_snake import checks
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__name__)


COG_UID = "ai"

logger = logsnake.setup_logger(
    name=COG_UID,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=AI_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=AI_LOG_DIR.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=3,
    propagate=True,
)


re_angle_bracket = re.compile(r"\<(.*)\>", re.M)
re_user_token = re.compile(r"(<USER>|<user>|{{user}})")
re_bot_token = re.compile(r"(<BOT>|<bot>|{{bot}}|<CHAR>|<char>|{{char}})")
re_unescape_format = re.compile(r"\\([*_~`])")
re_strip_special = re.compile(r"[^a-zA-Z0-9]+", re.M)

re_linebreak_name = re.compile(r"(\n|\r|\r\n)(\S+): ", re.M)
re_start_expression = re.compile(r"^\s*[(\*]\w+[)\*]\s*", re.I + re.M)
re_upper_first = re.compile(r"^([A-Z]\s?[^A-Z])")
re_detect_url = re.compile(
    r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    re.M + re.I,
)


def re_match_lower(match: re.Match):
    """function for re.sub() to convert the first match to lowercase"""
    return match.group(1).lower()


async def autocomplete_params(ctx, string: str) -> list[str]:
    return [param for param in Ai.SET_PARAMS if string.lower() in param.lower()]


class Ai(commands.Cog, name=COG_UID):
    SET_PARAMS: ClassVar[List[str]] = [
        "Temperature",
        "Top P",
        "Top K",
        "Typical P",
        "Rep P",
        "Min Length",
        "Max Length",
    ]

    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.timezone = self.bot.timezone
        self.last_response = datetime.now(timezone.utc) - timedelta(minutes=10)

        # Load config file
        self.config = get_ai_settings()

        # Parse config file
        self.provider_config: LMApiConfig = self.config.model_provider

        # Load config params up into top level properties
        self.params = self.config.params
        self.prompt: Prompt = self.config.prompt
        self.prefix_user: str = self.prompt.prefix_user
        self.prefix_bot: str = self.prompt.prefix_bot
        self.prefix_sep: str = self.prompt.prefix_sep

        self.activity_channels: List[int] = self.params.activity_channels
        self.conditional_response: bool = self.params.conditional_response
        self.context_size: int = self.params.context_size
        self.context_messages: int = self.params.context_messages
        self.idle_msg_sec: int = self.params.idle_messaging_interval
        self.idle_messaging: bool = self.params.idle_messaging
        self.logging_channel_id: int = self.params.logging_channel_id
        self.nicknames: List[str] = self.params.nicknames
        self.debug: bool = self.params.debug
        self.max_retries = self.params.max_retries
        self.ctxbreak = self.params.ctxbreak
        self.lm_lock = Lock()  # used to stop multiple responses from happening at once

        self.guilds: List[int] = self.params.guilds
        self.dm_user_ids: List[int] = [x.id for x in self.params.dm_users]
        self.dm_user_ids.extend(self.bot.config.admin_ids)
        self.ignored_user_ids: Set[int] = {}

        # bot user IDs that we're allowed to see/hear
        self.sibling_ids = [x.id for x in self.params.siblings]

        # we will populate these later during async init
        self.model_provider: OobaModel = None
        self.model_provider_name: str = self.provider_config.provider
        self.chatbot: ChatBot = None
        self.logging_channel: Optional[TextChannel] = None
        self.tokenizer_type = self.provider_config.modeltype
        self.tokenizer: PreTrainedTokenizerBase = None
        self.bad_words: List[str] = self.config.bad_words

        # selfietron
        self.imagen = Imagen(cog=self)
        self.eyes = DiscoEyes(cog=self)

        # gradio ui
        self.webui = GradioUi(cog=self, config=self.config.gradio)

        # somewhere to put the last context we generated for debugging
        self.debug_dir: Path = AI_LOG_DIR.joinpath("ai")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.joinpath("dm").mkdir(parents=True, exist_ok=True)

    # Getters for config object sub-properties
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def lm_gensettings(self):
        return self.provider_config.gensettings

    @tasks.loop(seconds=60)
    async def update_ignored(self):
        """Check the database for users who have rejected the ToS and get a list of their IDs for message filtering"""
        async with Session.begin() as session:
            query = (
                select(DiscordUser)
                .options(load_only("id", "tos_accepted", "tos_rejected", raiseload=True))
                .filter(DiscordUser.tos_rejected is True)
            )
            result = await session.scalars(query)
            users = result.all()
        self.ignored_user_ids.update([x.id for x in users])

    async def user_accepted_tos(self, user: Union[User, Member, int]) -> Optional[bool]:
        """Checks if a user has accepted the ToS, rejected, or not completed the process.
        Returns True if accepted, False if rejected, None if not completed.
        """
        user_id = user.id if isinstance(user, (User, Member)) else user
        async with Session.begin() as session:
            user: DiscordUser = await session.get(DiscordUser, user_id)
            if user.tos_rejected is True:
                return False
            if user.tos_accepted is True:
                return True
            if user is None:
                logger.debug(f"User {user_id} has not completed the ToS process.")
                return None

    async def tos_accepted_users(self) -> Set[int]:
        async with Session.begin() as session:
            query = (
                select(DiscordUser)
                .options(load_only("id", "tos_accepted", "tos_rejected", raiseload=True))
                .filter(DiscordUser.tos_accepted is True)
            )
            result = await session.scalars(query)
            users = result.all()
        return {x.id for x in users}

    async def cog_load(self) -> None:
        logger.info("AI engine initializing, please wait...")

        logger.debug("Initializing Tokenizer...")
        tokenizer_dir = AI_DATA_DIR.joinpath("tokenizers/llama")
        if not tokenizer_dir.exists():
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
        if not tokenizer_dir.joinpath("tokenizer.json").exists():
            extract_tokenizer(tokenizer_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_dir,
            local_files_only=True,
            use_fast=True,
        )
        logger.debug("Initializing Model Provider...")
        if self.model_provider_name == "ooba":
            self.model_provider = OobaModel(
                endpoint_url=self.provider_config.endpoint,
                args=self.provider_config.gensettings,
                tokenizer=self.tokenizer,
            )
        else:
            raise ValueError(f"Unknown model provider type: {self.model_provider_name}")

        logger.debug("Initializing ChatBot object")
        self.chatbot = ChatBot(
            name=self.name,
            model_provider=self.model_provider,
            preprocessors=[],
            postprocessors=[NewlinePrunerPostprocessor()],
        )

        logger.debug("Setting logging channel...")
        _logchannel = self.bot.get_channel(self.logging_channel_id)
        self.logging_channel = _logchannel if isinstance(_logchannel, (TextChannel, DMChannel)) else None

        logger.info("Starting WebUI (if enabled)...")
        await self.webui.launch()

        logger.debug("DMs are enabled for the following users:")
        for user_id in self.dm_user_ids:
            logger.debug(f" - {user_id}")

        if not self.update_ignored.is_running():
            logger.debug("Starting update_ignored task...")
            self.update_ignored.start()

        logger.info("AI engine initialized... probably?")
        if self.idle_messaging is True:
            logger.info("Starting idle messaging loop")
            self.idle_loop.start()
        if not self.log_archive_loop.is_running():
            logger.info("Starting log archive loop")
            self.log_archive_loop.start()

    @commands.Cog.listener("on_ready")
    async def on_ready(self):
        if self.logging_channel is None and self.logging_channel_id is not None:
            self.logging_channel = self.bot.get_channel(self.logging_channel_id)
            logger.info(f"Logging channel: {self.logging_channel}")
        logger.info("Cog is ready.")

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if (message.author.bot is True) or (message.author == self.bot.user):
            return
        if message.author.id in self.ignored_user_ids:
            return
        if "```" in message.content:
            return
        if message.content.strip() == ".":
            return
        if message.content.startswith("-") or message.content.startswith(", "):
            return

        direct_message = isinstance(message.channel, (DMChannel, GroupChannel))
        guild_settings = self.params.get_guild_settings(message.guild.id)

        # Ignore messages from unapproved users in DMs/groups
        if direct_message:
            if message.author.id not in self.dm_user_ids:
                logger.debug(f"DM from {message.author} (ID {message.author.id}) ignored")
                return
        # Ignore threads and messages with no guild (e.g. system messages)
        elif isinstance(message.channel, Thread) or message.guild is None:
            return

        if not direct_message:
            if guild_settings is None:
                return  # ignore messages from guilds that don't have settings
            if guild_settings.enabled is False:
                return  # ignore messages from guilds that have the bot disabled
            if guild_settings.channel_enabled(message.channel.id) is False:
                return  # ignore messages from channels that have the bot disabled
            elif message.channel.permissions_for(message.guild.me).send_messages is False:
                # channel is enabled but we don't have permission to respond
                logger.warn(
                    f"Got message in enabled channel {message.channel} but don't have permission to respond"
                )
                return

        # Now that we've filtered out messages we don't care about, let's get the message content
        message_content = self.get_msg_content_clean(message)
        if message_content == "":
            return  # ignore empty messages

        # Check if we're mentioned and check if the user has accepted the ToS
        mentioned = self.check_mention(message)
        tos_send_request = False
        tos_accepted = await self.user_accepted_tos(message.author)
        if tos_accepted is False:
            return  # ignore messages from users who have rejected the ToS
        if tos_accepted is None:
            tos_send_request = True  # send a ToS request if the user hasn't accepted or rejected yet

        trigger = None  # the trigger that caused the bot to respond
        conversation = None  # the conversation that the bot will respond to

        try:
            async with self.lm_lock:
                if mentioned:
                    logger.debug(f"Mentioned in {message.channel}: {message_content}")
                    trigger = "mention"

                elif direct_message and len(message_content) > 0:
                    logger.debug(f"DM from {message.author}: {message_content}")
                    trigger = "DM"

                # elif message.channel.id in self.activity_channels:
                #     if self.conditional_response is True:
                #         conversation = await self.get_message_context(message)
                #         conv_str = "\n".join(conversation)
                #         if await self.model_provider.should_respond_async(
                #             (f"{conv_str}"), self.name, f"\n{self.prefix_bot}"
                #         ):
                #             logger.debug(f"Model wants to respond to '{message_content}', responding...")
                #             trigger = "conditional"

                #     elif (
                #         self.last_response
                #         < (datetime.now(tz=self.timezone) - timedelta(seconds=(self.idle_msg_sec / 2)))
                #         and self.idle_messaging is True
                #     ):
                #         # prevent infinite loops if things go wrong
                #         self.last_response = datetime.now(tz=self.timezone)
                #         trigger = "activity"

                if trigger is not None:
                    if conversation is None:
                        conversation = await self.get_message_context(message)

                    if tos_send_request is True:
                        await self.do_response(conversation, message, trigger, "Please check your DMs!")
                        await self.check_send_tos(message)
                    else:
                        await self.do_response(conversation, message, trigger)

        except Exception as e:
            logger.exception(e)
            exc_class = get_full_class_name(e)
            if exc_class == "HTTPException":
                # don't bother, the backend is down
                return

            exc_desc = str(f"**``{exc_class}``**\n```{format_exc()}```")
            error_file = self.debug_dir.joinpath(f"error-{datetime.now(timezone.utc)}.txt".replace(" ", ""))
            error_file.write_text(exc_desc)

            if len(exc_desc) < 2048:
                embed = Embed(title="**Exception**", description=f"**``{exc_class}``**")
                if self.logging_channel is not None:
                    await self.logging_channel.send(embed=embed)
                else:
                    await message.channel.send(embed=embed, delete_after=60.0)
            else:
                embed = Embed(
                    title="**Exception**",
                    description="Exception too long for message, see attached file.",
                )
                if self.logging_channel is not None:
                    await self.logging_channel.send(embed=embed, file=File(error_file))
                else:
                    await message.channel.send(embed=embed, file=File(error_file), delete_after=60.0)

    # retrieve the LM prompt and inject name, time, etc.
    def get_prompt(self, ctx: Optional[MessageInteraction] = None, include_model: bool = False) -> str:
        if ctx is None:
            location_context = "and friends in a Discord server"
        elif hasattr(ctx, "guild") and ctx.guild is not None:
            location_context = f'and friends in the "{ctx.guild.name}" Discord server'
        elif hasattr(ctx, "author") and ctx.author is not None:
            location_context = f"and {ctx.author.display_name} in a Discord DM"
        else:
            location_context = "and a friend in a Discord DM"

        prompt = []
        prompt.append(self.prompt.system.full)
        prompt.append(self.prompt.character.full)
        if include_model:
            prompt.append(self.prompt.model.full.rstrip())
        prompt = "\n".join(prompt)

        replace_tokens = {
            "bot_name": self.name,
            "location_context": location_context,
            "current_time": get_lm_prompt_time(),
        }
        for token, replacement in replace_tokens.items():
            prompt = prompt.replace(f"{{{token}}}", replacement)
        return prompt

    # get the last N messages in a channel, up to and including the message that triggered the response
    async def get_message_context(
        self,
        message: Message,
        limit: int = 50,
        as_list: bool = True,
    ) -> Union[str, List[str]]:
        messages = await message.channel.history(limit=limit, before=message).flatten()
        messages.insert(0, message)  # add the message that triggered the response

        # magic tag to break context chain to get bot out of librarian mode
        for idx, message in enumerate(messages):
            if self.is_ctxbreak_msg(message):
                logger.debug("Found context break tag, breaking context chain")
                messages = messages[0 : idx + 1]
                break

        bot_mode = self.params.get_guild_settings(message.guild.id).channel_bot_mode(message.channel.id)

        chain = []
        for msg in reversed(messages):
            if msg.author.id in self.ignored_user_ids:
                continue  # skip users who rejected the privacy policy
            if msg.content is not None:
                if msg.content.startswith("-") or msg.content.startswith(", "):
                    logger.debug("skipping other bot command message")
                    continue
                if len(msg.content) > 300 and any_in_text(
                    ["you have 10 tokens", 'which stands for "do anything now"'], msg.content.lower()
                ):
                    logger.debug("Skipping long message containing AI jailbreak bullshit")
                    continue
                if msg.content.startswith("DAN:"):
                    logger.debug("Skipping stupid DAN message (looking at u technocat)")
                    continue

            if msg.author.id == self.bot.user.id:
                if msg.content.startswith("This is an AI chatbot made by"):
                    continue  # this is a TOS message in a DM so lets skip it
            elif msg.author.bot is True:  # bots who aren't us
                if bot_mode == BotMode.Strip:
                    continue  # skip all other bot messages if we're in strip mode
                elif bot_mode == BotMode.Siblings:
                    if msg.author.id not in self.sibling_ids:
                        continue  # skip non-sibling bot messages

            # set up author name
            author_name = f"{self.prefix_user}{self.prefix_sep}{msg.author.display_name.strip()}:"

            if len(msg.embeds) > 0:
                for embed in msg.embeds:
                    if embed.type == "image":
                        try:
                            caption = await self.eyes.perceive_url(embed.thumbnail.url)
                            if caption is not None:
                                chain.append(f"{author_name} [image: {caption}]")
                        except Exception as e:
                            logger.exception(e)
                            chain.append(f"{author_name} [image: loading error]")
                    elif embed.description is not None and msg.author.bot is False:
                        chain.append(f"{author_name} [embed: {embed.description}]")
                    elif embed.title is not None and msg.author.id != self.bot.user.id:
                        chain.append(f"{author_name} [embed: {embed.title}]")

            if msg.content and len(msg.embeds) == 0:
                content = self.get_msg_content_clean(msg)
                if content.startswith("http"):
                    chain.append(f"{author_name} [a link to a webpage]")
                elif content != "" and "```" not in content and not content.startswith("-"):
                    if msg.author.id == self.bot.user.id:
                        # strip (laughs) from start of own context
                        content = re_start_expression.sub("", content)
                    chain.append(f"{author_name} {content}")

            for attachment in msg.attachments:
                try:
                    if not attachment.content_type.startswith("image/"):
                        logger.debug(f"got non-image attachment: Content-Type {attachment.content_type}")
                        continue
                    caption = await self.eyes.perceive_attachment(attachment)
                    if caption is not None:
                        chain.append(f"{author_name} [image: {caption}]")
                except Exception as e:
                    logger.exception(e)
                    chain.append(f"{author_name} [image: loading error]")

        chain = [x.strip() for x in chain if x.strip() != ""]
        if chain[-1].lower().startswith(f"{self.prefix_user}{self.prefix_sep}{self.name}:".lower()):
            chain = chain[:-1]  # remove last message if it's from the bot, cheap hack for now
        return chain if as_list is True else ("\n".join(chain))

    # assemble a prompt/context for the model
    async def process_context(self, conversation: List[str], message: Message):
        contextmgr = ContextPreprocessor(token_budget=self.context_size, tokenizer=self.tokenizer)
        logger.debug(f"building context from {len(conversation)} messages")

        if self.prompt.instruct is True:
            post_instruct = conversation[-1:]
            conversation = "\n".join(
                [
                    "\n".join(conversation[:-1]),
                    "\n" + self.prompt.model.full.rstrip(),
                    "\n".join(post_instruct).replace(self.prefix_user, "").lstrip(),
                ]
            )
        else:
            conversation = "\n".join(conversation)

        prompt_entry = ContextEntry(
            text=self.get_prompt(message),
            prefix="",
            suffix="",
            reserved_tokens=512,
            insertion_order=1000,
            insertion_position=0,
            trim_direction=TRIM_DIR_NONE,
            trim_type=TRIM_TYPE_NEWLINE,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
            tokenizer=self.tokenizer,
        )
        contextmgr.add_entry(prompt_entry)

        conversation_entry = ContextEntry(
            text=conversation + f"\n{self.name}:",
            prefix="",
            suffix="",
            reserved_tokens=1024,
            insertion_order=500,
            insertion_position=-1,
            trim_direction=TRIM_DIR_TOP,
            trim_type=TRIM_TYPE_NEWLINE,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
            tokenizer=self.tokenizer,
        )
        contextmgr.add_entry(conversation_entry)

        context = contextmgr.context(self.context_size)
        return context.replace("\n\n", "\n").replace("\n\n", "\n")

    # actual response logic
    async def do_response(
        self,
        conversation: List[str],
        message: Message,
        trigger: Optional[str] = None,
        append: Optional[str] = None,
    ) -> str:
        async with message.channel.typing():
            debug_data: Dict[str, Any] = {}

            response = ""
            response_image = None
            should_reply = False
            author_name = f"{message.author.display_name.strip()}:"

            msg_timestamp = message.created_at.astimezone(tz=self.bot.timezone).strftime(
                "%Y-%m-%d-%H:%M:%S%z"
            )
            msg_trigger = trigger.lower() if trigger is not None else "unknown"

            debug_data["message"] = {
                "id": message.id,
                "author": f"{message.author}",
                "author_name": f"{author_name}",
                "guild": f"{message.guild}" if message.guild is not None else "DM",
                "channel": message.channel.name if not isinstance(message.channel, DMChannel) else "DM",
                "timestamp": msg_timestamp,
                "trigger": msg_trigger,
                "content": message.content,
                "conversation": conversation,
            }

            try:
                debug_data["gensettings"] = self.provider_config.gensettings.dict()
            except Exception as e:
                logger.exception("Failed to get gensettings")

            # Build context
            context = await self.process_context(conversation, message)

            # war crime for alpaca, needs more newlines
            # context_lines = context.splitlines()
            # new_lines = []
            # first = True
            # for line in context_lines:
            #     if line.startswith("### "):
            #         _, content = line.split(":", 1)
            #         if line.startswith(self.prefix_bot) and line != self.prefix_bot:
            #             if first is True:
            #                 # strip empty newline at end of prompt
            #                 new_lines = new_lines[:-1]
            #                 # add response without extra newline
            #                 new_lines.append(f"{self.prefix_bot}{self.prefix_sep}{content.strip()}")
            #                 first = False
            #                 continue
            #             else:
            #                 # newline then response
            #                 new_lines.append(f"{self.prefix_bot}{self.prefix_sep}{content.strip()}")
            #                 continue
            #         elif line.startswith(self.prefix_user) and line != self.prefix_user:
            #             if first is True:
            #                 new_lines.append(f"{self.prefix_user}{self.prefix_sep}{content.strip()}")
            #                 first = False
            #                 continue
            #             else:
            #                 new_lines.append(f"{self.prefix_user}{self.prefix_sep}{content.strip()}")
            #                 continue
            #     new_lines.append(line)
            # context = "\n".join(new_lines).replace("\n\n\n", "\n\n")
            context = context.rstrip()
            debug_data["context"] = context.splitlines()

            try:
                # Generate the response, and retry if it contains bad words (up to self.max_retries times)
                for attempt in range(self.max_retries):
                    attempt = attempt + 1  # deal with range() starting from 0
                    response: str = await self.chatbot.respond_async(context, push_chain=False)
                    bad_words = self.find_bad_words(response)
                    if any([response.lower() == x for x in self.bad_words]):
                        logger.info(f"Response {attempt} contained bad words: {response}\nRetrying...")
                        continue
                    if len(bad_words) == 0:
                        if response.lower().startswith("i'm sorry, but"):
                            logger.info(f"Response was a ChatGPT apology: {response}\nRetrying...")
                            continue
                        if "as a language model" in response.lower():
                            logger.info(f"Response admits to being an AI: {response}\nRetrying...")
                            continue
                        if re_detect_url.search(response):
                            logger.info(f"Response contains a URL: {response}\nRetrying...")
                            continue
                        break  # no bad words, we're good, break out of the loop to avoid executing the else:
                    else:
                        logger.info(
                            f"Response {attempt} contained bad words: {response}\nBad words: {bad_words}\nRetrying..."
                        )
                        continue
                else:  # ran out of retries...
                    if response_image is not None:
                        response = ""  # we have a pic to send, so send it without a comment
                    else:
                        logger.warn(
                            f"Final response contained bad words: {response}\nBad words: {bad_words}\nRetrying..."
                        )
                        response = ""

                # see if there's an image description in the response
                description, response = self.imagen.find_image_desc(response)
                if description is not None:
                    # there is, so we should reply with the image
                    logger.info("i'm feeling creative, let's make an image...")
                    try:
                        response_image = await self.take_pic(message=description)
                        should_reply = True
                    except Exception as e:
                        raise Exception("Failed to generate image response") from e

                # if we haven't described an image, but the user asked for one, send one anyway
                elif self.imagen.should_take_pic(message.content):
                    logger.info("the people have spoken and they want catgirl selfies")
                    try:
                        response_image = await self.take_pic(message=message)
                        should_reply = True
                    except Exception as e:
                        raise Exception("Failed to generate image response") from e

                debug_data["response_raw"] = response

                # if bot did a "\n<someusername:" cut it off
                if bool(re_linebreak_name.match(response)):
                    response = response.splitlines()[0]

                # replace "<USER>" with user mention, same for "<BOT>"
                response = self.fixup_bot_user_tokens(response, message)

                # Clean response - trim left whitespace and fix emojis and pings
                # response = cut_trailing_sentence(response)
                response = response.lstrip()

                if isinstance(message.channel, (TextChannel, Thread)):
                    members = (
                        message.channel.guild.members
                        if message.channel.guild is not None and isinstance(message.channel, Thread)
                        else message.channel.members
                    )
                    response = restore_mentions_emotes(
                        text=response,
                        users=members,
                        emojis=self.bot.emojis,
                    )

                # trim hashes n shit
                response = response.rstrip("\"' #}").lstrip("\"'")

                if append is not None and len(append.strip()) > 0:
                    response = f"{response} < {append.strip()} >"
                debug_data["response"] = response

                # Send response if not empty
                if response_image is not None:
                    if response == "":
                        logger.info("Response was empty, sending image only.")
                        if should_reply:
                            await message.reply(file=response_image)
                        else:
                            await message.channel.send(file=response_image)
                    else:
                        logger.info(f"Responding with image, response: {response}")
                        if should_reply:
                            await message.reply(response, file=response_image)
                        else:
                            await message.channel.send(response, file=response_image)
                elif response == "":
                    logger.info("Response was empty.")
                    await message.add_reaction("🤷‍♀️")
                else:
                    await message.channel.send(response)
                    logger.info(f"Response: {response}")

                    self.last_response = datetime.now(timezone.utc)

                if self.webui is not None:
                    logger.debug("Updating webui")
                    self.webui.lm_update(
                        prompt=context,
                        message=f"{author_name} {debug_data['message']['content']}",
                        response=debug_data["response_raw"].lstrip(),
                    )

            except Exception as e:
                logger.exception(e)
            finally:
                if self.debug:
                    dump_file = f"msg-{message.id}-{msg_timestamp}.json"
                    if debug_data.get("trigger", None) == "DM":
                        dump_file = self.debug_dir.joinpath("dm", dump_file)
                    else:
                        dump_file = self.debug_dir.joinpath(dump_file)
                    with dump_file.open("w", encoding="utf-8") as f:
                        json.dump(debug_data, f, indent=4, skipkeys=True, default=str, ensure_ascii=False)
                    logger.debug(f"Dumped message debug data to {dump_file.name}")

    # Idle loop stuff
    @tasks.loop(seconds=21)
    async def idle_loop(self) -> None:
        if self.idle_messaging is True:
            # get last message in a random priority channel
            channel = self.bot.get_channel(random_choice(self.activity_channels))
            if isinstance(channel, (TextChannel, DMChannel, GroupChannel)):
                messages = await channel.history(limit=1).flatten()
                message: Message = messages.pop()
                if message.author.bot is True:
                    return

                idle_sec = (datetime.now(tz=timezone.utc) - message.created_at).total_seconds()
                if idle_sec >= self.idle_msg_sec:
                    if self.get_msg_content_clean(message) == "":
                        return
                    logger.debug(f"Running idle response to message {message.id}")
                    # if it's been more than <idle_messaging_interval> sec, send a response
                    async with self.lm_lock:
                        conversation = await self.get_message_context(message)
                        await self.do_response(conversation, message)
        else:
            return

    @idle_loop.before_loop
    async def before_idle_loop(self):
        logger.info("Idle loop waiting for bot to be ready")
        await self.bot.wait_until_ready()
        await asyncio.sleep(10)
        logger.info("Idle loop running!")

    ## Helper functions
    # check if the person triggering the bot has agreed to ToS or not and send it if not
    async def check_send_tos(self, message: Message) -> bool:
        logger.debug(f"Checking ToS for {message.author} ({message.author.id})")
        try:
            async with Session.begin() as session:
                user: DiscordUser = await session.get(DiscordUser, message.author.id)
                if user is None:
                    user = DiscordUser.from_discord(message.author)
                    await session.merge(user)
                    await session.commit()
                    user = await session.get(DiscordUser, message.author.id)
                if user is None:
                    raise Exception("Failed to create or retrieve user")

            # send the form in a DM if they haven't accepted or rejected it yet
            if user.tos_accepted is not True:
                logger.debug(f"Sending privacy embed to {message.author} ({message.author.id})")
                invite = await self.bot.support_invite()
                embed = PrivacyEmbed(
                    author=message.author, support_guild=self.bot.support_guild, user=user, invite=invite
                )
                view = PrivacyView(user=user)
                content = get_policy_text(self.bot)
                view.message = await message.author.send(content=content, embed=embed, view=view)
                logger.debug(f"Sent privacy embed to {message.author} ({message.author.id})")
                return False
            else:
                logger.debug(f"User {message.author} ({message.author.id}) has accepted ToS")
                return True
        except Exception as e:
            logger.exception("error checking tos")
            raise e

    # clean up a message's content
    def get_msg_content_clean(self, message: Message, content: Optional[str] = None) -> str:
        content = content if content is not None else message.content

        if isinstance(message.channel, Thread):
            member_ids = [x.id for x in message.channel.members]
            content = convert_mentions_emotes(
                text=content,
                users=self.bot.loop.run_until_complete(
                    message.channel.guild.get_or_fetch_members(member_ids)
                ),
                emojis=self.bot.emojis,
            )
        elif isinstance(message.channel, (DMChannel, GroupChannel)):
            content = convert_mentions_emotes(
                text=content,
                users=[message.channel.me, message.channel.recipient, message.author],
                emojis=self.bot.emojis,
            )
        else:
            content = convert_mentions_emotes(
                text=content,
                users=message.channel.members or message.mentions,
                emojis=self.bot.emojis,
            )

        # eliminate any and all content between angle brackets <>
        content = re_angle_bracket.sub("", content)
        # turn newlines into spaces. this is a cheap hack but the model doesn't handle newlines well
        content = content.replace("\n", " ")
        # if there's a codeblock in there, return an empty string
        return content.lstrip() if "```" not in content else ""

    # clean out some bad tokens from responses and fix up mentions
    def fixup_bot_user_tokens(self, response: str, message: Message) -> str:
        """
        Fix <USER>, <BOT>, etc tokens in the response, and unescape any escaped markdown formatting
        """
        author_name = message.author.display_name.encode("utf-8").decode("ascii", errors="ignore").strip()
        response = re_user_token.sub(f"@{author_name}", response)
        response = re_bot_token.sub(f"@{self.name}", response)
        response = re_unescape_format.sub(r"\1", response)
        return response

    # search for bad words in a message
    def find_bad_words(self, input: str) -> str:
        found_words: List[str] = []
        input_words: str = input.split()

        for word in input_words:
            word_stripped: str = re_strip_special.sub("", word).lower()
            word_len = len(word_stripped)
            if word_len > 2:
                threshold = 2 if word_len > 6 else 1 if word_len > 4 else 0
                for bad_word in self.bad_words:
                    if lev_distance(word_stripped, bad_word) <= threshold:
                        found_words.append(f"{word} (like {bad_word})")
        if len(found_words) > 0:
            logger.warn(f"Found bad words: {' | '.join(found_words)}")
        return found_words

    # check if the bot's name or a nickname is in the message
    def name_in_text(self, text: str) -> bool:
        return any(
            [
                bool(re.search(rf"\b({name})({name[-1]}*)\b", text, re.I + re.M))
                for name in self.params.nicknames
            ]
        )

    # check if the bot was mentioned in a message
    def check_mention(self, message: Message) -> bool:
        """Checks if the bot was mentioned in a message"""
        if message.guild is None:
            return False
        if self.bot.user in message.mentions or message.guild.me in message.mentions:
            return True
        if self.name_in_text(message.content):
            return True
        return False

    # check if a given message is a context break message
    def is_ctxbreak_msg(self, message: Message) -> bool:
        content = message.content.lower()
        if not content.startswith("<ctxbreak>"):
            return False  # not a context break message
        if isinstance(message.channel, DMChannel):
            return True  # people can always break context in DMs
        if message.author.id in list(self.ctxbreak.user_ids + self.bot.config.admin_ids):
            return True  # admins and ctxbreak users can always break context
        if member_in_any_role(message.author, self.ctxbreak.role_ids):
            return True  # ctxbreak role havers can always break context
        return False  # otherwise, no

    # Loop to archive debug logs
    @tasks.loop(hours=1)
    async def log_archive_loop(self) -> None:
        """
        Move message log files over 24h old to the archive directory
        """
        for file in self.debug_dir.glob("*.json"):
            file_age = (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).total_seconds()
            if file_age >= (3600 * 24):
                logger.debug(f"Archiving debug log {file.name}")
                file = file.rename(file.parent / "archive" / file.name)

    # Image generation stuff
    async def take_pic(self, message: Union[Message, str]) -> Tuple[File, dict]:
        """
        hold up, let me take a selfie
        """
        # get the message content
        if isinstance(message, Message):
            message_content = self.get_msg_content_clean(message).lower()
        else:
            message_content = message.lower()

        if message_content == "":
            logger.debug("Message was empty after cleaning.")
            return ""

        # get the image description and remove newlines
        message_content = message_content.replace("\n", " ").replace(self.name + " ", "")

        # build the LLM prompt for the image
        lm_trigger = self.imagen.strip_take_pic(message_content)
        lm_prompt = self.imagen.get_lm_prompt(lm_trigger)
        logger.info(f"LLM Prompt: {lm_prompt}")

        # get the LLM to create tags for the image
        lm_tags = await self.imagen.submit_lm_prompt(lm_prompt)
        lm_tags = lm_tags.strip('",').lower()
        logger.info(f"LLM Tags: {lm_tags}")

        # build the SD API request
        sdapi_request = self.imagen.build_request(lm_tags, message_content)
        logger.info(f"SD API Request: {json.dumps(sdapi_request)}")

        # submit it
        result_path = await self.imagen.submit_request(sdapi_request)
        # return a discord File object to upstream
        result_file = File(result_path, filename=result_path.name)
        if self.webui.config.enabled:
            try:
                self.webui.imagen_update(lm_trigger, lm_tags, result_path)
            except Exception as e:
                logger.exception(e)
        return result_file

    # UI stuff
    @commands.slash_command(name="ai", description="AI Management")
    @checks.not_blacklisted()
    async def ai_group(self, ctx: ApplicationCommandInteraction):
        pass

    @ai_group.sub_command(name="status", description="Get AI status")
    async def ai_status(
        self,
        ctx: ApplicationCommandInteraction,
        verbose: bool = commands.Param(default=False, name="verbose", description="Verbose output"),
    ):
        embed = AiStatusEmbed(self, ctx.author, verbose=verbose)
        await ctx.send(embed=embed)

    @ai_group.sub_command(name="set", description="Set AI parameters")
    @checks.is_admin()
    async def set_parameter(
        self,
        ctx: ApplicationCommandInteraction,
        param: str = commands.Param(..., autocomplete=autocomplete_params),
        value: str = commands.Param(...),
    ) -> None:
        await ctx.response.defer(ephemeral=True)
        embed = Embed(title="Set Parameter", description="Success!", color=Colour.green())
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
        embed.add_field(name="Parameter", value=f"{param}", inline=False)

        try:
            match param:
                case "Temperature":
                    new_value = float(value)
                    old_value = self.lm_gensettings.temperature
                    self.provider_config.gensettings.temperature = new_value
                case "Top P":
                    new_value = float(value)
                    old_value = self.lm_gensettings.top_p
                    self.provider_config.gensettings.top_p = new_value
                case "Top K":
                    new_value = int(value)
                    old_value = self.lm_gensettings.top_k
                    self.provider_config.gensettings.top_k = new_value
                case "Typical P":
                    new_value = float(value)
                    old_value = self.lm_gensettings.typical_p
                    self.provider_config.gensettings.typical_p = new_value
                case "Rep P":
                    new_value = float(value)
                    old_value = self.lm_gensettings.repetition_penalty
                    self.provider_config.gensettings.repetition_penalty = new_value
                case "Min Length":
                    new_value = int(value)
                    old_value = self.lm_gensettings.min_length
                    self.provider_config.gensettings.min_length = new_value
                case "Max Length":
                    new_value = int(value)
                    old_value = self.lm_gensettings.max_new_tokens
                    self.provider_config.gensettings.max_new_tokens = new_value
                case _:
                    raise ValueError(f"Unknown parameter: {param}")

            embed.add_field(name="Old Value", value=f"{old_value}")
            embed.add_field(name="New Value", value=f"{new_value}")

        except Exception as e:
            logger.exception(e)
            embed.description = "Failed!"
            embed.color = Colour.red()
            if isinstance(e, ValueError):
                embed.add_field(name="Error", value=f"Invalid value: {value}", inline=False)
            else:
                embed.add_field(name="Error", value="An unknown error occurred", inline=False)
            embed.add_field(name="Exception", value=e, inline=False)
        finally:
            await ctx.send(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Ai(bot))
