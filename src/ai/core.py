# ruff: noqa: E712
import asyncio
import json
import logging
import re
from asyncio import Lock
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, Optional, Tuple, Union

import jinja2 as j2
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
from fastapi import HTTPException
from Levenshtein import distance as lev_distance
from sqlalchemy import select
from sqlalchemy.orm import load_only
from transformers import AutoTokenizer, BatchEncoding

import logsnake
from ai.eyes import DiscoEyes
from ai.imagen import Imagen
from ai.settings import (
    AI_DATA_DIR,
    AI_LOG_DIR,
    AI_LOG_FORMAT,
    BotMode,
    BotParameters,
    GuildSettings,
    GuildSettingsList,
    LMApiConfig,
    NamedSnowflake,
    Prompt,
    get_ai_settings,
)
from ai.tokenizers import PreTrainedTokenizerBase, extract_tokenizer
from ai.tokenizers.yi.tokenization_yi import YiTokenizer
from ai.types import LruDict
from ai.ui import AiParam, AiStatusEmbed, set_choices, settable_params
from ai.utils import (
    MentionMixin,
    get_prompt_datetime,
    member_in_any_role,
)
from ai.web import GradioUi
from cogs.privacy import PrivacyEmbed, PrivacyView, get_policy_text
from db import DiscordUser, Session, SessionType
from db.ai import AiResponseLog
from disco_snake import checks
from disco_snake.blacklist import Blacklist
from disco_snake.bot import DiscoSnake
from shimeji.chatbot import ChatBot
from shimeji.model_provider import OobaGenRequest, OobaModel
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.preprocessor import ContextPreprocessor
from shimeji.util import BreakType, ContextEntry, TrimDir

COG_UID = "ai"

ai_logger = logsnake.setup_logger(
    name=COG_UID,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=AI_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=AI_LOG_DIR.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=1,
    propagate=True,
)
logger = logging.getLogger(__name__)


re_angle_bracket = re.compile(r"\<(.*)\>", re.M)
re_user_token = re.compile(r"(<USER>|<user>|{{user}})")
re_bot_token = re.compile(r"(<bot>|{{bot}}|<char>|{{char}}|<assistant>|{{assistant}})", re.I)
re_unescape_md = re.compile(r"\\([*_~`])")
re_nonword = re.compile(r"[^a-zA-Z0-9]+", re.M + re.I)
re_nonword_end = re.compile(r"([^a-zA-Z0-9])[^a-zA-Z0-9()]+$", re.M + re.I)

re_linebreak_name = re.compile(r"(\n|\r|\r\n)(\S+): ", re.M)
re_start_expression = re.compile(r"^\s*[(\*]\w+[)\*]\s*", re.I + re.M)
re_upper_first = re.compile(r"^([A-Z]\s?[^A-Z])")
re_detect_url = re.compile(
    r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    re.M + re.I,
)

# find consecutive newlines (at least 2) optionally with spaces in between (blank lines)
re_consecutive_newline = re.compile(r"(\n[\s\n]*\n\s*)", re.M + re.I)

# capture mentions and emojis
re_mention = re.compile(r"<@(\d+)>", re.I)
re_emoji = re.compile(r"<:([^:]+):(\d+)>", re.I)


def re_match_lower(match: re.Match):
    """function for re.sub() to convert the first match to lowercase"""
    return match.group(1).lower()


def available_params(ctx: ApplicationCommandInteraction) -> list[str]:
    return [param.name for param in settable_params]


def convert_param(ctx: ApplicationCommandInteraction, input: str) -> AiParam:
    return next((param for param in settable_params if param.name == input or param.id == input))


class Ai(MentionMixin, commands.Cog, name=COG_UID):
    _mention_cache: LruDict
    _emoji_cache: LruDict

    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.timezone = self.bot.timezone
        self.last_response = datetime.now(timezone.utc) - timedelta(minutes=10)

        # init the MentionMixin cache
        super(MentionMixin, self).__init__()

        # Load config file
        self.config = get_ai_settings()
        self.lm_lock = Lock()  # used to stop multiple responses from happening at once

        # Parse config file
        self.provider_config: LMApiConfig = self.config.model_provider

        # Load config params up into top level properties
        self.params: BotParameters = self.config.params
        self.nicknames: list[str] = self.params.nicknames

        self.prompt: Prompt = self.config.prompt
        self.max_retries = self.params.max_retries

        self.idle_enable: bool = self.params.idle_enable
        self.idle_channels: list[int] = self.params.get_idle_channels()

        self.ctxbreak = self.params.ctxbreak
        self.guilds: GuildSettingsList = self.params.guilds
        self.dm_user_ids: list[int] = self.params.dm_user_ids
        self.dm_user_ids.extend(self.bot.config.admin_ids)
        self.tos_reject_ids: set[int] = set()

        # we will populate these later during async init
        self.model_provider: OobaModel = None
        self.model_provider_name: str = self.provider_config.provider
        self.chatbot: ChatBot = None
        self.logging_channel: Optional[TextChannel] = None
        self.tokenizer_type = self.provider_config.modeltype
        self.tokenizer: PreTrainedTokenizerBase = None
        self.bad_words: list[str] = self.config.bad_words

        # database client (async init)
        self.db_client: SessionType = Session
        self.blacklist: Blacklist = Blacklist()

        ## addon modules
        self.imagen = Imagen(cog=self)  # selfietron
        self.eyes = DiscoEyes(cog=self)  # image caption engine
        self.webui = GradioUi(cog=self, config=self.config.gradio)  # gradio ui

        # debugging related
        self.debug: bool = self.params.debug
        self.debug_dir: Path = AI_LOG_DIR.joinpath("ai")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.joinpath("dm").mkdir(parents=True, exist_ok=True)
        self._last_debug_log: dict[str, Any] = {}
        self.logging_channel_id: int = self.params.logging_channel_id

        # cache dicts and props
        self._trigger_cache: LruDict = LruDict(max_size=100)
        self._mention_cache: dict[int, Any] = {}
        self._emoji_cache: dict[int, Any] = {}
        self._n_prompt_tokens: int = None
        # other cached vars
        self.j2_env: j2.Environment = j2.Environment(
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
            cache_size=10,
        )
        self._chat_template: Optional[j2.Template] = None

    # Getters for config object sub-properties
    @property
    def name(self) -> str:
        return self.config.name.split(" ")[0]

    @property
    def lm_gensettings(self) -> OobaGenRequest:
        return self.provider_config.gensettings

    @property
    def context_size(self) -> int:
        if self.params.context_size < 0:
            return self.lm_gensettings.truncation_length - self.lm_gensettings.max_tokens - 32
        return self.params.context_size

    @property
    def siblings(self) -> list[NamedSnowflake]:
        return self.params.siblings

    @property
    def sibling_ids(self) -> list[int]:
        return self.params.siblings.ids

    @property
    def n_prompt_tokens(self):
        if self._n_prompt_tokens is None:
            prompt_str = self.get_prompt()
            if isinstance(prompt_str, list):
                prompt_str = "\n".join(prompt_str)
            encoded: BatchEncoding = self.tokenizer.batch_encode_plus([prompt_str], return_length=True)
            n_tokens = encoded.get("length", [0])[0]
            self._n_prompt_tokens = max(n_tokens + 64, 512)
        return self._n_prompt_tokens

    @property
    def chat_template(self) -> Optional[j2.Template]:
        if self._chat_template is None and self.prompt.chat_template_str is not None:
            logger.debug("Compiling chat template...")
            self._chat_template = self.j2_env.from_string(self.prompt.chat_template_str)
            logger.debug("Template compiled")
        return self._chat_template

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        if self.chat_template is None:
            return "\n".join([x["content"] for x in messages])
        rendered = self.chat_template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return rendered

    async def cog_load(self) -> None:
        logger.info("AI engine initializing, please wait...")

        logger.info("Initializing Tokenizer...")
        tokenizer_dir = AI_DATA_DIR.joinpath(f"tokenizers/{self.tokenizer_type}")
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        if not tokenizer_dir.joinpath("tokenizer.json").is_file():
            extract_tokenizer(name=self.tokenizer_type, target_dir=tokenizer_dir)

        self.tokenizer: YiTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_dir,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=self.lm_gensettings.truncation_length,
        )

        logger.info("Initializing Model Provider...")
        if self.model_provider_name == "ooba":
            self.model_provider = OobaModel(
                endpoint_url=self.provider_config.endpoint,
                default_args=self.lm_gensettings,
                tokenizer=self.tokenizer,
                api_v2=self.provider_config.api_v2,
                debug=self.params.debug,
            )
        else:
            raise ValueError(f"Unknown model provider type: {self.model_provider_name}")

        logger.info("Initializing ChatBot object")
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

        logger.debug("Known sibling bots:")
        for sibling in self.siblings:
            logger.debug(f" - {sibling.id=} {sibling.name=}")

        if self.eyes is not None:
            logger.info("Starting DiscoEyes...")
            await self.eyes.start()

        if not self.update_ignored.is_running():
            logger.debug("Starting update_ignored task...")
            self.update_ignored.start()

        if self.idle_enable is True:
            logger.info("Starting idle messaging loop")
            self.idle_loop.start()

        if not self.log_archive_loop.is_running():
            logger.info("Starting log archive loop")
            self.log_archive_loop.start()

        _ = self.chat_template  # force template compilation

        logger.info("AI engine initialized... probably? yolo!")

    def cog_unload(self) -> None:
        if self.eyes is not None:
            logger.info("Shutting down DiscoEyes...")
            self.eyes.shutdown()

        if self.webui is not None:
            logger.info("Shutting down WebUI...")
            self.webui.shutdown()

        logger.info("AI engine unloaded.")

    @commands.Cog.listener("on_ready")
    async def on_ready(self):
        if self.logging_channel is None and self.logging_channel_id is not None:
            self.logging_channel = self.bot.get_channel(self.logging_channel_id)
            logger.info(f"Logging channel: {self.logging_channel}")

        logger.debug("DMs are enabled for the following users:")
        for user_id in self.dm_user_ids:
            try:
                user_obj = self.bot.get_user(user_id)
                logger.debug(f"  - {user_obj} ({user_id})")
            except Exception:
                logger.exception(f"Failed to get user object for {user_id}")
        # done
        logger.info("AI engine is ready, uwu~")

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if message.author.id == self.bot.user.id:
            return  # ignore messages from self
        if message.author.id in self.blacklist:
            return  # ignore messages from blacklisted users
        if message.author.id in self.tos_reject_ids:
            return  # ignore messages from users who have rejected the ToS
        if message.author.bot is True:
            if self.last_trigger_was_bot(message) is True:
                return  # ignore messages from bots if the last message we responded to was from a bot
        if message.content.startswith("-") or message.content.startswith(", "):
            return  # ignore messages that start with a dash or comma-space
        if message.content.strip() == ".":
            return  # ignore messages with only a period
        if "```" in message.content:
            return  # ignore messages with code blocks

        direct_message = isinstance(message.channel, DMChannel)
        guild_settings = self.params.get_guild_settings(message.guild.id) if message.guild else None

        # Ignore messages from unapproved users in DMs/groups
        if direct_message is True:
            if message.author.id not in self.dm_user_ids:
                logger.debug(f"DM from {message.author} (ID {message.author.id}) ignored")
                return
        # Ignore threads, group DMs, and messages with no guild (e.g. system messages)
        elif isinstance(message.channel, (Thread, GroupChannel)) or message.guild is None:
            return

        if not direct_message:
            if guild_settings is None:
                logger.debug(f"Got message in unknown guild {message.guild=}")
                return  # ignore messages from guilds that don't have settings
            if guild_settings.enabled is False:
                # logger.debug(f"Got message in disabled guild {message.guild=}")
                return  # ignore messages from guilds that have the bot disabled
            if guild_settings.channel_enabled(message.channel.id) is False:
                # logger.debug(f"Got message in disabled channel {message.channel=}")
                return  # ignore messages from channels that have the bot disabled
            elif message.channel.permissions_for(message.guild.me).send_messages is False:
                # channel is enabled but we don't have permission to respond
                logger.info(f"Got message in {message.channel} but don't have permission to respond.")
                return

        # Check if the user has accepted the ToS
        tos_accepted = await self.user_accepted_tos(message.author)
        if tos_accepted is False:
            return  # ignore messages from users who have rejected the ToS
        if message.author.bot is True:
            tos_accepted = True  # ignore ToS check for bots

        # Now that we've filtered out messages we don't care about, let's get the message content
        content = self.get_msg_content_clean(message)
        if content is None:
            return  # ignore empty messages

        trigger: Optional[str] = None  # response trigger reason (tfw no StrEnum in 3.10)
        conversation: Optional[list[str]] = None  # message's conversation context
        append = None  # optional masked message to append to the response

        try:
            async with self.lm_lock:
                if direct_message:
                    logger.debug(f"DM from {message.author}: {content}")
                    trigger = "DM"

                elif self.check_mention(message):
                    logger.debug(f"Mentioned in {message.channel}: {content}")
                    trigger = "mention"

                if trigger is not None:
                    await message.channel.trigger_typing()
                    if conversation is None:
                        conversation = await self.get_message_context(message, guild_settings=guild_settings)

                    # save whether we're responding to a bot or not to prevent loops
                    self._trigger_cache.update({message.channel.id: message.author.bot is True})

                    # send ToS if it's not been accepted or rejected yet
                    if tos_accepted is None:
                        append = "Please check your DMs for a privacy notice!"
                        await self.check_send_tos(message)

                    # actually respond
                    await self.do_response(conversation, message, trigger, append=append)
                    return

        except Exception as e:
            logger.exception(e)
            if isinstance(e, HTTPException):
                return  # ignore HTTP errors
            exc_class = e.__class__.__name__
            exc_desc = str(f"**``{exc_class}``**\n```{format_exc()}```")
            err_filename = f"error-{datetime.now(timezone.utc)}.txt".replace(" ", "")
            self.debug_dir.joinpath(err_filename).write_text(exc_desc)

    # retrieve the LM prompt and inject name, time, etc.
    def get_prompt(self, ctx: Optional[MessageInteraction] = None) -> str:
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

        prompt = "\n".join([x for x in prompt if x is not None and len(x) > 0])

        replace_tokens = {
            "bot_name": self.name,
            "location_context": location_context,
            "current_time": get_prompt_datetime(with_date=self.prompt.with_date),
            "time_type": "time and date" if self.prompt.with_date else "time",
        }
        for token, replacement in replace_tokens.items():
            prompt = prompt.replace(f"{{{token}}}", replacement)
        return prompt

    # get the last N messages in a channel, up to and including the message that triggered the response
    async def get_message_context(
        self,
        message: Message,
        *,
        guild_settings: Optional[GuildSettings] = None,
        max_messages: Optional[int] = None,
    ) -> Union[str, list[str]]:
        if max_messages is None:
            max_messages = self.params.context_messages

        messages = await message.channel.history(limit=max_messages, before=message).flatten()
        messages.insert(0, message)  # add the message that triggered the response

        # magic tag to break context chain to get bot out of librarian mode
        for idx, message in enumerate(messages):
            if self.check_ctxbreak(message):
                logger.debug("Found context break tag, breaking context chain")
                messages = messages[0 : idx + 1]
                break

        if guild_settings is None:
            # attempt to retrieve guild settings if we don't have them
            guild_settings = self.params.get_guild_settings(message.guild.id) if message.guild else None

        if guild_settings is not None:
            # get bot mode for this channel
            bot_mode = guild_settings.channel_bot_mode(message.channel.id)
        else:
            # default to siblings mode if we still don't have guild settings
            bot_mode = BotMode.Siblings

        def wrap_message(content: str, role: str = "user") -> dict[str, str]:
            return {"role": role.strip(), "content": content.strip()}

        chain: list[dict[str, str]] = []
        for msg in reversed(messages):
            if msg.author.id in self.tos_reject_ids:
                continue  # skip users who rejected the privacy policy
            if msg.content is not None:
                if any((msg.content.startswith(x) for x in ["-", "/", ", "])):
                    continue  # skip messages that start with a command prefix or a comma-space
                if "```" in msg.content:
                    continue  # skip messages with code blocks

            if msg.author.id == self.bot.user.id:
                if msg.content.startswith("This is an AI chatbot made by"):
                    continue  # this is a TOS message in a DM so lets skip it
            elif msg.author.bot is True:  # bots who aren't us
                if bot_mode == BotMode.Strip:
                    logger.debug(f"Stripping bot message {msg.id} from context chain")
                    continue  # skip all other bot messages if we're in strip mode
                elif bot_mode == BotMode.Siblings:
                    if msg.author.id not in self.sibling_ids:
                        logger.debug(f"Stripping non-sibling bot message {msg.id} from context chain")
                        continue  # skip non-sibling bot messages

            # set up author name
            if msg.author.id == self.bot.user.id:
                msg_role = "assistant"
                author_name = f"{self.name}:"
            else:
                msg_role = "user"
                author_name = f"{msg.author.display_name.strip()}:"

            if len(msg.embeds) > 0:
                nitro_bs = self.handle_stupid_fucking_embed(message)
                if nitro_bs is not None:
                    chain.append(wrap_message(f"{author_name} {nitro_bs}", msg_role))
                    continue
                for embed in msg.embeds:
                    if embed.type == "image":
                        try:
                            caption = await self.eyes.perceive_url(embed.thumbnail.url, msg.id)
                            if caption is not None:
                                chain.append(wrap_message(f"{author_name} [image: {caption}]", msg_role))
                        except Exception as e:
                            logger.exception(e)
                            chain.append(wrap_message(f"{author_name} [image: loading error]", msg_role))
                    elif embed.description is not None and msg.author.bot is False:
                        chain.append(wrap_message(f"{author_name} [embed: {embed.description}]", msg_role))
                    elif embed.title is not None and msg.author.id != self.bot.user.id:
                        chain.append(wrap_message(f"{author_name} [embed: {embed.title}]", msg_role))

            if msg.content and len(msg.embeds) == 0:
                content = self.get_msg_content_clean(msg)
                if content is None:
                    logger.debug(f"Message {msg.id} was empty after cleaning, skipping...")
                    continue  # skip empty-after-cleaning messages

                if content.startswith("http"):
                    chain.append(wrap_message(f"{author_name} [a link to a webpage]", msg_role))
                    continue

                if msg.author.id == self.bot.user.id:
                    content = re_start_expression.sub("", content)
                    if len(content) >= 0:
                        chain.append(wrap_message(f"{author_name} {content}", msg_role))
                    else:
                        logger.debug(f"Self-sent message {msg.id} was empty after cleaning, skipping...")
                        continue
                else:
                    chain.append(wrap_message(f"{author_name} {content}", msg_role))

            for attachment in msg.attachments:
                try:
                    if not attachment.content_type.startswith("image/"):
                        logger.debug(f"got non-image content-type: '{attachment.content_type}', skipping")
                        continue
                    # caption image
                    caption = await self.eyes.perceive_attachment(attachment)
                    if caption is not None:
                        chain.append(wrap_message(f"{author_name} [image: {caption}]", msg_role))
                    else:
                        chain.append(wrap_message(f"{author_name} [image: unknown content]", msg_role))
                    continue
                except Exception as e:
                    logger.exception(e)
                    chain.append(wrap_message(f"{author_name} [image: loading error]", msg_role))

        return chain

    # assemble a prompt/context for the model
    async def process_context(self, conversation: list[str | dict[str, str]], message: Message):
        if self.prompt.disco_mode:
            logger.debug("Disco mode enabled, returning only the message that triggered the response")
            context = message.content
            context = re_mention.sub("", context)  # remove mentions
            return context.strip()

        contextmgr = ContextPreprocessor(token_budget=self.context_size, tokenizer=self.tokenizer)
        logger.debug(f"building context from {len(conversation)} messages")

        if isinstance(conversation[0], dict):
            conversation = self.apply_chat_template(conversation, add_generation_prompt=False).splitlines()
            logger.debug(f"applied chat template:\n{conversation}")

        if isinstance(conversation, list):
            conversation = "\n".join([x.strip() for x in conversation])

        prompt_entry = ContextEntry(
            text=self.get_prompt(message),
            prefix="",
            suffix="",
            reserved_tokens=self.n_prompt_tokens,
            insertion_order=1000,
            insertion_position=0,
            trim_direction=TrimDir.Never,
            trim_type=BreakType.Newline,
            insertion_type=BreakType.Newline,
            forced_activation=True,
            cascading_activation=False,
            tokenizer=self.tokenizer,
        )
        contextmgr.add_entry(prompt_entry)

        conversation_entry = ContextEntry(
            text=conversation + f"\n<|im_start|>assistant\n{self.name}:",
            prefix="",
            suffix="",
            reserved_tokens=1024,
            insertion_order=500,
            insertion_position=-1,
            trim_direction=TrimDir.Top,
            trim_type=BreakType.Newline,
            insertion_type=BreakType.Newline,
            forced_activation=True,
            cascading_activation=False,
            tokenizer=self.tokenizer,
        )
        contextmgr.add_entry(conversation_entry)

        context = contextmgr.context(self.context_size)
        context = "\n".join([x.strip() for x in context.splitlines() if len(x.strip()) > 0])
        context = re_consecutive_newline.sub("\n", context)  # yeet all consecutive newlines (empty lines)
        context = context.replace("|> ", "|>")  # remove any spurious whitespace from the roles
        return context

    # actual response logic
    async def do_response(
        self,
        conversation: list[str],
        message: Message,
        trigger: Optional[str] = None,
        append: Optional[str] = None,
    ) -> str:
        async with message.channel.typing():
            debug_data: dict[str, Any] = {}

            response = ""
            response_image = None
            should_reply = False
            author_name = f"{message.author.display_name.strip()}:"

            msg_timestamp = message.created_at.astimezone(tz=self.bot.timezone).strftime(
                "%Y-%m-%d-%H:%M:%S%z"
            )
            msg_trigger = trigger.lower() if trigger is not None else "unknown"

            # build debug data
            debug_data["id"] = message.id
            debug_data["app_id"] = self.bot.user.id
            debug_data["instance"] = self.name.title()
            debug_data["message"] = {
                "id": message.id,
                "timestamp": msg_timestamp,
                "guild_id": message.guild.id if message.guild else None,
                "guild": message.guild.name if message.guild else "DM",
                "author_id": message.author.id,
                "author": f"{message.author}",
                "channel_id": message.channel.id or None,
                "channel": message.channel.name if not isinstance(message.channel, DMChannel) else "DM",
                "author_name": f"{author_name}",
                "trigger": msg_trigger,
                "content": message.content,
            }
            # make conversation be a top level key
            if isinstance(conversation[0], dict):
                debug_data["conversation"] = [f'{x["content"]}' for x in conversation]
            else:
                debug_data["conversation"] = conversation

            try:
                debug_data["parameters"] = self.lm_gensettings.dict()
            except Exception as e:
                logger.exception("Failed to get gensettings")

            # Build context
            context = await self.process_context(conversation, message)
            context = context.rstrip()
            debug_data["context"] = context.splitlines()
            debug_data["n_prompt_tokens"] = self.n_prompt_tokens
            ctokens = self.tokenizer.batch_encode_plus([context], return_length=True)
            debug_data["n_context_tokens"] = ctokens.get("length")[0]

            try:
                # Generate the response, and retry if it contains bad words (up to self.max_retries times)
                for attempt in range(self.max_retries):
                    attempt = attempt + 1  # deal with range() starting from 0
                    response: str = await self.chatbot.respond_async(context)
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
                if description is not None and "loading error" not in description.lower():
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
                        logger.exception("Failed to generate image response")
                        pass

                debug_data["response_raw"] = response

                # if bot did a "\n<someusername:" cut it off
                if bool(re_linebreak_name.match(response)):
                    response = response.splitlines()[0]

                # replace "<USER>" with user mention, same for "<BOT>"
                response = self.fixup_bot_user_tokens(response, message).lstrip()
                if self.params.force_lowercase:
                    if response.isupper() is False:  # only force lowercase if we are not yelling
                        response = response.lower()
                # Clean response - trim left whitespace and fix emojis and pings
                response = self.restore_mentions_emoji(text=response, message=message)
                # Unescape markdown
                response = re_unescape_md.sub(r"\1", response)
                # Clean up multiple non-word characters at the end of the response
                response = re_nonword_end.sub(r"\1", response)
                # scream into the void
                response = response.replace("\\r", "").replace("\\n", "\n")
                response = response.replace("\\u00a0", "\n")

                if self.prompt.disco_mode:
                    logger.debug("Prepending prompt to response (disco mode)")
                    response = f"{context} {response}"

                response_file = None
                if len(response) > 1900:
                    if self.prompt.disco_mode:
                        logger.debug("Overlength response in disco mode, will send as file")
                        response_file = File(StringIO(response), filename="story.txt")
                    else:
                        logger.debug("Response is too long, trimming...")
                        response = response[:1900].strip() + "-"

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
                    if response_file is not None:
                        content = " ".join(response.split(" ")[:20]) + "... (too long, attached)"
                        await message.channel.send(content=content, file=response_file)
                        logger.info(f"Response: {content} (with file)")
                    else:
                        await message.channel.send(response)
                        logger.info(f"Response: {response}")

                self.last_response = datetime.now(timezone.utc)

                # update timestamp for this channel in the image caption engine
                if self.eyes.enabled and self.eyes.background:
                    self.eyes.watch(message.channel)

                # update webui state if it's enabled
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
                    dump_file = f"msg-{self.name.lower()}-{message.id}-{msg_timestamp}.json"
                    if debug_data.get("trigger", None) == "DM":
                        dump_file = self.debug_dir.joinpath("dm", dump_file)
                        dump_file.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        dump_file = self.debug_dir.joinpath(dump_file)
                    with dump_file.open("w", encoding="utf-8") as f:
                        json.dump(debug_data, f, indent=4, skipkeys=True, default=str, ensure_ascii=False)
                    logger.debug(f"Dumped message debug data to {dump_file.name}")
        await self.log_response(debug_data)

    async def log_response(self, debug_data: dict) -> None:
        debug_data["timestamp"] = datetime.now(tz=self.bot.timezone)
        try:
            log_entry = AiResponseLog(**debug_data)
            async with self.db_client.begin() as session:
                session.add(log_entry)
            logger.debug(f"Logged response to database: {log_entry.id}")
        except Exception:
            logger.exception("Failed to log response to database, continuing anyway")

    # Idle loop, broken rn, need to factor on_message logic out into functions or smth
    @tasks.loop(seconds=21)
    async def idle_loop(self) -> None:
        if self.idle_enable is True:
            # get last message in a random priority channel
            channel = self.bot.get_channel(random_choice(self.idle_channels))
            if isinstance(channel, (TextChannel, DMChannel, GroupChannel)):
                messages = await channel.history(limit=1).flatten()
                message: Message = messages.pop()
                if message.author.bot is True:
                    return
                idle_sec = (datetime.now(tz=timezone.utc) - message.created_at).total_seconds()
                if idle_sec >= 2**31:
                    if self.get_msg_content_clean(message) is None:
                        return
                    logger.debug(f"Running idle response to message {message.id}")
                    # if it's been more than <idle_interval> sec, send a response
                    async with self.lm_lock:
                        conversation = await self.get_message_context(message)
                        await self.do_response(conversation, message)
        else:
            self.idle_loop.stop()
            return

    @idle_loop.before_loop
    async def before_idle_loop(self):
        logger.info("Idle loop waiting for bot to be ready")
        await self.bot.wait_until_ready()
        await asyncio.sleep(10)
        logger.info("Idle loop running!")

    ## Helper functions
    # Check and refresh the list of users who rejected the ToS
    @tasks.loop(seconds=60)
    async def update_ignored(self):
        """Check the database for users who have rejected the ToS and get a list of their IDs for message filtering"""
        async with self.db_client.begin() as session:
            query = (
                select(DiscordUser)
                .where(DiscordUser.tos_rejected == True)
                .with_only_columns(DiscordUser.id, DiscordUser.tos_accepted, DiscordUser.tos_rejected)
            )
            results = await session.scalars(query)
            users: list[DiscordUser] = results.all()
        self.tos_reject_ids = set([x.id for x in users])

    # Check if a user has accepted the ToS
    async def user_accepted_tos(self, user: Union[User, Member, int]) -> Optional[bool]:
        """Checks if a user has accepted the ToS, rejected, or not completed the process.
        Returns True if accepted, False if rejected, None if not completed.
        """
        user_id = user.id if isinstance(user, (User, Member)) else user
        async with self.db_client.begin() as session:
            user: DiscordUser = await session.get(DiscordUser, user_id)
            if user is None:
                logger.debug(f"User {user_id} has not completed the ToS process.")
                return None
            elif user.tos_rejected is True:
                return False
            elif user.tos_accepted is True:
                return True

    # get a list of users who have accepted the ToS
    async def tos_accepted_users(self) -> set[int]:
        async with self.db_client.begin() as session:
            query = (
                select(DiscordUser)
                .options(load_only("id", "tos_accepted", "tos_rejected", raiseload=True))
                .filter(DiscordUser.tos_accepted is True)
            )
            result = await session.scalars(query)
            users = result.all()
        return {x.id for x in users}

    # check if the person triggering the bot has agreed to ToS or not and send it if not
    async def check_send_tos(self, message: Message) -> bool:
        logger.debug(f"Checking ToS for {message.author} ({message.author.id})")
        try:
            async with self.db_client.begin() as session:
                user: DiscordUser = await session.get(DiscordUser, message.author.id)
                if user is None:
                    logger.info(f"User {message.author} not found in database, creating entry")
                    user = DiscordUser.from_discord(message.author)
                    await session.merge(user)
                    await session.flush()
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
            return False

    # clean up a message's content
    def get_msg_content_clean(self, message: Message, content: Optional[str] = None) -> Optional[str]:
        # if no content is provided, use the message's content
        content = content or message.content
        # if there's a codeblock in there, return None
        if "```" in content:
            return None
        # convert mentions and emoji from user IDs to to plain text
        content = self.stringify_mentions_emoji(content, message=message)
        # eliminate any and all content between angle brackets <>
        content = re_angle_bracket.sub("", content)
        # turn newlines into spaces. this is a cheap hack but the model doesn't handle newlines well
        # content = content.replace("\n", " ")
        # logger.debug(f"Cleaned message content: {content}")
        # strip leading and trailing whitespace
        content = content.strip()
        # return the cleaned content, or None if it's empty
        return content if len(content) > 0 else None

    # clean out some bad tokens from responses and fix up mentions
    def fixup_bot_user_tokens(self, response: str, message: Message) -> str:
        """
        Fix <USER>, <BOT>, etc tokens in the response, and unescape any escaped markdown formatting
        """
        author_name = message.author.display_name.encode("utf-8").decode("ascii", errors="ignore").strip()
        response = re_user_token.sub(f"@{author_name}", response)
        response = re_bot_token.sub(f"@{self.name}", response)
        response = re_unescape_md.sub(r"\1", response)
        response = response.replace("</s>", "").strip()
        return response

    # search for bad words in a message
    def find_bad_words(self, input: str) -> str:
        found_words: list[str] = []
        input_words: str = input.split()

        for word in input_words:
            word_stripped: str = re_nonword.sub("", word).lower()
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
    def check_ctxbreak(self, message: Message) -> bool:
        content = message.content.lower()
        if not content.startswith("<ctxbreak>"):
            return False  # not a context break message

        if self.config.params.ctxbreak_restrict is False:
            return True  # ctxbreak is unrestricted

        if isinstance(message.channel, DMChannel):
            return True  # people can always break context in DMs
        if message.author.id in list(self.ctxbreak.user_ids + self.bot.config.admin_ids):
            return True  # admins and ctxbreak users can always break context
        if member_in_any_role(message.author, self.ctxbreak.role_ids):
            return True  # ctxbreak role havers can always break context
        return False  # otherwise, no

    # check if the last message we responded to in this context was from a bot
    def last_trigger_was_bot(self, message: Message) -> bool:
        """Check if the last message we responded to in this context was from a bot."""
        cache_entry = self._trigger_cache.get(message.channel.id, False)
        return cache_entry

    # deal with fake nitro bullshit
    def handle_stupid_fucking_embed(self, message: Message) -> Optional[str]:
        """Check if a message is a stupid fucking spec-beaking fake nitro emote embed
        Returns the emote name if it is, otherwise returns None.
        """
        if message.content.startswith("https://cdn.discordapp.com/emojis/"):
            logger.debug("Found a stupid fake nitro emote")
            name = message.content.split("name=", 1)[-1]
            name = name.split("&", 1)[0]
            return f":{name}:"
        else:
            return None

    ## Loop to archive debug logs
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

    @log_archive_loop.before_loop
    async def before_archive_loop(self):
        logger.info("Archive loop waiting for bot to be ready")
        await self.bot.wait_until_ready()
        self.debug_dir.joinpath("archive").mkdir(exist_ok=True, parents=True)
        logger.info("Archive loop running!")

    ## listener to background caption images
    @commands.Cog.listener("on_message")
    async def background_caption(self, message: Message):
        if not message.attachments and not message.embeds:
            pass  # no attachments or embeds

        # check if we're watching this channel
        if not self.eyes.watching(message.channel, False):
            return  # not watching this channel (no recent activity)

        for embed in message.embeds:
            if embed.type == "image":
                try:
                    if embed.thumbnail.height < 100 or embed.thumbnail.width < 100:
                        continue  # too small to caption, probably an emoji, just skip it
                    logger.debug(f"Captioning embed from {message.id=}")
                    caption = await self.eyes.perceive_url(embed.thumbnail.url, message.id)
                    logger.info(f"Success: {message.id=}, {caption=}")
                except Exception:
                    logger.exception(f"Error processing image {embed.thumbnail.url}")
            else:
                continue
            break  # only caption one image per embed. limitations of how i key the DB :/

        for attachment in message.attachments:
            if attachment.content_type.startswith("image"):
                try:
                    logger.debug(f"Captioning attachment {attachment.id=} from {message.id=}")
                    caption = await self.eyes.perceive_attachment(attachment)
                    logger.info(f"Success: {attachment.id=}: {caption=}")
                except Exception:
                    logger.exception(f"Error processing image {attachment.url}")

    ## Image generation stuff
    async def take_pic(self, message: Union[Message, str]) -> Optional[Tuple[File, dict]]:
        """
        hold up, let me take a selfie
        """
        # get the message content
        if isinstance(message, Message):
            content: Optional[str] = self.get_msg_content_clean(message)
        elif isinstance(message, str):
            content: str = message.lower()
        else:
            raise ValueError("take_pic got an unexpected message type (not str or Message)")

        if content is None:
            logger.debug("Message was empty after cleaning.")
            return None

        # lowercase the content
        content = content.lower().strip()

        # get the image description and remove newlines
        content = content.replace("\n", " ").replace(f"{self.name} ", "")

        # build the LLM prompt for the image
        lm_trigger = self.imagen.strip_take_pic(content)
        lm_prompt, user_request = self.imagen.get_lm_prompt(lm_trigger)
        logger.info(f"LLM Request: {user_request}")

        # get the LLM to create tags for the image
        lm_tag_string = await self.imagen.submit_lm_prompt(lm_prompt)
        lm_tag_string = lm_tag_string.strip('",').lower()
        logger.info(f"LLM Response: {lm_tag_string}")

        # build the SD API request
        sdapi_request = self.imagen.build_request(lm_tag_string, content)
        logger.info(f"SD API Request: {json.dumps(sdapi_request)}")

        # submit it
        try:
            result_path = await self.imagen.submit_request(sdapi_request)
            # return a discord File object to upstream
            result_file = File(result_path, filename=result_path.name)
            if self.webui.config.enabled:
                try:
                    self.webui.imagen_update(lm_trigger, lm_tag_string, result_path)
                except Exception as e:
                    logger.exception(e)
            return result_file
        except RuntimeError as e:
            logger.exception(e)
            raise e

    ## UI stuff
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
        await ctx.send(embed=embed, ephemeral=True)

    @ai_group.sub_command(name="set", description="Set AI parameters")
    @checks.is_admin()
    async def set_parameter(
        self,
        ctx: ApplicationCommandInteraction,
        param: AiParam = commands.Param(
            description="Parameter", choices=set_choices, converter=convert_param
        ),
        value: str = commands.Param(...),
    ) -> None:
        await ctx.response.defer(ephemeral=True)
        embed = Embed(title="Set Parameter", description="Success!", color=Colour.green())
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
        embed.add_field(name="Parameter", value=f"{param.name}", inline=False)

        try:
            if not hasattr(self.lm_gensettings, param.id):
                raise ValueError(f"Unknown parameter: {param} (got {param.id}, {param.kind})")

            # cast the value to the correct type using the class from the tuple
            new_value = param.kind(value)
            # save the old value and set the new one
            old_value = getattr(self.lm_gensettings, param.id)
            setattr(self.lm_gensettings, param.id, new_value)

            # add the old and new values to the embed
            embed.add_field(name="Old", value=f"{old_value}")
            embed.add_field(name="New", value=f"{new_value}")

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
            # send the embed
            await ctx.send(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Ai(bot))
