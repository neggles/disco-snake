import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, Dict, List, Optional

from dacite import from_dict
from disnake import DMChannel, Embed, File, GroupChannel, Message, TextChannel, Thread, User
from disnake.ext import commands, tasks
from shimeji import ChatBot
from shimeji.memory import array_to_str, memory_context
from shimeji.memorystore_provider import PostgreSQLMemoryStore
from shimeji.model_provider import SukimaModel
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.preprocessor import ContextPreprocessor
from shimeji.util import (
    INSERTION_TYPE_NEWLINE,
    TRIM_DIR_TOP,
    TRIM_TYPE_NEWLINE,
    TRIM_TYPE_SENTENCE,
    ContextEntry,
)
from transformers import GPT2Tokenizer

import logsnake
from ai.config import ChatbotConfig, MemoryStoreConfig, ModelProviderConfig
from ai.model import get_sukima_model
from ai.types import MessageChannel
from ai.utils import (
    anti_spam,
    convert_mentions_emotes,
    cut_trailing_sentence,
    get_full_class_name,
    get_role_by_name,
    restore_mentions_emotes,
)
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake

COG_UID = "ai"


# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=2,
)

re_angle_bracket = re.compile(r"\<([^>]*)\>")
re_user_token = re.compile(r"(<USER>|<user>|{{user}})")
re_bot_token = re.compile(r"(<BOT>|<bot>|{{bot}}|<CHAR>|<char>|{{char}})")


class Ai(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.timezone = self.bot.timezone
        self.last_response = datetime.utcnow() - timedelta(minutes=10)

        # Load config file
        self.cfg_path: Path = DATADIR_PATH.joinpath("ai", "config.json")
        try:
            self.config = from_dict(data_class=ChatbotConfig, data=json.loads(self.cfg_path.read_text()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.cfg_path}")

        # Parse config file
        self.memory_store_cfg: MemoryStoreConfig = self.config.memory_store
        self.model_provider_cfg: ModelProviderConfig = self.config.model_provider

        # Load config params up into top level properties
        self.activity_channels: List[int] = self.config.params.activity_channels
        self.conditional_response: bool = self.config.params.conditional_response
        self.context_size: int = self.config.params.context_size
        self.idle_messaging_interval: int = self.config.params.idle_messaging_interval
        self.idle_messaging: bool = self.config.params.idle_messaging
        self.logging_channel_id: int = self.config.params.logging_channel_id
        self.nicknames: List[str] = self.config.params.nicknames
        self.debug: bool = self.config.params.debug

        # we will populate these later during async init
        self.memory_store: PostgreSQLMemoryStore = None
        self.model_provider: SukimaModel = None
        self.chatbot: ChatBot = None
        self.logging_channel: Optional[TextChannel] = None
        self.guilds: dict = {}
        self.tokenizer: GPT2Tokenizer = None

        # somewhere to put the last context we generated for debugging
        self.debug_datadir: Path = LOGDIR_PATH.joinpath("ai")
        self.debug_datadir.mkdir(parents=True, exist_ok=True)

    # Getters for config object sub-properties
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def prompt(self) -> str:
        return self.config.prompt

    async def cog_load(self) -> None:
        logger.info("AI engine initializing, please wait...")
        # Set up MemoryStoreProvider
        self.memory_store = PostgreSQLMemoryStore(
            database_uri=self.memory_store_cfg.database_uri,
            model=self.memory_store_cfg.model,
            model_layer=self.memory_store_cfg.model_layer,
            short_term_amount=self.memory_store_cfg.short_term_amount,
            long_term_amount=self.memory_store_cfg.long_term_amount,
        )
        self.model_provider = get_sukima_model(cfg=self.model_provider_cfg)
        self.chatbot = ChatBot(
            name=self.name,
            model_provider=self.model_provider,
            preprocessors=[ContextPreprocessor(self.context_size)],
            postprocessors=[NewlinePrunerPostprocessor()],
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.cfg_path.parent.joinpath("tokenizer").as_posix(),
        )
        _logchannel = self.bot.get_channel(self.logging_channel_id)
        self.logging_channel = _logchannel if isinstance(_logchannel, (TextChannel, DMChannel)) else None

        logger.info("AI engine initialized... probably?")
        if self.idle_messaging is True:
            logger.info("Starting idle messaging loop")
            self.idle_loop.start()
        if not self.log_archive_loop.is_running():
            logger.info("Archiving logs from over 15 min ago")
            self.archive_logs(seconds=900)
            logger.info("Starting log archive loop")
            self.log_archive_loop.start()

    # build a context from the last 40 messages in the channel
    async def get_msg_ctx(self, channel: MessageChannel) -> str:
        messages = await channel.history(limit=40).flatten()
        messages, to_remove = anti_spam(messages)
        if to_remove:
            logger.info(f"Removed {to_remove} messages from context.")
        chain = []
        for message in reversed(messages):
            if len(message.embeds) > 0:
                content = message.embeds[0].description
                if content != "":
                    chain.append(f"{message.author.name}: [Embed: {content}]")
                continue
            elif message.content != "":
                content = self.get_msg_content_clean(message)
                if content != "" and not "```" in content:
                    chain.append(f"{message.author.name}: {content}")
                continue
            elif message:
                chain.append(f"{message.author.name}: [Image attached]")
        return "\n".join(chain)

    async def build_ctx(self, conversation: str):
        contextmgr = ContextPreprocessor(token_budget=self.context_size, tokenizer=self.tokenizer)

        prompt_entry = ContextEntry(
            text=self.prompt + "\n\n<START>\n",
            prefix="",
            suffix="",
            reserved_tokens=512,
            insertion_order=1000,
            insertion_position=-1,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
        )
        contextmgr.add_entry(prompt_entry)

        # memories
        if self.memory_store is not None:
            memories = await self.memory_store.get()
            if not memories:
                logger.info("No memories found.")
            else:
                memories_ctx = memory_context(
                    memories[-1],
                    memories,
                    short_term=self.memory_store.short_term_amount,
                    long_term=self.memory_store.long_term_amount,
                )
                memories_entry = ContextEntry(
                    text=memories_ctx,
                    prefix="",
                    suffix="",
                    reserved_tokens=0,
                    insertion_order=800,
                    insertion_position=-1,
                    trim_direction=TRIM_DIR_TOP,
                    trim_type=TRIM_TYPE_SENTENCE,
                    insertion_type=INSERTION_TYPE_NEWLINE,
                    forced_activation=True,
                    cascading_activation=False,
                )
                contextmgr.add_entry(memories_entry)

        # conversation
        conversation_entry = ContextEntry(
            text="\n<START>\n" + conversation,
            prefix="",
            suffix=f"\n{self.name}:",
            reserved_tokens=512,
            insertion_order=0,
            insertion_position=-1,
            trim_direction=TRIM_DIR_TOP,
            trim_type=TRIM_TYPE_NEWLINE,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
        )
        contextmgr.add_entry(conversation_entry)

        return contextmgr.context(self.context_size)

    async def respond(self, conversation: str, message: Message):
        async with message.channel.typing():
            encoded_image_label = ""
            debug_data: Dict[str, Any] = {}

            msg_timestamp = message.created_at.astimezone(tz=self.bot.timezone).strftime(
                "%Y-%m-%d %H:%M:%S%z"
            )

            debug_data["message"] = {
                "id": message.id,
                "author": f"{message.author}",
                "guild": f"{message.guild}" if message.guild is not None else "DM",
                "channel": message.channel.name if not isinstance(message.channel, DMChannel) else "DM",
                "timestamp": msg_timestamp,
                "content_raw": message.content,
            }

            if self.memory_store is not None and message.guild is not None:
                private_role = get_role_by_name(name="Private", guild=message.guild)
                anonymous_role = get_role_by_name(name="Anonymous", guild=message.guild)
                if private_role is None or isinstance(message.author, User):
                    pass

                elif private_role not in message.author.roles:
                    if anonymous_role is not None and anonymous_role in message.author.roles:
                        author_name = "Deleted User"
                    else:
                        author_name = message.author.name

                    message_content = self.get_msg_content_clean(message)
                    encoded_user_message = f"{author_name}: {message_content}"
                    is_dupe = await self.memory_store.check_duplicates(
                        text=encoded_user_message, duplicate_ratio=0.8
                    )
                    if message_content != "" and not is_dupe:
                        logger.info(f"adding message from {message.author} to memory as '{author_name}'")
                        await self.memory_store.add(
                            author_id=message.author.id,
                            author=author_name,
                            text=message_content,
                            encoding_model=self.memory_store.model,
                            encoding=array_to_str(
                                await self.model_provider.hidden_async(
                                    self.memory_store.model,
                                    encoded_user_message,
                                    layer=self.memory_store.model_layer,
                                )
                            ),
                        )

            debug_data["conversation"] = conversation.splitlines()

            # Build conversation context
            conversation = await self.build_ctx(conversation + encoded_image_label)
            debug_data["context"] = conversation.splitlines()

            # Generate response
            response: str = await self.chatbot.respond_async(conversation, push_chain=False)
            debug_data["response_raw"] = response

            # replace "<USER>" with user mention, same for "<BOT>"
            response = re_user_token.sub(f"@{message.author.name}", response)
            response = re_bot_token.sub(f"{self.name}", response)

            # Clean response - trim left whitespace and fix emojis and pings
            response = cut_trailing_sentence(response)
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
            debug_data["response"] = response

            self.bot.get_emoji

            # Send response if not empty
            if response == "":
                logger.info(f"Response was empty.")
            else:
                logger.info(f"Response: {response}")
                await message.channel.send(response)
                self.last_response = datetime.utcnow()

        if self.debug:
            message_time = message.created_at.astimezone(self.bot.timezone).strftime("%Y-%m-%d-%H%M%S%z")
            dump_file = self.debug_datadir.joinpath(f"msg-{message_time}-{message.id}.json")
            dump_file.write_text(json.dumps(debug_data, indent=4, skipkeys=True, default=str))
            logger.debug(f"Dumped message debug data to {dump_file.name}")

        # add to memory store in background
        if self.memory_store is not None:
            # encode bot response
            if await self.memory_store.check_duplicates(text=response, duplicate_ratio=0.8) is False:
                await self.memory_store.add(
                    author_id=self.bot.user.id,
                    author=self.name,
                    text=response,
                    encoding_model=self.memory_store.model,
                    encoding=array_to_str(
                        await self.model_provider.hidden_async(
                            self.memory_store.model,
                            f"{self.name}: {response}",
                            layer=self.memory_store.model_layer,
                        )
                    ),
                )

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if (message.author.bot is True) or (message.author == self.bot.user):
            return
        if "```" in message.content:
            return

        # Ignore messages from unapproved users in DMs/groups
        if isinstance(message.channel, (DMChannel, GroupChannel)):
            if message.author.id not in self.bot.config["owners"]:
                logger.info(
                    f"Got a DM from non-owner {message.author.name}#{message.author.discriminator}. Ignoring..."
                )
                return
        # Ignore threads
        elif isinstance(message.channel, Thread):
            return
        # Ignore messages with no guild
        elif message.guild is None:
            return
        elif message.guild.id not in self.bot.config["ai_guilds"]:
            return

        try:
            logger.info(f"Raw message: {message.content}")
            message_content = self.get_msg_content_clean(message)
            if message_content == "":
                logger.info("Message was empty after cleaning.")
                return
            else:
                logger.info(f"Message: {message_content}")

            conversation = await self.get_msg_ctx(message.channel)

            if self.bot.user.mentioned_in(message) or any(t in message_content for t in self.nicknames):
                await self.respond(conversation, message)

            elif isinstance(message.channel, DMChannel) and len(message_content) > 8:
                await self.respond(conversation, message)

            elif self.conditional_response is True:
                if await self.chatbot.should_respond_async(conversation, push_chain=False):
                    logger.debug("Model wants to respond, responding...")
                    await self.respond(conversation, message)
                    return
                else:
                    logger.debug("No conditional response.")

            elif (
                message.channel.id in self.activity_channels
                and self.last_response
                < datetime.utcnow() - timedelta(seconds=(self.idle_messaging_interval / 2))
            ):
                await self.respond(conversation, message)

        except Exception as e:
            logger.error(e)
            logger.error(format_exc())
            exc_class = get_full_class_name(e)
            if exc_class == "HTTPException":
                # don't bother, the backend is down
                return

            exc_desc = str(f"**``{exc_class}``**\n```{format_exc()}```")
            error_file = self.debug_datadir.joinpath(f"error-{datetime.utcnow()}.txt")
            error_file.write_text(exc_desc)

            if len(exc_desc) < 2048:
                embed = Embed(title="**Exception**", description=f"**``{exc_class}``**")
                if self.logging_channel is not None:
                    await self.logging_channel.send(embed=embed)
                else:
                    await message.channel.send(embed=embed, delete_after=300.0)
            else:
                embed = Embed(
                    title="**Exception**",
                    description="Exception too long for message, see attached file.",
                )
                if self.logging_channel is not None:
                    await self.logging_channel.send(embed=embed, file=File(error_file))
                else:
                    await message.channel.send(embed=embed, file=File(error_file), delete_after=300.0)

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
                if idle_sec >= self.idle_messaging_interval:
                    if self.get_msg_content_clean(message) == "":
                        return
                    logger.debug(f"Running idle response to message {message.id}")
                    # if it's been more than <idle_messaging_interval> sec, send a response
                    conversation = await self.get_msg_ctx(message.channel)
                    await self.respond(conversation, message)
        else:
            return

    @idle_loop.before_loop
    async def before_idle_loop(self):
        logger.info("Idle loop waiting for bot to be ready")
        await self.bot.wait_until_ready()
        logger.info("Idle loop running!")

    @tasks.loop(hours=1)
    async def log_archive_loop(self) -> None:
        """
        Move message log files over 24h old to the archive directory
        """
        self.archive_logs(seconds=86400)

    def archive_logs(self, seconds: int = 86400):
        for file in self.debug_datadir.glob("*.json"):
            file_age = (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).total_seconds()
            if file_age >= seconds:
                logger.debug(f"Archiving debug log {file.name}")
                file = file.rename(file.parent / "archive" / file.name)

    def get_msg_content_clean(self, message: Message) -> str:
        message_content = convert_mentions_emotes(
            text=message.content,
            users=message.mentions,
            emojis=self.bot.emojis,
        )
        message_content = re_angle_bracket.sub("", message_content)
        return message_content.lstrip()


def setup(bot):
    bot.add_cog(Ai(bot))