import json
import logging
import re
from asyncio import sleep
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, Dict, List, Optional, Union

from dacite import from_dict
from disnake import (
    DMChannel,
    Embed,
    File,
    GroupChannel,
    Message,
    TextChannel,
    Thread,
    User,
    MessageInteraction,
)
from disnake.ext import commands, tasks
from Levenshtein import distance as lev_distance
from shimeji import ChatBot
from shimeji.memory import array_to_str, memory_context
from shimeji.memory.providers import PostgresMemoryStore
from shimeji.model_provider import EnmaModel, OobaModel
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.preprocessor import ContextPreprocessor
from shimeji.util import (
    INSERTION_TYPE_NEWLINE,
    TRIM_DIR_TOP,
    TRIM_TYPE_NEWLINE,
    TRIM_TYPE_SENTENCE,
    ContextEntry,
)
from transformers import LlamaTokenizerFast
from transformers.utils import logging as transformers_logging

import logsnake
from ai.config import ChatbotConfig, MemoryStoreConfig, ModelProviderConfig
from ai.model import get_enma_model, get_ooba_model
from ai.types import MessageChannel
from ai.utils import (
    anti_spam,
    convert_mentions_emotes,
    cut_trailing_sentence,
    get_full_class_name,
    get_role_by_name,
    restore_mentions_emotes,
    get_lm_prompt_time,
)
from ai.imagen import Imagen
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

COG_UID = "ai"


# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=3,
)

re_angle_bracket = re.compile(r"\<([^>]*)\>")
re_user_token = re.compile(r"(<USER>|<user>|{{user}})")
re_bot_token = re.compile(r"(<BOT>|<bot>|{{bot}}|<CHAR>|<char>|{{char}})")
re_unescape_format = re.compile(r"\\([*_~`])")
re_strip_special = re.compile(r"[^a-zA-Z0-9]+")


class Ai(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.timezone = self.bot.timezone
        self.last_response = datetime.now(timezone.utc) - timedelta(minutes=10)

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
        self.memory_enable = self.config.params.memory_enable

        # we will populate these later during async init
        self.memory_store: PostgresMemoryStore = None
        self.model_provider: Union[OobaModel, EnmaModel] = None
        self.model_provider_type: str = self.model_provider_cfg.type
        self.chatbot: ChatBot = None
        self.logging_channel: Optional[TextChannel] = None
        self.guilds: dict = {}
        self.tokenizer: LlamaTokenizerFast = None
        self.bad_words: List[str] = self.config.bad_words

        # selfietron
        self.imagen = Imagen()

        # somewhere to put the last context we generated for debugging
        self.debug_datadir: Path = LOGDIR_PATH.joinpath("ai")
        self.debug_datadir.mkdir(parents=True, exist_ok=True)

        if self.debug is True:
            transformers_logging.set_verbosity_debug()
        else:
            transformers_logging.set_verbosity_info()

    # Getters for config object sub-properties
    @property
    def name(self) -> str:
        return self.config.name

    def get_prompt(self, ctx: Optional[MessageInteraction] = None) -> str:
        if ctx is None:
            location_context = f'with your friends in the "{self.bot.home_guild.name}" Discord server'
        else:
            if ctx.guild is not None:
                location_context = f'with your friends in the "{ctx.guild.name}" Discord server'
            elif ctx.channel is not None:
                location_context = (
                    f'with your friends in the "{ctx.channel.name}" channel of a Discord server'
                )
            elif ctx.user is not None:
                location_context = f"with {ctx.user.name} in a Discord DM"

        return (
            self.config.prompt.replace("{bot_name}", self.name)
            .replace("{location_context}", location_context)
            .replace("{current_time}", get_lm_prompt_time())
        )

    async def cog_load(self) -> None:
        logger.info("AI engine initializing, please wait...")
        if self.memory_enable:
            # Set up MemoryStoreProvider
            logger.debug("Memory Store is enabled, initializing...")
            self.memory_store = PostgresMemoryStore(
                database_uri=self.memory_store_cfg.database_uri,
                model=self.memory_store_cfg.model,
                model_layer=self.memory_store_cfg.model_layer,
                short_term_amount=self.memory_store_cfg.short_term_amount,
                long_term_amount=self.memory_store_cfg.long_term_amount,
            )
        else:
            logger.debug("Memory Store is disabled, skipping...")
            self.memory_store = None

        if self.model_provider_type == "ooba":
            self.model_provider = get_ooba_model(cfg=self.model_provider_cfg)
        elif self.model_provider_type == "enma":
            self.model_provider = get_enma_model(cfg=self.model_provider_cfg)
        else:
            raise ValueError(f"Unknown model provider type: {self.model_provider_type}")

        logger.debug("Initializing ChatBot object")
        self.chatbot = ChatBot(
            name=self.name,
            model_provider=self.model_provider,
            preprocessors=[ContextPreprocessor(self.context_size)],
            postprocessors=[NewlinePrunerPostprocessor()],
        )

        logger.debug("Initializing Tokenizer...")
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.cfg_path.parent.joinpath("tokenizer").as_posix(),
            local_files_only=True,
        )

        logger.debug("Setting logging channel...")
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

    @commands.Cog.listener("on_ready")
    async def on_ready(self):
        if self.logging_channel is None and self.logging_channel_id is not None:
            logger.info("Logging channel not found, attempting to find it...")
            self.logging_channel = self.bot.get_channel(self.logging_channel_id)
            logger.info(f"Logging channel found: {self.logging_channel}")
        logger.info("Cog is ready.")

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
                await self.respond(conversation, message, "mention")

            elif isinstance(message.channel, DMChannel) and len(message_content) > 0:
                await self.respond(conversation, message, "DM")

            elif self.conditional_response is True:
                if await self.chatbot.should_respond_async(conversation, push_chain=False):
                    logger.debug("Model wants to respond, responding...")
                    await self.respond(conversation, message, "conditional")
                    return
                else:
                    logger.debug("No conditional response.")

            elif message.channel.id in self.activity_channels and self.last_response < datetime.now(
                timezone.utc
            ) - timedelta(seconds=(self.idle_messaging_interval / 2)):
                await self.respond(conversation, message, "activity")

        except Exception as e:
            logger.error(e)
            logger.error(format_exc())
            exc_class = get_full_class_name(e)
            if exc_class == "HTTPException":
                # don't bother, the backend is down
                return

            exc_desc = str(f"**``{exc_class}``**\n```{format_exc()}```")
            error_file = self.debug_datadir.joinpath(f"error-{datetime.now(timezone.utc)}.txt")
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

    # get the last 40 messages in the channel (or however many there are if less than 40)
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
            elif message.content:
                content = self.get_msg_content_clean(message)
                if content != "" and "```" not in content:
                    chain.append(f"{message.author.name}: {content}")
                continue
            elif message:
                chain.append(f"{message.author.name}: pic.png")
        chain = [x.strip() for x in chain if x.strip() != ""]
        return "\n".join(chain)

    # assemble a prompt/context for the model
    async def build_ctx(self, conversation: str, message: Optional[Message] = None):
        contextmgr = ContextPreprocessor(token_budget=self.context_size, tokenizer=self.tokenizer)

        prompt_entry = ContextEntry(
            text=self.get_prompt(message),
            prefix="",
            suffix="\n",
            reserved_tokens=512,
            insertion_order=1000,
            insertion_position=-1,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
        )
        contextmgr.add_entry(prompt_entry)

        # memories
        if self.memory_store is not None and self.memory_enable is True:
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
                    suffix="</s>\n",
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
        conv_prefix = "</s>\n" if self.memory_store is not None and self.memory_enable is True else ""
        conversation_entry = ContextEntry(
            text=conversation,
            prefix=conv_prefix,
            suffix="",
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

    # actual response logic
    async def respond(self, conversation: str, message: Message, trigger: str = None) -> str:
        async with message.channel.typing():
            encoded_image_label = ""
            debug_data: Dict[str, Any] = {}

            msg_timestamp = message.created_at.astimezone(tz=self.bot.timezone).strftime(
                "%Y-%m-%d-%H:%M:%S%z"
            )
            msg_trigger = trigger.lower() if trigger is not None else "unknown"

            debug_data["message"] = {
                "id": message.id,
                "author": f"{message.author}",
                "guild": f"{message.guild}" if message.guild is not None else "DM",
                "channel": message.channel.name if not isinstance(message.channel, DMChannel) else "DM",
                "timestamp": msg_timestamp,
                "trigger": msg_trigger,
                "content_raw": message.content,
            }

            try:
                debug_data["gensettings"] = asdict(self.model_provider_cfg)["gensettings"]
                # remove bad_words from debug data because it's long and irrelevant
            except Exception as e:
                logger.error(f"Failed to get gensettings: {e}\n{format_exc()}")

            if self.imagen.should_take_pic(message.content) is True:
                logger.info("Hold up, let me take a selfie...")
                try:
                    response_image = await self.take_pic(message=message)
                except Exception as e:
                    logger.error(f"Failed to generate image label: {e}\n{format_exc()}")
                    response_image = None
            else:
                response_image = None

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
                    if self.is_memorable(message_content) and not is_dupe:
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
            conversation = await self.build_ctx(conversation + encoded_image_label, message)
            debug_data["context"] = conversation.splitlines()

            # Generate response
            try:
                response: str = await self.chatbot.respond_async(conversation, push_chain=False)
                debug_data["response_raw"] = response

                # replace "<USER>" with user mention, same for "<BOT>"
                response = self.fixup_bot_user_tokens(response, message)

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

                # Send response if not empty
                if response == "":
                    logger.info("Response was empty.")
                elif response_image is not None:
                    logger.info("Responding with image...")
                    await message.channel.send(response, file=response_image)
                    logger.info(f"Response: {response}")
                else:
                    await message.channel.send(response)
                    logger.info(f"Response: {response}")

                    self.last_response = datetime.now(timezone.utc)

                # add to memory store in background
                if self.memory_store is not None:
                    # encode bot response
                    response = self.get_msg_content_clean(message, response)
                    is_dupe = await self.memory_store.check_duplicates(text=response, duplicate_ratio=0.8)
                    if response != "" and not is_dupe and self.is_memorable(response):
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
            except Exception as e:
                logger.exception(e)
            finally:
                if self.debug:
                    dump_file = self.debug_datadir.joinpath(f"msg-{message.id}-{msg_timestamp}.json")
                    dump_file.write_text(json.dumps(debug_data, indent=4, skipkeys=True, default=str))
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

    # Helper functions
    def get_msg_content_clean(self, message: Message, content: str = None) -> str:
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
        content = re_angle_bracket.sub("", content)
        # if there's a codeblock in there, return an empty string
        return content.lstrip() if "```" not in content else ""

    def fixup_bot_user_tokens(self, response: str, message: Message) -> str:
        """
        Fix <USER>, <BOT>, etc tokens in the response, and unescape any escaped markdown formatting
        """
        response = re_user_token.sub(f"@{message.author.name}", response)
        response = re_bot_token.sub(f"{self.name}", response)
        response = re_unescape_format.sub(r"\1", response)
        return response

    def is_memorable(self, text: str) -> bool:
        """
        Don't memorize messages with [] tags in them or code blocks etc
        """
        for string in self.model_provider_cfg.gensettings["sample_args"]["bad_words"]:
            if string in text:
                return False
        return True if text != "" else False

    def like_bad_word(self, input: str) -> str:
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

    # Loop stuff to archive logs
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

    # Image generation stuff
    async def take_pic(self, message: Message) -> File:
        """
        hold up, let me take a selfie
        """
        # get the message content
        message_content = self.get_msg_content_clean(message).lower()
        if message_content == "":
            logger.info("Message was empty after cleaning.")
            return ""

        # get the image description and remove newlines
        message_content = message_content.replace("\n", " ").replace(self.name + " ", "")

        # build the LLM prompt for the image
        lm_prompt = self.imagen.get_lm_prompt(self.imagen.strip_take_pic(message_content))
        logger.info(f"[take_pic] LLM Prompt: {lm_prompt}")

        # get the LLM to create tags for the image
        lm_tags = await self.chatbot.respond_async(lm_prompt, push_chain=False, is_respond=False)
        logger.info(f"[take_pic] LLM Tags: {lm_tags}")

        # build the SD API request
        sdapi_request = self.imagen.build_request(lm_tags, message_content)
        logger.info(f"[take_pic] SD API Request: {json.dumps(sdapi_request)}")
        # submit it
        result_path = await self.imagen.submit_request(sdapi_request)
        # drop the meta next to it
        result_path.with_suffix(".json").write_text(
            json.dumps(sdapi_request, indent=4, skipkeys=True, default=str)
        )
        # return a discord File object to upstream
        result_file = File(result_path, filename=result_path.name)
        return result_file


def setup(bot):
    bot.add_cog(Ai(bot))
