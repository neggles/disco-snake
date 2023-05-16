import asyncio
import json
import logging
import re
from asyncio import Lock
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice as random_choice
from traceback import format_exc
from typing import Any, Dict, List, Optional, Tuple, Union

from dacite import from_dict
from disnake import (
    DMChannel,
    Embed,
    File,
    GroupChannel,
    Message,
    MessageInteraction,
    TextChannel,
    Thread,
    User,
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
    TRIM_DIR_NONE,
    TRIM_DIR_TOP,
    TRIM_TYPE_NEWLINE,
    ContextEntry,
)
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.utils import logging as transformers_logging

import logsnake
from ai.config import ChatbotConfig, MemoryStoreConfig, ModelProviderConfig
from ai.eyes import DiscoEyes
from ai.imagen import Imagen
from ai.model import get_enma_model, get_ooba_model
from ai.types import MessageChannel
from ai.utils import (
    anti_spam,
    any_in_text,
    convert_mentions_emotes,
    cut_trailing_sentence,
    get_full_class_name,
    get_lm_prompt_time,
    get_role_by_name,
    member_in_any_role,
    restore_mentions_emotes,
)
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake

COG_UID = "ai"

logger = logsnake.setup_logger(
    name=__name__,
    level=logging.DEBUG,
    isRootLogger=False,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{__name__}.log"),
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
re_start_expression = re.compile(r"^\s*\(\w+\)\s*", re.I + re.M)


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
        self.params = self.config.params

        self.activity_channels: List[int] = self.config.params.activity_channels
        self.conditional_response: bool = self.config.params.conditional_response
        self.context_size: int = self.config.params.context_size
        self.context_messages: int = self.config.params.context_messages
        self.idle_msg_sec: int = self.config.params.idle_messaging_interval
        self.idle_messaging: bool = self.config.params.idle_messaging
        self.logging_channel_id: int = self.config.params.logging_channel_id
        self.nicknames: List[str] = self.config.params.nicknames
        self.debug: bool = self.config.params.debug
        self.memory_enable = self.config.params.memory_enable
        self.max_retries = self.config.params.max_retries
        self.ctxbreak_users = self.config.params.ctxbreak_users
        self.ctxbreak_roles = self.config.params.ctxbreak_roles

        self.ctx_lock = Lock()  # used to stop multiple context builds from happening at once
        self.lm_lock = Lock()  # used to stop multiple conditional responses from happening at once

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
        self.imagen = Imagen(lm_api_host=self.model_provider_cfg.endpoint)
        self.eyes = DiscoEyes(config=self.config.vision)

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

    # retrieve the LM prompt and inject name, time, etc.
    def get_prompt(self, ctx: Optional[MessageInteraction] = None) -> str:
        if ctx is None:
            location_context = "and her friends in a Discord server"
        elif hasattr(ctx, "guild") and ctx.guild is not None:
            location_context = f"and her friends in the {ctx.guild.name} Discord server"
        elif hasattr(ctx, "author") and ctx.author is not None:
            location_context = f"and {ctx.author.display_name} in a Discord DM"
        else:
            location_context = "and a friend in a Discord DM"

        if isinstance(self.config.prompt, List):
            prompt = "\n".join(self.config.prompt)
        else:
            prompt = self.config.prompt

        replace_tokens = {
            "bot_name": self.name,
            "location_context": location_context,
            "current_time": get_lm_prompt_time(),
        }
        for token, replacement in replace_tokens.items():
            prompt = prompt.replace(f"{{{token}}}", replacement)

        return prompt

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

        logger.debug("Initializing Tokenizer...")
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.cfg_path.parent.joinpath("tokenizers/llama").as_posix(),
            local_files_only=True,
        )

        logger.debug("Initializing Model Provider...")
        if self.model_provider_type == "ooba":
            self.model_provider = get_ooba_model(cfg=self.model_provider_cfg, tokenizer=self.tokenizer)
        elif self.model_provider_type == "enma":
            self.model_provider = get_enma_model(cfg=self.model_provider_cfg, tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Unknown model provider type: {self.model_provider_type}")

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
            self.logging_channel = self.bot.get_channel(self.logging_channel_id)
            logger.info(f"Logging channel: {self.logging_channel}")
        logger.info("Cog is ready.")

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if (message.author.bot is True) or (message.author == self.bot.user):
            return
        if "```" in message.content:
            return

        # Ignore messages from unapproved users in DMs/groups
        if isinstance(message.channel, (DMChannel, GroupChannel)):
            if message.author.id not in self.bot.config["owner_ids"]:
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
            # logger.debug(f"Raw message: {message.content}")
            message_content = self.get_msg_content_clean(message)
            if message_content == "":
                logger.debug("Message was empty after cleaning.")
                return

            async with self.lm_lock:
                trigger = None
                conversation = None

                if self.bot.user.mentioned_in(message) or self.name_in_text(message_content):
                    logger.debug(f"Message: {message_content}")
                    trigger = "mention"

                elif isinstance(message.channel, DMChannel) and len(message_content) > 0:
                    trigger = "DM"

                elif message.channel.id in self.activity_channels:
                    if self.conditional_response is True:
                        conversation = await self.get_context_messages(message.channel)
                        if await self.chatbot.should_respond_async(conversation, push_chain=False):
                            logger.debug(f"Model wants to respond to '{message_content}', responding...")
                            trigger = "conditional"

                    elif self.last_response < (
                        datetime.now(tz=self.timezone) - timedelta(seconds=(self.idle_msg_sec / 2))
                    ):
                        # prevent infinite loops if things go wrong
                        self.last_response = datetime.now(tz=self.timezone)
                        trigger = "activity"

                if trigger is not None:
                    if conversation is None:
                        conversation = await self.get_context_messages(message.channel)
                    await self.respond(conversation, message, trigger)

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
    async def get_context_messages(self, channel: MessageChannel) -> str:
        messages = await channel.history(limit=50).flatten()
        messages, to_remove = anti_spam(messages)
        if to_remove:
            logger.info(f"Removed {to_remove} messages from context.")

        # magic tag to break context chain to get bot out of librarian mode
        for idx, message in enumerate(messages):
            if "<ctxbreak>" in message.content.lower() and (
                message.author.id in self.ctxbreak_users
                or member_in_any_role(message.author, self.ctxbreak_roles)
            ):
                messages = messages[:idx]
                break

        chain = []
        for message in reversed(messages):
            # set up author name
            if message.author.bot is False:
                author_name = f"### Instruction: {message.author.display_name.strip()}"
            elif message.author.id == self.bot.user.id:
                author_name = f"### Response: {self.name}"
            else:
                logger.debug("Skipping non-self bot message")
                continue

            if len(message.embeds) > 0:
                for embed in message.embeds:
                    if embed.type == "image":
                        try:
                            caption = await self.eyes.perceive_image_embed(embed.thumbnail.url)
                            if caption is not None:
                                chain.append(f"{author_name}: [image: {caption}]")
                        except Exception as e:
                            logger.exception(e)
                            chain.append(f"{author_name}: [image: loading error]")
                    elif embed.description is not None:
                        chain.append(f"{author_name}: [embed: {embed.description}]")
                    elif embed.title is not None:
                        chain.append(f"{author_name}: [embed: {embed.title}]")

            if message.content and len(message.embeds) == 0:
                content = self.get_msg_content_clean(message)
                if content.startswith("http"):
                    chain.append(f"{author_name}: [a link to a webpage]")
                elif content != "" and "```" not in content and not content.startswith("-"):
                    if message.author.id == self.bot.user.id:
                        # strip (laughs) from start of own context
                        content = re_start_expression.sub("", content)
                        chain.append(f"{author_name}: {content}")
                    else:
                        chain.append(f"{author_name}: {content}")

            for attachment in message.attachments:
                try:
                    caption = await self.eyes.perceive_attachment(attachment)
                    if caption is not None:
                        chain.append(f"{author_name}: [image: {caption}]")
                except Exception as e:
                    logger.exception(e)
                    chain.append(f"{author_name}: [image: loading error]")

        chain = [x.strip() for x in chain if x.strip() != ""]
        return "\n".join(chain)

    # assemble a prompt/context for the model
    async def build_ctx(self, conversation: str, message: Optional[Message] = None):
        contextmgr = ContextPreprocessor(token_budget=self.context_size, tokenizer=self.tokenizer)

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

        # conversation
        conversation_entry = ContextEntry(
            text=conversation + f"\n### Response: {self.name}:",
            prefix="",
            suffix="",
            reserved_tokens=512,
            insertion_order=0,
            insertion_position=-1,
            trim_direction=TRIM_DIR_TOP,
            trim_type=TRIM_TYPE_NEWLINE,
            insertion_type=INSERTION_TYPE_NEWLINE,
            forced_activation=True,
            cascading_activation=False,
            tokenizer=self.tokenizer,
        )
        contextmgr.add_entry(conversation_entry)

        return contextmgr.context(self.context_size)

    # actual response logic
    async def respond(self, conversation: str, message: Message, trigger: str = None) -> str:
        async with message.channel.typing():
            debug_data: Dict[str, Any] = {}

            response = ""
            response_image = None
            author_name = message.author.display_name.strip()

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
                "conversation": conversation.splitlines(),
            }

            try:
                debug_data["gensettings"] = asdict(self.model_provider_cfg)["gensettings"]
            except Exception as e:
                logger.exception("Failed to get gensettings")

            # Build context
            context = await self.build_ctx(conversation, message)
            debug_data["context"] = conversation.splitlines()

            if self.imagen.should_take_pic(message.content) is True:
                logger.info("Hold up, let me take a selfie...")
                try:
                    response_image = await self.take_pic(message=message)
                except Exception as e:
                    raise Exception("Failed to generate image response") from e

            try:
                # Generate the response, and retry if it contains bad words (up to self.max_retries times)
                for attempt in range(self.max_retries):
                    attempt = attempt + 1  # deal with range() starting from 0
                    response: str = await self.chatbot.respond_async(context, push_chain=False)
                    bad_words = self.find_bad_words(response)
                    if len(bad_words) == 0:
                        if response.lower().startswith("i'm sorry, but") is True:
                            logger.info(f"Response was a ChatGPT apology: {response}\nRetrying...")
                            continue
                        if "as a language model" in response.lower():
                            logger.info(f"Response admits to being an AI: {response}\nRetrying...")
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

                debug_data["response_raw"] = response

                # if bot did a "\n<someusername:" cut it off
                if bool(re_linebreak_name.match(response)):
                    response = response.splitlines()[0]

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
                if response_image is not None:
                    if response == "":
                        logger.info("Response was empty, sending image only.")
                        await message.channel.send(file=response_image)
                    else:
                        logger.info(f"Responding with image, response: {response}")
                        await message.channel.send(response, file=response_image)
                elif response == "":
                    logger.info("Response was empty.")
                    if not any_in_text(["conditional", "activity"], trigger):
                        await message.channel.send("...")
                    await message.add_reaction("")
                else:
                    await message.channel.send(response)
                    logger.info(f"Response: {response}")

                    self.last_response = datetime.now(timezone.utc)

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
                if idle_sec >= self.idle_msg_sec:
                    if self.get_msg_content_clean(message) == "":
                        return
                    logger.debug(f"Running idle response to message {message.id}")
                    # if it's been more than <idle_messaging_interval> sec, send a response
                    async with self.lm_lock:
                        conversation = await self.get_context_messages(message.channel)
                        await self.respond(conversation, message)
        else:
            return

    @idle_loop.before_loop
    async def before_idle_loop(self):
        logger.info("Idle loop waiting for bot to be ready")
        await self.bot.wait_until_ready()
        await asyncio.sleep(10)
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
        author_name = message.author.display_name.encode("utf-8").decode("ascii", errors="ignore").strip()
        response = re_user_token.sub(f"@{author_name}", response)
        response = re_bot_token.sub(f"@{self.name}", response)
        response = re_unescape_format.sub(r"\1", response)
        return response

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

    def name_in_text(self, text: str) -> bool:
        return any(
            [
                bool(re.search(rf"\b({name})({name[-1]}*)\b", text, re.I + re.M))
                for name in self.params.nicknames
            ]
        )

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
    async def take_pic(self, message: Message) -> Tuple[File, dict]:
        """
        hold up, let me take a selfie
        """
        # get the message content
        message_content = self.get_msg_content_clean(message).lower()
        if message_content == "":
            logger.debug("Message was empty after cleaning.")
            return ""

        # get the image description and remove newlines
        message_content = message_content.replace("\n", " ").replace(self.name + " ", "")

        # build the LLM prompt for the image
        lm_prompt = self.imagen.get_lm_prompt(self.imagen.strip_take_pic(message_content))
        logger.info(f"LLM Prompt: {lm_prompt}")

        # get the LLM to create tags for the image
        lm_tags = await self.imagen.submit_lm_prompt(lm_prompt)
        logger.info(f"LLM Tags: {lm_tags}")

        # build the SD API request
        sdapi_request = self.imagen.build_request(lm_tags, message_content)
        logger.info(f"SD API Request: {json.dumps(sdapi_request)}")

        # submit it
        result_path = await self.imagen.submit_request(sdapi_request)
        # return a discord File object to upstream
        result_file = File(result_path, filename=result_path.name)
        return result_file


def setup(bot):
    bot.add_cog(Ai(bot))
