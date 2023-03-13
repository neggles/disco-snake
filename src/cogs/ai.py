import json
import logging
import re
from random import randint
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
from traceback import format_exc

from dacite import from_dict
from disnake import (
    ApplicationCommandInteraction,
    Role,
    Message,
    Guild,
    DMChannel,
    Embed,
    File,
    TextChannel,
)
from disnake.ext import commands
from shimeji import ChatBot
from shimeji.memory import array_to_str, memory_context
from shimeji.memorystore_provider import PostgreSQLMemoryStore
from shimeji.model_provider import (
    SukimaModel,
    ModelGenArgs,
    ModelSampleArgs,
    ModelLogitBiasArgs,
    ModelPhraseBiasArgs,
    ModelGenRequest,
)
from shimeji.postprocessor import NewlinePrunerPostprocessor
from shimeji.preprocessor import ContextPreprocessor
from shimeji.util import (
    INSERTION_TYPE_NEWLINE,
    TRIM_DIR_TOP,
    TRIM_TYPE_SENTENCE,
    TRIM_TYPE_NEWLINE,
    ContextEntry,
)
from transformers import GPT2Tokenizer

import logsnake
from cogs.common import utils, MessageChannel
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
    fileLoglevel=logging.INFO,
    maxBytes=2 * (2**20),
    backupCount=2,
)


# some dataclasses
@dataclass
class MemoryStoreConfig:
    database_uri: str
    model: str
    model_layer: int
    short_term_amount: int
    long_term_amount: int


@dataclass
class ModelProviderConfig:
    endpoint: str
    username: str
    password: str
    gensettings: dict


@dataclass
class ChatbotConfig:
    name: str
    prompt: str
    postprompt: str
    conditional_response: bool
    idle_messaging: bool
    idle_messaging_interval: int
    nicknames: List[str]
    context_size: int
    logging_channel_id: int
    debug: bool
    memory_store: MemoryStoreConfig
    model_provider: ModelProviderConfig


def get_item(obj, key):
    if key in obj:
        return obj[key]
    else:
        return None


def get_sukima_model(cfg: ModelProviderConfig) -> SukimaModel:
    # load model provider gen_args into basemodel
    gen_args = ModelGenArgs(
        max_length=get_item(cfg.gensettings["gen_args"], "max_length"),
        max_time=get_item(cfg.gensettings["gen_args"], "max_time"),
        min_length=get_item(cfg.gensettings["gen_args"], "min_length"),
        eos_token_id=get_item(cfg.gensettings["gen_args"], "eos_token_id"),
        logprobs=get_item(cfg.gensettings["gen_args"], "logprobs"),
        best_of=get_item(cfg.gensettings["gen_args"], "best_of"),
    )
    # logit biases are an array in args['model_provider']['gensettings']['logit_biases']
    logit_biases = None
    if "logit_biases" in cfg.gensettings["sample_args"]:
        logit_biases = [
            ModelLogitBiasArgs(id=logit_bias["id"], bias=logit_bias["bias"])
            for logit_bias in cfg.gensettings["sample_args"]["logit_biases"]
        ]
    # phrase biases are an array in args['model_provider']['gensettings']['phrase_biases']
    phrase_biases = None
    if "phrase_biases" in cfg.gensettings["sample_args"]:
        phrase_biases = [
            ModelPhraseBiasArgs(
                sequences=phrase_bias["sequences"],
                bias=phrase_bias["bias"],
                ensure_sequence_finish=phrase_bias["ensure_sequence_finish"],
                generate_once=phrase_bias["generate_once"],
            )
            for phrase_bias in cfg.gensettings["sample_args"]["phrase_biases"]
        ]

    sample_args = ModelSampleArgs(
        temp=get_item(cfg.gensettings["sample_args"], "temp"),
        top_p=get_item(cfg.gensettings["sample_args"], "top_p"),
        top_a=get_item(cfg.gensettings["sample_args"], "top_a"),
        top_k=get_item(cfg.gensettings["sample_args"], "top_k"),
        typical_p=get_item(cfg.gensettings["sample_args"], "typical_p"),
        tfs=get_item(cfg.gensettings["sample_args"], "tfs"),
        rep_p=get_item(cfg.gensettings["sample_args"], "rep_p"),
        rep_p_range=get_item(cfg.gensettings["sample_args"], "rep_p_range"),
        rep_p_slope=get_item(cfg.gensettings["sample_args"], "rep_p_slope"),
        bad_words=get_item(cfg.gensettings["sample_args"], "bad_words"),
        logit_biases=logit_biases,
        phrase_biases=phrase_biases,
    )

    request = ModelGenRequest(
        model=cfg.gensettings["model"],
        prompt="",
        sample_args=sample_args,
        gen_args=gen_args,
    )

    return SukimaModel(
        endpoint_url=cfg.endpoint,
        username=cfg.username,
        password=cfg.password,
        args=request,
    )


def get_role_by_name(guild: Guild, name: str) -> Role:
    for role in guild.roles:
        if role.name == name:
            return role
    return None


class Ai(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot

        # Load config file
        self.cfg_path: Path = DATADIR_PATH.joinpath("ai", "config.json")
        try:
            self.config = from_dict(data_class=ChatbotConfig, data=json.loads(self.cfg_path.read_text()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.cfg_path}")

        # Parse config file
        self.memory_store_cfg: MemoryStoreConfig = self.config.memory_store
        self.model_provider_cfg: ModelProviderConfig = self.config.model_provider

        # we will populate these later during async init
        self.memory_store: PostgreSQLMemoryStore = None
        self.model_provider: SukimaModel = None
        self.chatbot: ChatBot = None
        self.logging_channel: TextChannel = None
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
        self.model_provider: SukimaModel = get_sukima_model(cfg=self.model_provider_cfg)
        self.chatbot = ChatBot(
            name=self.name,
            model_provider=self.model_provider,
            preprocessors=[ContextPreprocessor(self.config.context_size)],
            postprocessors=[NewlinePrunerPostprocessor()],
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.cfg_path.parent.joinpath("tokenizer").as_posix(),
        )

        logger.info("AI engine initialized... probably?")

    # build a context from the last 40 messages in the channel
    async def get_msg_ctx(self, channel: MessageChannel) -> str:
        messages = await channel.history(limit=40).flatten()
        messages, to_remove = utils.anti_spam(messages)
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
                content = utils.convert_mentions_emotes(
                    message.content, message.guild.members, message.guild.emojis
                )
                content = re.sub(r"\<[^>]*\>", "", message.content.lstrip().rstrip()).lstrip().rstrip()
                if content != "" and not "```" in content:
                    chain.append(f"{message.author.name}: {content}")
                continue
            elif message:
                chain.append(f"{message.author.name}: [Image attached]")
        return "\n".join(chain)

    async def build_ctx(self, conversation: str):
        contextmgr = ContextPreprocessor(token_budget=self.config.context_size, tokenizer=self.tokenizer)

        prompt_entry = ContextEntry(
            text=self.prompt,
            prefix="",
            suffix="\n<START>",
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
                    insertion_position=0,
                    trim_direction=TRIM_DIR_TOP,
                    trim_type=TRIM_TYPE_SENTENCE,
                    insertion_type=INSERTION_TYPE_NEWLINE,
                    forced_activation=True,
                    cascading_activation=False,
                )
                contextmgr.add_entry(memories_entry)

        # conversation
        conversation_entry = ContextEntry(
            text=conversation,
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

        return contextmgr.context(self.config.context_size)

    async def respond(self, conversation: str, message: Message):
        did_respond = False
        async with message.channel.typing():
            encoded_image_label = ""
            debug_data = {}

            if message.guild is not None:
                map_users = message.channel.members
                map_emojis = list(message.guild.emojis)
            else:
                map_users = [message.author, self.bot.user]
                map_emojis = []

            debug_data["message"] = {
                "id": message.id,
                "author": message.author.name + "#" + message.author.discriminator,
                "guild": message.guild.name if message.guild is not None else "DM",
                "channel": message.channel.name if message.guild is not None else "DM",
                "timestamp": message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "content": message.content,
            }

            if self.memory_store is not None:
                encoded_user_message = ""
                private_role = get_role_by_name(guild=message.guild, name="Private")
                if private_role is not None:
                    # check if the user has the private role, if the user does, don't encode
                    if private_role not in message.author.roles:
                        message_content = utils.convert_mentions_emotes(
                            text=message.content, users=map_users, emojis=map_emojis
                        )
                        message_content = (
                            re.sub(r"\<[^>]*\>", "", message_content.lstrip().rstrip()).lstrip().rstrip()
                        )
                        author_name = message.author.name
                        encoded_user_message = f"{message.author.name}: {message_content}"
                        is_dupe = await self.memory_store.check_duplicates(
                            text=encoded_user_message, duplicate_ratio=0.8
                        )
                        if not is_dupe:
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
            response = response.replace("<USER>", message.author.mention)
            response = response.replace("<BOT>", self.name)

            # Clean response - trim left whitespace and fix emojis and pings
            response = utils.cut_trailing_sentence(response)
            response = response.lstrip()
            if message.guild:
                response = utils.restore_mentions_emotes(text=response, users=map_users, emojis=map_emojis)
            debug_data["response"] = response

            # Send response if not empty
            if response == "":
                logger.info(f"Response was empty.")
            else:
                logger.info(f"Response: {response}")
                did_respond = True
                await message.channel.send(response)

        if self.config.debug and did_respond:
            dump_file = self.debug_datadir.joinpath(f"{message.created_at.isoformat()}-{message.id}.json")
            dump_file.write_text(json.dumps(debug_data, indent=4, skipkeys=True, default=str))

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

        mentioned = (
            True
            if self.bot.user in message.mentions
            or self.bot.user.display_name.lower() in message.content.lower()
            else False
        )
        direct = True if isinstance(message.channel, DMChannel) else False

        if not mentioned and not direct:
            if message.thread is not None:
                # Don't respond to threads
                return

        # for now only respond to direct messages from owners and in approved guilds
        if direct:
            if message.author.id not in self.bot.config["owners"]:
                logger.info(
                    f"Got a DM from non-owner {message.author.name}#{message.author.discriminator}. Ignoring..."
                )
            return
        elif message.guild.id not in self.bot.config["ai_guilds"]:
            return

        try:
            logger.info(f"Raw message: {message.content}")
            conversation = await self.get_msg_ctx(message.channel)
            if message.channel.guild:
                message_content = utils.convert_mentions_emotes(
                    text=message.content,
                    users=message.guild.members,
                    emojis=list(message.guild.emojis),
                )
            else:
                message_content = re.sub(r"\<[^>]*\>", "", message.content.lower())

            logger.info(f"Message: {message_content}")

            if self.config.conditional_response is True:
                if self.bot.user.mentioned_in(message) or any(
                    t in message_content for t in self.config.nicknames
                ):
                    await self.respond(conversation, message)
                elif await self.chatbot.should_respond_async(conversation, push_chain=False):
                    await self.respond(conversation, message)
            else:
                if self.bot.user.mentioned_in(message) or any(
                    t in message_content for t in self.config.nicknames
                ):
                    await self.respond(conversation, message)
                elif isinstance(message.channel, DMChannel):
                    await self.respond(conversation, message)

        except Exception as e:
            logger.error(e)
            logger.error(format_exc())
            embed = Embed(
                title="**Exception**",
                description=str(f"**``{repr(e)}``**\n```{format_exc()}```"),
            )
            if self.config.logging_channel_id == 0:
                await message.channel.send(embed=embed, delete_after=15.0)
            else:
                if self.logging_channel is None:
                    self.logging_channel = self.bot.get_channel(self.config.logging_channel_id)
                if len(str(f"**Exception:** **``{repr(e)}``**\n```{format_exc()}```")) < 5900:
                    await self.logging_channel.send(embed=embed)
                else:
                    errorfile = open("error.txt", "w")
                    errorfile.write(f"**Exception:** **``{repr(e)}``**\n```{format_exc()}```")
                    errorfile.close()
                    embed = Embed(
                        title="**Exception**",
                        description=str("The error is too large, check the attached file"),
                    )
                    await self.logging_channel.send(embed=embed, file=File("error.txt"))


def setup(bot):
    bot.add_cog(Ai(bot))
