import asyncio
import logging
import random
from pathlib import Path
from traceback import format_exc
from typing import List

import disnake
from disnake import ApplicationCommandInteraction, DMChannel, Message, MessageInteraction
from disnake.ext import commands, tasks
from transformers.utils import logging as t2logging

import logsnake
from aitextgen import aitextgen
from disco_snake.bot import DiscoSnake
from disco_snake.cli import DATADIR_PATH, LOGDIR_PATH
from helpers import checks
from helpers.misc import parse_log_level


MODELS_ROOT = DATADIR_PATH.joinpath("models")
EMBED_COLOR = 0xFF9D0B
NEWLINE = "\n"
MBYTE = 2**20

EMOJI = {
    "huggingface": "🤗",
    "disco": "💃",
    "snake": "🐍",
    "disco_snake": "💃🐍",
    "robot": "🤖",
    "confus": "😕",
    "thonk": "🤔",
}

logfmt = logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S")
logger: logging.Logger = logsnake.setup_logger(
    __name__,
    level=logging.INFO,
    logfile=str(LOGDIR_PATH.joinpath("ai.log").resolve()),
    maxBytes=MBYTE,
    formatter=logfmt,
    fileLoglevel=logging.DEBUG,
    backupCount=5,
)
logger.propagate = False

# set up the transformers logger
t2logging.set_verbosity_info()
t2logger = t2logging.get_logger("transformers")
for handler in logger.handlers:
    t2logger.addHandler(handler)


class ModelSelect(disnake.ui.Select):
    def __init__(self):
        model_folders = [x for x in MODELS_ROOT.iterdir() if x.is_dir()]
        options = [disnake.SelectOption(label=x.name, value=x.name) for x in model_folders]

        # The placeholder is what will be shown when no option is chosen
        # The min and max values indicate we can only pick one of the three options
        # The options parameter defines the dropdown options. We defined this above
        super().__init__(
            placeholder="Choose an available model:",
            min_values=1,
            max_values=1,
            options=options,
            row=2,
        )

    async def callback(self, inter: MessageInteraction):
        await inter.response.defer()
        await inter.component.stop()
        model_name = self.values[0]
        ai_cog: AiCog = inter.bot.get_cog("Ai")

        logger.debug(f"Selected model: {model_name}")
        try:
            await ai_cog.set_model(model_name=model_name, reinit=True)
        except Exception as e:
            embed = disnake.Embed(
                title="Error!",
                description=f"Failed initializing model '{model_name}'",
                color=0xFF0000,
            )
            embed.add_field(name="Model name:", value=model_name, inline=False)
            embed.add_field(name="Error:", value=str(e), inline=False)
            await inter.edit_original_message(embed=embed, content=None, view=None)
            return

        embed = disnake.Embed(
            title="Success!", description=f"Initialized model '{model_name}'", color=EMBED_COLOR
        )
        embed.add_field(name="Model name:", value=model_name, inline=False)
        embed.add_field(
            name="AI status:", value="Online" if ai_cog.ai is not None else "Offline", inline=False
        )
        embed.set_footer(text=f"Requested by {inter.author}", icon_url=inter.author.avatar.url)
        await inter.edit_original_message(embed=embed, content=None, view=None)


class AiSettingsView(disnake.ui.View):
    def __init__(self):
        super().__init__()

        self.add_item(ModelSelect())


class AiCog(commands.Cog, name="Ai"):
    def __init__(self, bot: DiscoSnake) -> None:
        super().__init__()
        self.bot: DiscoSnake = bot

        # somme objects
        self.ai: aitextgen = None
        self.model_name: str = None
        self.model_folder: Path = None

        # ai model generation lock - not sure if needed but
        self.lock = asyncio.Lock()

        # ai model generation settings
        self.instance_runs = 0
        self.prompt_queue = asyncio.Queue(maxsize=8)

        # load config
        config: dict = bot.config["ai"]
        if config["model_name"] is None:
            raise ValueError("No model name specified in config")
        conf_keys = config.keys()

        # set model
        self.model_name = config["model_name"]
        self.model_folder = MODELS_ROOT.joinpath(self.model_name)
        if not self.model_folder.is_dir():
            raise ValueError(f"Model folder {self.model_folder} does not exist")
        if not self.model_folder.joinpath("pytorch_model.bin").is_file():
            raise ValueError(f"Model folder {self.model_folder} does not contain a pytorch_model.bin file")

        # set log level
        if "log_level" in conf_keys:
            self.log_level = parse_log_level(config["log_level"])
        elif "log_level" in bot.config.keys():
            self.log_level = parse_log_level(bot.config["log_level"])
        else:
            self.log_level = logging.INFO

        # parse other config
        self.use_gpu: bool = config["use_gpu"] if "use_gpu" in conf_keys else False
        self.base_length: int = config["base_length"] if "base_length" in conf_keys else 100
        self.max_lines: int = config["max_lines"] if "max_lines" in conf_keys else 5
        self.temperature: float = config["temperature"] if "temperature" in conf_keys else 0.9
        self.response_chance: float = config["response_chance"] if "response_chance" in conf_keys else 0.5
        self.context_messages: int = config["context_messages"] if "context_messages" in conf_keys else 9
        self.warmup_channels: List[int] = (
            config["warmup_channels"] if "warmup_channels" in conf_keys else None
        )

    @property
    def verbose(self) -> bool:
        return True if self.log_level == logging.DEBUG else False

    @verbose.setter
    def verbose(self, value: bool) -> None:
        if value:
            self.log_level = logging.DEBUG
        else:
            self.log_level = logging.INFO

    @property
    def log_level(self) -> int:
        return self._log_level

    @log_level.setter
    def log_level(self, value: int) -> None:
        self._log_level = value
        logger.setLevel(value)
        t2logging.set_verbosity(value)

    async def cog_load(self) -> None:
        # TODO: Refactor model and generation out into a separate module/class that runs in a separate thread,
        # to prevent blocking the main thread when changing models or generating text.
        # bonus: model becomes accessible from other cogs! how to handle IPC? queues probably?
        await self.ai_init(reinit=True)
        await self.background_generate.start()
        return await super().cog_load()

    async def cog_unload(self) -> None:
        return await super().cog_unload()

    # Slash Commands

    @commands.slash_command(name="ai", description="Manage the AI")
    @checks.not_blacklisted()
    async def ai_group(self, inter: ApplicationCommandInteraction):
        pass

    @ai_group.sub_command(
        name="status",
        description="Returns the current AI model name and config.",
    )
    @checks.not_blacklisted()
    async def ai_status(self, inter: ApplicationCommandInteraction):
        """
        Returns the current AI model name and config.
        :param inter: The application command inter.
        """
        embed = disnake.Embed(description="Module State", color=EMBED_COLOR)
        embed.set_author(name="AiCog", icon_url=self.bot.user.avatar.url)
        embed.add_field(name="Model:", value=self.model_name, inline=False)
        if self.ai is not None:
            embed.add_field(name="Initialized:", value="Yes", inline=True)
            embed.add_field(name="Running on:", value=self.ai.get_device(), inline=True)
        else:
            embed.add_field(name="Initialized:", value="No", inline=True)
            embed.add_field(name="Running on:", value="N/A", inline=True)
        embed.add_field(name="Use GPU:", value="Yes" if self.use_gpu else "No", inline=True)
        embed.add_field(name="Log level:", value=str(self._log_level), inline=True)
        embed.add_field(name="Base length:", value=str(self.base_length), inline=True)
        embed.add_field(name="Max lines:", value=str(self.max_lines), inline=True)
        embed.add_field(name="Temperature:", value=str(self.temperature), inline=True)
        embed.add_field(name="Response Rate:", value=str(self.response_chance), inline=True)
        embed.add_field(name="Context:", value=str(self.context_messages) + " Messages", inline=True)
        embed.add_field(name="Instance runs:", value=str(self.instance_runs), inline=True)

        embed.set_footer(text=f"Requested by {inter.author}", icon_url=inter.author.avatar.url)
        await inter.send(embed=embed)

    @ai_group.sub_command(name="set-model", description="Change the active AI model.")
    @checks.is_owner()
    @checks.not_blacklisted()
    async def ai_set_model(self, inter: ApplicationCommandInteraction) -> None:
        """
        Change the active AI model.
        :param inter: The application command inter.
        """
        view = AiSettingsView()
        await inter.send("Please choose an available model:", view=view)

    # parameter tweak command group

    @ai_group.sub_command_group(name="params", description="Change the AI parameters.")
    async def ai_params(self, inter: ApplicationCommandInteraction):
        pass

    @ai_params.sub_command(name="temperature", description="Change the AI temperature parameter.")
    @checks.is_owner()
    async def ai_params_temperature(self, inter: ApplicationCommandInteraction, temperature: float) -> None:
        """
        Change the temperature of AI responses.
        :param inter: The application command interaction.
        :param temperature: The new temperature.
        """
        self.temperature = temperature
        await inter.send(f"Temperature set to {temperature}", delete_after=5.0, ephemeral=True)

    @ai_params.sub_command(name="base-length", description="Change max token length of generated responses.")
    @checks.is_owner()
    async def ai_params_base_length(self, inter: ApplicationCommandInteraction, length: int) -> None:
        """
        Change the base length of generated responses.
        :param inter: The application command interaction.
        :param length: The new base length.
        """
        self.base_length = length
        await inter.send(f"Base length set to {length}", delete_after=5.0, ephemeral=True)

    @ai_params.sub_command(name="max-lines", description="Change max number of lines in generated responses.")
    @checks.is_owner()
    async def ai_params_max_lines(self, inter: ApplicationCommandInteraction, lines: int) -> None:
        """
        Change the max number of lines in generated responses.
        :param inter: The application command interaction.
        :param lines: The max number of lines.
        """
        self.max_lines = lines
        await inter.response.send_message(f"Max lines set to {lines}", delete_after=5.0, ephemeral=True)

    @ai_params.sub_command(
        name="response-chance", description="Change chance of responding to a non-reply message."
    )
    @checks.is_owner()
    async def ai_params_response_chance(self, inter: ApplicationCommandInteraction, chance: float) -> None:
        """
        Change chance of responding to a non-reply message.
        :param inter: The application command interaction.
        :param chance: The chance of responding to a non-reply message from 0.0 to 1.0
        """
        self.response_chance = chance
        await inter.send(f"Response chance set to {chance}", delete_after=5.0, ephemeral=True)

    @ai_params.sub_command(
        name="context", description="Change the number of messages used for context in response generation."
    )
    @checks.is_owner()
    async def ai_params_context_messages(
        self,
        inter: ApplicationCommandInteraction,
        messages: commands.Range[0, 25],
    ) -> None:
        """
        Change the number of messages used for context in response generation.
        :param inter: The application command interaction.
        :param messages: The number of messages used for context in response generation.
        """
        messages = int(messages)
        self.context_messages = messages
        await inter.send(f"Prompt context set to {messages}", delete_after=5.0, ephemeral=True)

    # Event Listeners

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if message.author.bot or message.author == self.bot.user:
            return

        if self.ai is None:
            logger.debug("AI is not initialized but received a message")
            return

        # Don't bother replying to zero-content messages
        message_text = self.clean_input(message.content)
        if len(message_text) == 0:
            return

        mentioned = (
            True
            if self.bot.user in message.mentions or self.bot.user.display_name in message.content
            else False
        )

        direct = True if isinstance(message.channel, DMChannel) else False

        if random.random() > float(self.response_chance) and not mentioned and not direct:
            if random.random() > float(0.5) and not self.prompt_queue.full():
                logger.debug("Adding message to background prompt queue")
                history = await message.channel.history(limit=max(self.context_messages, 5)).flatten()
                history.reverse()
                context = [
                    self.clean_input(m.content.strip().replace("\n", ", "))
                    for m in history
                    if m.id != message.id
                ]
                context = [line for line in context if len(line) > 5 and "```" not in line]
                # make our prompt
                prompt = "\n".join(context) + "\n" + message_text
                self.prompt_queue.put_nowait(message)
            elif self.prompt_queue.full():
                logger.debug("Prompt queue is full.")
            return

        try:
            async with message.channel.typing():
                if direct:
                    history = await message.channel.history(limit=min(self.context_messages, 3)).flatten()
                else:
                    history = await message.channel.history(limit=min(self.context_messages, 25)).flatten()
                history.reverse()
                context = [
                    self.clean_input(m.content.strip().replace("\n", ", "))
                    for m in history
                    if m.id != message.id and len(m.content.strip()) > 5
                ]

                # make our prompt
                prompt = "\n".join(context) + "\n" + message_text
                # send it to the AI and get the response
                response = await self.generate_response(prompt)

                # if this is a DM, just send the response
                if direct:
                    await message.channel.send(response)
                # otherwise, send it in a reply
                else:
                    await message.reply(response)

        except Exception as e:
            logger.error("Error processing message content")
            logger.error(e)
        finally:
            logger.debug("Finished processing message content")
        return

    # AI functions

    async def ai_init(self, reinit: bool = False) -> None:
        async with self.lock:
            if self.ai is not None and reinit:
                logger.info("AI reinitialization requested, disposing old AI")
                del self.ai
                self.ai = None

            # this can't be an 'else' since we might have deleted the old ai
            if self.ai is None:
                try:
                    logger.info(f"Initializing AI model {self.model_name}")
                    self.ai = aitextgen(
                        model_folder=str(self.model_folder.resolve()),
                        to_gpu=self.use_gpu,
                        verbose=self.verbose,
                    )
                    logger.info("AI model initialized")
                except Exception as e:
                    logger.error(f"Error initializing AI model {self.model_name}")
                    logger.error(f"{format_exc(e)}")
                    self.ai = None
            else:
                logger.info("AI model initialized")

    async def set_model(self, model_name: str, reinit: bool = False) -> None:
        """
        Reinitializes the AI with the selected model.
        """
        model_folder = MODELS_ROOT.joinpath(model_name)
        if not model_folder.is_dir():
            raise ValueError(f"Model folder {model_folder} does not exist")
        if not model_folder.joinpath("pytorch_model.bin").is_file():
            raise ValueError(f"Model folder {model_folder} does not contain a pytorch_model.bin file")

        if model_name != self.model_name:
            logger.info(f"Setting model to {model_name}")
            self.model_name = model_name
            self.model_folder = model_folder
        else:
            logger.info(f"Model {model_name} already set")
        if reinit:
            return await self.ai_init(reinit=True)
        return

    def clean_input(self, message: str) -> str:
        """Process the input message"""
        message = message.replace(f"<@{self.bot.user.id}> ", "").replace(f"<@!{self.bot.user.id}> ", "")
        # re.sub(r"<@\d+>", " ", message)
        return message.strip()

    async def generate_response(self, prompt: str) -> str:
        if self.ai is None:
            raise ValueError("AI is not initialized")
        prompt_oneline = " | ".join(prompt.splitlines())
        logger.info(f"Generating response for '{prompt_oneline}'")

        # append a newline at the end if it's missing
        prompt = prompt + "\n" if not prompt.endswith("\n") else prompt

        num_tokens = len(self.ai.tokenizer(prompt)["input_ids"])
        logger.debug(f"Number of tokens in prompt: {num_tokens}")
        if num_tokens > 1000:
            logger.warn("Prompt is too long, dropping lines...")
            while num_tokens >= 1000:
                prompt = " ".join(prompt.split(" ")[20:])  # pretty arbitrary
                num_tokens = len(self.ai.tokenizer(prompt)["input_ids"])

        # do the generation
        async with self.lock:
            response: str = self.ai.generate_one(
                prompt=prompt,
                base_length=num_tokens
                + self.base_length
                + (5 * self.max_lines)
                + (5 * self.context_messages),
                temperature=self.temperature,
            )
            self.instance_runs += 1

        raw_oneline = " | ".join(response.splitlines()).replace(prompt_oneline + " | ", "")
        logger.info(f"Raw response: '{raw_oneline}'")

        promptlines = [x.strip() for x in prompt.splitlines() if len(x.strip()) > 0]
        num_lines = len(promptlines)
        logger.debug(f"Removing {num_lines} prompt lines from raw response")

        resplines = [x.strip() for x in iter(response.splitlines()) if len(x.strip()) > 0]
        resplines = resplines[num_lines:]

        response = "\n".join(resplines[: random.randint(1, self.max_lines)])
        if response == "":
            response = EMOJI["thonk"] + " i don't know what to say " + EMOJI["snake"]
        trimmed_oneline = "\\n".join(response.splitlines())
        logger.info(f"Trimmed response: '{trimmed_oneline}'")
        return response

    @tasks.loop(seconds=30, reconnect=True)
    async def background_generate(self) -> None:
        """
        Feeds the AI with context from random messages and generates a response.
        Used to keep it fresh and spicy.
        """
        if self.ai is None:
            raise ValueError("AI is not initialized")

        if self.lock.locked():
            logger.debug("Skipping background generation, AI is locked")
            return

        # get a message from the context queue
        if self.prompt_queue.empty():
            logger.debug("No queued message to generate from, waiting...")
            return
        else:
            prompt = self.prompt_queue.get_nowait()

        logger.info("Generating background response from queue")
        return await self.generate_response(prompt=prompt)


def setup(bot):
    bot.add_cog(AiCog(bot))