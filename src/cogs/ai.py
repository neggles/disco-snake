import asyncio
import logging
import random
import re
from pathlib import Path

import disnake
from disnake import ApplicationCommandInteraction, Message, MessageInteraction
from disnake.ext import commands

from aitextgen import aitextgen
from disco_snake.bot import DiscoSnake
from disco_snake.cli import DATADIR_PATH
from helpers import checks

MODELS_ROOT = DATADIR_PATH.joinpath("models")
EMBED_COLOR = 0xFF9D0B
REACT_EMOJI = "🤗"
NEWLINE = "\n"

logger = logging.getLogger(__package__)


class ModelSelect(disnake.ui.Select):
    def __init__(self):
        model_folders = [x for x in MODELS_ROOT.iterdir() if x.is_dir()]
        options = [disnake.SelectOption(label=x.name, value=str(x)) for x in model_folders]

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
        await inter.response.defer(with_message=True)
        model_name = self.values[0]

        ai_cog: AiCog = inter.bot.get_cog("Ai")
        await ai_cog.set_model(model_name=model_name)

        embed = disnake.Embed(title="Model changed", description=f" {model_name}", color=EMBED_COLOR)
        embed.add_field(name="New Model:", value=model_name, inline=False)
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
        self.bot = bot
        self.lock = asyncio.Lock()

        self.ai: aitextgen = None
        self.model_name: str = None
        self.model_folder: Path = None

        config: dict = bot.config["ai"]
        if config["model_name"] is None:
            raise ValueError("No model name specified in config")

        self.model_name = config["model_name"]
        self.model_folder = MODELS_ROOT.joinpath(self.model_name)
        if not self.model_folder.is_dir():
            raise ValueError(f"Model folder {self.model_folder} does not exist")
        if not self.model_folder.joinpath("pytorch_model.bin").is_file():
            raise ValueError(f"Model folder {self.model_folder} does not contain a pytorch_model.bin file")

        conf_keys = config.keys()
        self.use_gpu: bool = config["use_gpu"] if "use_gpu" in conf_keys else False
        self.verbose: bool = config["verbose"] if "verbose" in conf_keys else False
        self.max_lines: int = config["max_lines"] if "max_lines" in conf_keys else 5
        self.max_length: int = config["max_length"] if "max_length" in conf_keys else 100
        self.temperature: float = config["temperature"] if "temperature" in conf_keys else 0.9
        self.response_chance: float = config["response_chance"] if "response_chance" in conf_keys else 0.5

    async def cog_load(self) -> None:
        await self.ai_init(reinit=False)
        return await super().cog_load()

    async def cog_unload(self) -> None:
        return await super().cog_unload()

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
                    logger.error(e)
                    self.ai = None
            else:
                logger.info("AI model initialized")

    def clean_input(self, message: str) -> str:
        """Process the input message"""
        message = message.replace(f"<@{self.bot.user.id}> ", "").replace(f"<@!{self.bot.user.id}> ", "")
        # re.sub(r"<@\d+>", " ", message)
        return message.strip()

    async def respond(self, prompt: str) -> str:
        if self.ai is None:
            raise ValueError("AI is not initialized")
        logger.info(f"Generating response for '{prompt.replace(NEWLINE, ' ')}'")

        num_tokens = len(self.ai.tokenizer(prompt)["input_ids"])
        logger.debug(f"Number of tokens in prompt: {num_tokens}")
        if num_tokens > 1000:
            logger.info("Prompt is too long, dropping lines...")
            while num_tokens >= 1000:
                message = " ".join(prompt.split(" ")[20:])  # pretty arbitrary
                num_tokens = len(self.ai.tokenizer(prompt)["input_ids"])

        # append a newline at the end if it's missing
        prompt = prompt + "\n" if not prompt.endswith("\n") else prompt

        # do the generation
        async with self.lock:
            response: str = self.ai.generate_one(
                prompt=prompt,
                max_length=num_tokens + self.max_length + (5 * self.max_lines),
                temperature=self.temperature,
            )
            for line in prompt.splitlines():
                response = response.replace(line.strip(), "", 1).strip()
            logger.debug(f"Raw response: '{response.replace(NEWLINE, ' ')}'")

        lines = [x.strip() for x in response.splitlines() if x != ""]
        response = "\n".join(lines[: random.randint(1, self.max_lines)])
        logger.info(f"Response: '{response.replace(NEWLINE, ' ')}'")
        return response

    # Slash Commands

    @commands.slash_command(name="ai", description="Manage the AI")
    async def ai_group(self, inter: ApplicationCommandInteraction):
        pass

    @ai_group.sub_command(
        name="show-config",
        description="Returns the current AI model name and config.",
    )
    @checks.not_blacklisted()
    async def ai_show_config(self, inter: ApplicationCommandInteraction):
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
        embed.add_field(name="Verbose:", value="Yes" if self.verbose else "No", inline=True)
        embed.add_field(name="Max lines:", value=str(self.max_lines), inline=True)
        embed.add_field(name="Max length:", value=str(self.max_length), inline=True)
        embed.add_field(name="Temperature:", value=str(self.temperature), inline=True)
        embed.add_field(name="Response chance:", value=str(self.response_chance), inline=True)

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

    async def set_model(self, model_name: str):
        """
        Reinitializes the AI with the selected model.
        """
        self.model_name = model_name
        model_folder = MODELS_ROOT.joinpath(model_name)
        if not model_folder.is_dir():
            raise ValueError(f"Model folder {model_folder} does not exist")
        if not model_folder.joinpath("pytorch_model.bin").is_file():
            raise ValueError(f"Model folder {model_folder} does not contain a pytorch_model.bin file")
        self.model_folder = model_folder
        return await self.ai_init(reinit=True)

    # parameter tweak command group

    @ai_group.sub_command_group(name="params", description="Change the AI parameters.")
    async def ai_params(self, inter: ApplicationCommandInteraction):
        pass

    @ai_params.sub_command(name="temperature", description="Change the AI temperature parameter.")
    @checks.is_owner()
    @checks.not_blacklisted()
    async def ai_params_temperature(self, inter, temperature: float) -> None:
        """
        Change the AI parameters.
        :param inter: The application command interaction.
        """
        self.temperature = temperature
        await inter.send(f"Temperature set to {temperature}")

    @ai_params.sub_command(name="max-length", description="Change max token length of generated responses.")
    @checks.is_owner()
    @checks.not_blacklisted()
    async def ai_params_max_length(self, inter, length: int) -> None:
        """
        Change the AI parameters.
        :param inter: The application command interaction.
        """
        self.max_length = length
        await inter.send(f"Max length set to {length}")

    @ai_params.sub_command(name="max-lines", description="Change max number of lines in generated responses.")
    @checks.is_owner()
    @checks.not_blacklisted()
    async def ai_params_max_lines(self, inter, lines: int) -> None:
        """
        Change the AI parameters.
        :param inter: The application command interaction.
        """
        self.max_lines = lines
        await inter.send(f"Max lines set to {lines}")

    @ai_params.sub_command(
        name="response-chance", description="Change chance of responding to a non-reply message."
    )
    @checks.is_owner()
    @checks.not_blacklisted()
    async def ai_params_response_chance(self, inter, chance: float) -> None:
        """
        Change the AI parameters.
        :param inter: The application command interaction.
        """
        self.response_chance = chance
        await inter.send(f"Response chance set to {chance}")

    # @commands.slash_command(
    #     name="generate",
    #     description="Generates a response to the given message.",
    #     kwargs={"message": "The message to generate a response for."},
    # )
    # @checks.not_blacklisted()
    # async def generate(self, message: str = None) -> str:
    #     if self.ai is None:
    #         raise ValueError("AI is not initialized")
    #     elif message is not None:
    #         return self.ai.generate_one(
    #             prompt=message + "\n",
    #             temperature=self.temperature,
    #             max_length=self.max_length,
    #         )
    #     else:
    #         return self.ai.generate_one(
    #             temperature=self.temperature,
    #             max_length=self.max_length,
    #         )

    # Event Listeners

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if self.ai is None:
            logger.debug("AI is not initialized but received a message")
            return

        mentioned = (
            True
            if self.bot.user in message.mentions or self.bot.user.display_name in message.content
            else False
        )

        if random.random() > float(self.response_chance) and not mentioned:
            return

        async with message.channel.typing():
            await message.add_reaction(REACT_EMOJI)
            try:
                history = await message.channel.history(limit=9).flatten()
                history.reverse()
                context = [
                    self.clean_input(m.content.strip())
                    for m in history
                    if m.author != self.bot.user and m.id != message.id and len(m.content.strip()) > 2
                ]
                context = [x.strip() for x in context if x != ""]

                # make our prompt
                prompt = "\n".join(context) + "\n" + self.clean_input(message.content)
                # send it to the AI and get the response
                response = await self.respond(prompt)
                # send the response
                await message.reply(response)

            except Exception as e:
                logger.error("Error processing message content")
                logger.error(e)
            finally:
                await message.clear_reaction(REACT_EMOJI)
        return


def setup(bot):
    bot.add_cog(AiCog(bot))
