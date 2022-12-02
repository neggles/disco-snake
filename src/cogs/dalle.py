import logging
from pathlib import Path

import openai
from disnake import ApplicationCommandInteraction, Embed, Message, User
from disnake.ext import commands
from openai.error import OpenAIError

import logsnake
from disco_snake import DATADIR_PATH, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

# setup package logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=__package__,
    formatter=logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{Path(__file__).stem}.log"),
    fileLoglevel=logging.INFO,
    maxBytes=2 * (2**20),
    backupCount=5,
)

OPENAI_FAVICON_URL = "https://openaiapi-site.azureedge.net/public-assets/d/15b4ef1489/favicon.png"

DALLE_DATADIR = DATADIR_PATH.joinpath("dalle")
DALLE_DATADIR.mkdir(parents=True, exist_ok=True)


class DalleEmbed(Embed):
    def __init__(self, prompt: str, message: Message, requestor: User, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.colour = requestor.accent_colour if requestor.accent_colour else 0xF7F7F8
        self.title = f"{prompt}:"

        self.set_author(name=requestor.display_name, icon_url=requestor.avatar.url, url=message.jump_url)
        self.set_footer(text="Powered by OpenAI DALL·E", icon_url=OPENAI_FAVICON_URL)


# Here we name the cog and create a new class for the cog.
class DallE(commands.Cog, name="dall-e"):
    def __init__(self, bot: DiscoSnake):
        self.bot = bot

        self.config: dict = bot.config["dall-e"] or None
        if self.config is None:
            raise ValueError("Dall-E config not found, disabling cog.")

        self.oai_config: dict = bot.config["openai"] or None
        if self.oai_config is None:
            raise ValueError("OpenAI config not found, disabling cog.")

        # configure openai module
        openai.api_key = self.config["api_key"]
        openai.organization = self.oai_config["org_id"] if self.oai_config["org_id"] else None

        # test the API key
        try:
            engines = openai.Engine.list()
            logger.info(f"OpenAI API key is valid, found {len(engines)} engines.")
        except OpenAIError as e:
            logger.error("Failed to validate OpenAI API key, disabling cog.")
            raise ValueError("Failed to validate OpenAI API key") from e

    # Cog slash command group
    @commands.slash_command(name="dalle", description="Generate and edit images with DALL·E")
    @checks.not_blacklisted()
    @commands.check_any(commands.is_nsfw(), commands.dm_only())
    async def dalle(self, ctx: ApplicationCommandInteraction):
        logger.info("running dall-e subcommand")
        pass

    @dalle.sub_command(name="gen", description="This is a testing command that does nothing.")
    @checks.not_blacklisted()
    async def generate(self, ctx: ApplicationCommandInteraction):
        """
        This is a testing command that does nothing.
        :param interaction: The application command interaction.
        """
        return await ctx.send("pranked", ephemeral=True)


def setup(bot):
    bot.add_cog(DallE(bot))
