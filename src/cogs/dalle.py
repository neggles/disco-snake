import logging
from pathlib import Path

import openai
from disnake import GuildCommandInteraction, Embed, User
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
    name=Path(__file__).stem,
    formatter=logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{Path(__file__).stem}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=5,
)

OPENAI_FAVICON_URL = "https://openaiapi-site.azureedge.net/public-assets/d/15b4ef1489/favicon.png"

DALLE_DATADIR = DATADIR_PATH.joinpath("dalle")
DALLE_DATADIR.mkdir(parents=True, exist_ok=True)


class DalleEmbed(Embed):
    def __init__(self, prompt: str, image_url: str, requestor: User, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.colour = requestor.accent_colour if requestor.accent_colour else 0xF7F7F8
        self.title = f"{prompt}:"

        self.set_image(url=image_url)
        self.set_author(name=requestor.display_name, icon_url=requestor.avatar.url)
        self.set_footer(text="Powered by OpenAI DALL·E", icon_url=OPENAI_FAVICON_URL)


# Here we name the cog and create a new class for the cog.
class DallE(commands.Cog, name="dall-e"):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot

        self.config: dict = bot.config.get("dalle", {})
        if self.config is None:
            raise ValueError("Dall·E config not found, disabling cog.")

        self.oai_config: dict = bot.config["openai"] or None
        if self.oai_config is None:
            raise ValueError("OpenAI config not found, disabling cog.")

        # configure openai module
        openai.api_key = self.oai_config["api_key"] or None
        openai.organization = self.oai_config["org_id"] if self.oai_config["org_id"] else None

        if openai.api_key is None:
            raise ValueError("OpenAI API key not found, disabling cog.")

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
    @commands.cooldown(1, 60.0, commands.BucketType.user)
    async def dalle(self, ctx: GuildCommandInteraction):
        pass

    @dalle.sub_command(name="gen", description="Generate an image from a prompt")
    @checks.not_blacklisted()
    async def generate(
        self,
        ctx: GuildCommandInteraction,
        prompt: str = commands.Param(description="The prompt to generate an image from."),
    ):
        """
        make an image from a prompt
        """
        if not prompt:
            await ctx.send("i can't generate an image from nothing...", ephemeral=True)
            return
        await ctx.response.defer()

        try:
            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
            image_url = response["data"][0]["url"]
        except OpenAIError as e:
            logger.error(f"Failed to generate image from prompt {prompt}: ERROR {e.http_status} {e.error}")
            await ctx.send(f"i couldn't generate an image from that prompt...\n{e.http_status} {e.error}")
            return

        embed = DalleEmbed(prompt, image_url, ctx.author)
        await ctx.send(embed=embed)
        return


def setup(bot):
    bot.add_cog(DallE(bot))
