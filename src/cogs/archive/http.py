import logging

from aiohttp import ClientSession

from disnake import ApplicationCommandInteraction
from disnake.ext import commands

from helpers import checks
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__package__)


class Template(commands.Cog, name="template"):
    def __init__(self, bot: DiscoSnake):
        self.bot = bot
        self.session: ClientSession = None

    async def cog_load(self) -> None:
        self.session = ClientSession()

    def cog_unload(self):
        self.bot.loop.run_until_complete(self.session.close())

    @commands.slash_command(
        name="get",
        description="This is a testing command that does a GET and returns the response.",
    )
    @checks.not_blacklisted()
    async def http_get(
        self,
        inter: ApplicationCommandInteraction,
        url: str = commands.Param(description="The URL to GET.", default="https://httpbin.org/get"),
    ):
        """
        This is a testing command that does a GET and returns the response.
        :param ctx: The application command interaction context.
        :param url: The URL to GET.
        """
        async with self.session.get(url) as resp:
            await inter.response.send_message(f"Response: {resp.status}")


def setup(bot):
    bot.add_cog(Template(bot))
