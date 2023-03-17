import logging

from disnake import ApplicationCommandInteraction
from disnake.ext import commands

from helpers import checks
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__package__)


class Template(commands.Cog, name="template"):
    def __init__(self, bot: DiscoSnake):
        self.bot = bot

    @commands.slash_command(
        name="test",
        description="This is a testing command that does nothing.",
    )
    @checks.not_blacklisted()
    @checks.is_owner()
    async def test(self, ctx: ApplicationCommandInteraction):
        """
        This is a testing command that does nothing.
        Note: This is a SLASH command
        :param ctx: The application command interaction.
        """
        pass


def setup(bot):
    bot.add_cog(Template(bot))
