import logging

from disnake import ApplicationCommandInteraction
from disnake.ext import commands

from helpers import checks

logger = logging.getLogger(__package__)


# Here you can just add your own commands, you'll always need to provide "self" as first parameter.
@commands.slash_command(
    name="testcommand",
    description="This is a testing command that does nothing.",
)
# This will only allow non-blacklisted members to execute the command
@checks.not_blacklisted()
# This will only allow owners of the bot to execute the command -> config.json
@checks.is_owner()
async def testcommand(inter: ApplicationCommandInteraction):
    """
    This is a testing command that does nothing.
    Note: This is a SLASH command
    :param inter: The application command interaction.
    """
    # Do your stuff here

    # Don't forget to remove "pass", that's just because there's no content in the method.
    pass


# And then we finally add the cog to the bot so that it can load, unload, reload and use it's content.
def setup(bot):
    bot.add_command(testcommand)
