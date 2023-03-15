import json
import logging

import disnake
from disnake import ApplicationCommandInteraction, Option, OptionType
from disnake.ext import commands

from disco_snake import DATADIR_PATH
from helpers import checks, json_manager

logger = logging.getLogger(__package__)


class Owner(commands.Cog, name="owner"):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot
        self.blacklist_file = DATADIR_PATH.joinpath("blacklist.json")

    @commands.slash_command(
        name="shutdown",
        description="Make the bot shutdown.",
    )
    @checks.is_owner()
    async def shutdown(self, inter: ApplicationCommandInteraction) -> None:
        """
        Makes the bot shutdown.
        :param interaction: The application command interaction.
        """
        embed = disnake.Embed(description="Shutting down. Bye! :wave:", color=0x9C84EF)
        await inter.send(embed=embed)
        await self.bot.close()

    @commands.slash_command(
        name="blacklist",
        description="Get the list of all blacklisted users.",
    )
    @checks.is_owner()
    async def blacklist(self, inter: ApplicationCommandInteraction) -> None:
        """
        Lets you add or remove a user from not being able to use the bot.
        :param interaction: The application command interaction.
        """
        pass

    @blacklist.sub_command(
        base="blacklist",
        name="add",
        description="Lets you add a user from not being able to use the bot.",
        options=[
            Option(
                name="user",
                description="The user you want to add to the blacklist.",
                type=OptionType.user,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def blacklist_add(self, inter: ApplicationCommandInteraction, user: disnake.User = None) -> None:
        """
        Lets you add a user from not being able to use the bot.
        :param interaction: The application command interaction.
        :param user: The user that should be added to the blacklist.
        """
        try:
            user_id = user.id
            with self.blacklist_file.open() as file:
                blacklist = json.load(file)
            if user_id in blacklist["ids"]:
                embed = disnake.Embed(
                    title="Error!",
                    description=f"**{user.name}** is already in the blacklist.",
                    color=0xE02B2B,
                )
                return await inter.send(embed=embed)
            json_manager.add_user_to_blacklist(user_id)
            embed = disnake.Embed(
                title="User Blacklisted",
                description=f"**{user.name}** has been successfully added to the blacklist",
                color=0x9C84EF,
            )
            with self.blacklist_file.open() as file:
                blacklist = json.load(file)
            embed.set_footer(text=f"There are now {len(blacklist['ids'])} users in the blacklist")
            await inter.send(embed=embed)
        except Exception as exception:
            embed = disnake.Embed(
                title="Error!",
                description=f"An unknown error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            await inter.send(embed=embed)
            print(exception)

    @blacklist.sub_command(
        base="blacklist",
        name="remove",
        description="Lets you remove a user from not being able to use the bot.",
        options=[
            Option(
                name="user",
                description="The user you want to remove from the blacklist.",
                type=OptionType.user,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def blacklist_remove(self, inter: ApplicationCommandInteraction, user: disnake.User = None):
        """
        Lets you remove a user from not being able to use the bot.
        :param interaction: The application command interaction.
        :param user: The user that should be removed from the blacklist.
        """
        try:
            json_manager.remove_user_from_blacklist(user.id)
            embed = disnake.Embed(
                title="User removed from blacklist",
                description=f"**{user.name}** has been successfully removed from the blacklist",
                color=0x9C84EF,
            )
            with self.blacklist_file.open() as file:
                blacklist = json.load(file)
            embed.set_footer(text=f"There are now {len(blacklist['ids'])} users in the blacklist")
            await inter.send(embed=embed)
        except ValueError:
            embed = disnake.Embed(
                title="Error!", description=f"**{user.name}** is not in the blacklist.", color=0xE02B2B
            )
            await inter.send(embed=embed)
        except Exception as exception:
            embed = disnake.Embed(
                title="Error!",
                description=f"An unknown error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            await inter.send(embed=embed)
            print(exception)

    @commands.slash_command(
        name="send",
    )
    @checks.is_owner()
    async def send(self, inter: ApplicationCommandInteraction, message: str):
        await inter.channel.send(message)
        await inter.send("Message sent!", ephemeral=True)


def setup(bot):
    bot.add_cog(Owner(bot))
