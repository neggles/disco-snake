import json
import logging

import disnake
from disnake import ApplicationCommandInteraction, Option, OptionType
from disnake.ext import commands

from helpers import checks, json_manager

logger = logging.getLogger(__package__)


class Owner(commands.Cog, name="owner"):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot

    @commands.slash_command(
        name="shutdown",
        description="Shut down the bot.",
    )
    @checks.is_owner()
    async def shutdown(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Makes the bot shutdown.
        :param interaction: The application command interaction.
        """
        embed = disnake.Embed(description="Shutting down. Bye! :wave:", color=0x9C84EF)
        await ctx.send(embed=embed)
        await self.bot.close()

    @commands.slash_command(name="blacklist", description="Manage blacklisted users")
    @checks.is_owner()
    async def blacklist(self, ctx: ApplicationCommandInteraction) -> None:
        pass

    @blacklist.sub_command(
        base="blacklist",
        name="add",
        description="Add a user to the blacklist",
        options=[
            Option(
                name="user",
                description="The user you want to blacklist",
                type=OptionType.user,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def blacklist_add(self, ctx: ApplicationCommandInteraction, user: disnake.User = None) -> None:
        """
        Lets you add a user from not being able to use the bot.
        :param interaction: The application command interaction.
        :param user: The user that should be added to the blacklist.
        """
        try:
            user_id = user.id
            with open("blacklist.json") as file:
                blacklist = json.load(file)
            if user_id in blacklist["ids"]:
                embed = disnake.Embed(
                    title="Error!",
                    description=f"**{user.name}** is already in the blacklist.",
                    color=0xE02B2B,
                )
                return await ctx.send(embed=embed)
            json_manager.add_user_to_blacklist(user_id)
            embed = disnake.Embed(
                title="User Blacklisted",
                description=f"**{user.name}** has been successfully added to the blacklist",
                color=0x9C84EF,
            )
            with open("blacklist.json") as file:
                blacklist = json.load(file)
            embed.set_footer(text=f"There are now {len(blacklist['ids'])} users in the blacklist")
            await ctx.send(embed=embed)
        except Exception as exception:
            embed = disnake.Embed(
                title="Error!",
                description=f"An unknown error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            await ctx.send(embed=embed)
            print(exception)

    @blacklist.sub_command(
        base="blacklist",
        name="remove",
        description="Remove a user from the blacklist",
        options=[
            Option(
                name="user",
                description="The user you want to un-blacklist",
                type=OptionType.user,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def blacklist_remove(self, ctx: ApplicationCommandInteraction, user: disnake.User = None):
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
            with open("blacklist.json") as file:
                blacklist = json.load(file)
            embed.set_footer(text=f"There are now {len(blacklist['ids'])} users in the blacklist")
            await ctx.send(embed=embed)
        except ValueError:
            embed = disnake.Embed(
                title="Error!", description=f"**{user.name}** is not in the blacklist.", color=0xE02B2B
            )
            await ctx.send(embed=embed)
        except Exception as exception:
            embed = disnake.Embed(
                title="Error!",
                description=f"An unknown error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            await ctx.send(embed=embed)
            print(exception)


def setup(bot):
    bot.add_cog(Owner(bot))
