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

    @commands.slash_command(
        name="cog",
        description="Lets you enable, disable, or reload a cog.",
    )
    @checks.is_owner()
    async def extensions(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Lets you enable, disable, or reload a cog.
        :param ctx: The application command interaction.
        """
        pass

    @extensions.sub_command(
        base="cog",
        name="list",
        description="List all available cogs.",
    )
    @checks.is_owner()
    async def list_cog(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Enable a cog.
        :param interaction: The application command interaction.
        :param name: The name of the cog you would like to load.
        """
        await ctx.response.defer()
        descr = "Loaded cogs:\n"
        for cog in self.bot.cogs:
            descr += f"> {cog}\n"

        descr += "\nAvailable cogs:\n"
        for cog in self.bot.available_cogs():
            descr += f"> {cog}\n"
        descr += "\n\nUse `/cog enable <cog>` to enable a cog."

        embed = disnake.Embed(
            title="Cogs",
            description=descr,
            color=0x9C84EF,
        )
        await ctx.send(embed=embed)

    @extensions.sub_command(
        base="cog",
        name="enable",
        description="Enable a cog.",
        options=[
            Option(
                name="name",
                description="The name of the cog you would like to load.",
                type=OptionType.string,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def enable_cog(self, ctx: ApplicationCommandInteraction, name: str) -> None:
        """
        Enable a cog.
        :param interaction: The application command interaction.
        :param name: The name of the cog you would like to load.
        """
        await ctx.response.defer()
        try:
            cog = self.bot.add_cog(f"cogs.{name}", True)
            if cog is not None:
                embed = disnake.Embed(
                    title="Cog Loaded",
                    description=f"Cog **{name}** loaded successfully.",
                    color=0x9C84EF,
                )
                return await ctx.send(embed=embed, delete_after=10)
            else:
                embed = disnake.Embed(
                    title="Error!",
                    description=f"Cog **{name}** failed to load or does not exist.",
                    color=0xE02B2B,
                )
                return await ctx.send(embed=embed, delete_after=10)

        except Exception as e:
            embed = disnake.Embed(
                title="Error!",
                description=f"An error occurred when trying to unload **{name}**",
                color=0xE02B2B,
            )
            await ctx.send(embed=embed, delete_after=10)
            logger.error(e)

    @extensions.sub_command(
        base="cog",
        name="disable",
        description="Disable a cog.",
        options=[
            Option(
                name="name",
                description="The name of the cog you would like to unload.",
                type=OptionType.string,
                required=True,
            )
        ],
    )
    @checks.is_owner()
    async def disable_cog(self, ctx: ApplicationCommandInteraction, name: str) -> None:
        """
        Disable a cog.
        :param interaction: The application command interaction.
        :param name: The name of the cog you would like to unload.
        """
        await ctx.response.defer(ephemeral=True)

        try:
            cog: commands.Cog = self.bot.get_cog(f"cogs.{name}")
            if cog is not None:
                self.bot.remove_cog(cog.qualified_name)
                embed = disnake.Embed(
                    title="Cog Unloaded",
                    description=f"Cog **{name}** unloaded successfully.",
                    color=0x9C84EF,
                )
                return await ctx.send(embed=embed, ephemeral=True, delete_after=15.0)
            else:
                embed = disnake.Embed(
                    title="Error!",
                    description=f"Cog **{name}** is not loaded or does not exist.",
                    color=0xE02B2B,
                )
                return await ctx.send(embed=embed, ephemeral=True, delete_after=15.0)

        except Exception as e:
            embed = disnake.Embed(
                title="Error!",
                description=f"An error occurred when trying to unload **{name}**",
                color=0xE02B2B,
            )
            return await ctx.send(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Owner(bot))