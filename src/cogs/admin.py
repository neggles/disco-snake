import json
import logging
from asyncio import sleep
from typing import Union

import disnake
from disnake import (
    ApplicationCommandInteraction,
    Colour,
    DMChannel,
    Embed,
    GroupChannel,
    Message,
    MessageCommandInteraction,
    ModalInteraction,
    Option,
    OptionType,
    StageChannel,
    TextChannel,
    TextInputStyle,
    Thread,
)
from disnake.ext import commands
from disnake.ui import Modal, TextInput

from disco_snake import checks

logger = logging.getLogger(__package__)

COG_NAME = "admin"


class Admin(commands.Cog, name=COG_NAME):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot

    @commands.slash_command(
        name="shutdown",
        description="Shut down the bot.",
    )
    @checks.is_admin()
    async def shutdown(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Makes the bot shutdown.
        :param interaction: The application command interaction.
        """
        embed = disnake.Embed(description="Shutting down. Bye! :wave:", color=0x9C84EF)
        await ctx.send(embed=embed, ephemeral=True)
        await self.bot.close()

    @commands.slash_command(
        name="blacklist",
        description="Manage blacklisted users",
        auto_sync=False,
    )
    @checks.is_admin()
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
    @checks.is_admin()
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
            # json_manager.add_user_to_blacklist(user_id)
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
    @checks.is_admin()
    async def blacklist_remove(self, ctx: ApplicationCommandInteraction, user: disnake.User = None):
        """
        Lets you remove a user from not being able to use the bot.
        :param interaction: The application command interaction.
        :param user: The user that should be removed from the blacklist.
        """
        try:
            # json_manager.remove_user_from_blacklist(user.id)
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

    @commands.slash_command(name="clear", dm_permission=True, guild_ids=[])
    @checks.is_admin()
    async def clear_messages(
        self,
        ctx: ApplicationCommandInteraction,
        count: float = commands.Param(
            name="count",
            default=10.0,
            ge=0.0,
            le=100.0,
            description="Target number of messages to delete (max 100, may delete less if all=False)",
        ),
        clear_all: bool = commands.Param(
            name="all",
            default=False,
            description="Clear all messages, not just ones I sent",
        ),
    ):
        # send thinking message
        await ctx.response.defer(ephemeral=True)

        # make count an int
        count = int(count)

        # get channel info
        channel: Union[DMChannel, TextChannel, GroupChannel, StageChannel] = await self.bot.fetch_channel(
            ctx.channel.id
        )
        if isinstance(channel, Thread):
            # don't even bother
            await ctx.send("I can't delete messages in threads, sorry.", ephemeral=True)
            return
        elif isinstance(channel, (DMChannel, GroupChannel)):
            clear_all = False

        delet_self = 0
        delet_other = 0
        async for message in channel.history(limit=100):
            try:
                if message.author.id == self.bot.user.id:
                    await message.delete()
                    delet_self += 1
                elif clear_all is True:
                    await message.delete()
                    delet_other += 1
            except Exception as e:
                continue
            finally:
                if (delet_self + delet_other) >= count:
                    break
                await sleep(0.75)

        deleted = delet_self + delet_other
        delet_embed = Embed(title="Deletion complete", colour=Colour.red())
        delet_embed.add_field(name="Requested", value=count, inline=True)
        delet_embed.add_field(name="Deleted", value=deleted, inline=True)
        delet_embed.add_field(name="All Users", value=clear_all, inline=True)
        if clear_all is True:
            delet_embed.add_field(name="From Self", value=delet_self, inline=True)
            delet_embed.add_field(name="From Others", value=delet_other, inline=True)
        delet_embed.set_footer(text=f"Requested by {ctx.author.name}")

        await ctx.send(embed=delet_embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Admin(bot))
