import json
import logging
from asyncio import sleep
from collections import UserList
from typing import Any, Union

import disnake
from disnake import (
    ApplicationCommandInteraction,
    Colour,
    DMChannel,
    Embed,
    GroupChannel,
    Option,
    OptionType,
    StageChannel,
    TextChannel,
    Thread,
)
from disnake.ext import commands

from disco_snake import DATADIR_PATH, checks

COG_NAME = "admin"
logger = logging.getLogger(__name__)


class Blacklist(UserList):
    def __init__(self) -> None:
        self._path = DATADIR_PATH.joinpath("blacklist.json")
        self._blacklist: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))

    @property
    def data(self) -> list[int]:
        return json.loads(self._path.read_text(encoding="utf-8"))["ids"]

    @data.setter
    def data(self, value: list[int]) -> None:
        self._path.write_text(json.dumps({"ids": value}, indent=4), encoding="utf-8")

    @data.deleter
    def data(self) -> None:
        self._path.write_text(json.dumps({"ids": []}, indent=4), encoding="utf-8")

    def add(self, id: int) -> list[int]:
        self.data.append(id)
        return self.data

    def remove(self, id: int) -> list[int]:
        self.data = [i for i in self.data if i != id]
        return self.data


class Admin(commands.Cog, name=COG_NAME):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot
        self.blacklist = Blacklist()

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

    @commands.slash_command(name="clear", dm_permission=True, guild_ids=[])
    @checks.is_admin()
    async def clear_messages(
        self,
        ctx: ApplicationCommandInteraction,
        count: float = commands.Param(
            name="count",
            default=1.0,
            ge=0.0,
            le=100.0,
            description="Target number of messages to delete (max 100, may delete less if all=False)",
        ),
        clear_all: bool = commands.Param(
            name="all",
            default=False,
            description="Clear all messages, not just ones I sent",
        ),
        user: disnake.User = commands.Param(
            name="user",
            default=None,
            description="Clear messages from a specific user",
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
        delet_user = 0
        async for message in channel.history(limit=100):
            try:
                if message.author.id == self.bot.user.id:
                    await message.delete()
                    delet_self += 1
                    continue
                elif user is not None and message.author.id == user.id:
                    await message.delete()
                    delet_other += 1
                    delet_user += 1
                elif clear_all is True:
                    await message.delete()
                    delet_other += 1
                    continue
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
        delet_embed.add_field(name="User", value=(user if user is not None else False), inline=True)
        if clear_all is True:
            delet_embed.add_field(name="From Self", value=delet_self, inline=True)
            delet_embed.add_field(name="From Others", value=delet_other, inline=True)
        if user is not None:
            delet_embed.add_field(name="From User", value=user.mention, inline=True)
        delet_embed.set_footer(text=f"Requested by {ctx.author.name}")

        await ctx.send(embed=delet_embed, ephemeral=True)


if False:

    @commands.slash_command(
        name="blacklist",
        description="Manage blacklisted users",
        auto_sync=False,
    )
    @checks.is_admin()
    async def blacklist_group(self, ctx: ApplicationCommandInteraction) -> None:
        return await ctx.send("You need to specify a subcommand!", ephemeral=True)

    @blacklist_group.sub_command(
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
            if user_id in self.blacklist:
                embed = disnake.Embed(
                    title="Error!",
                    description=f"**{user.name}** is already in the blacklist.",
                    color=0xE02B2B,
                )
                return await ctx.send(embed=embed)

            self.blacklist.append(user_id)
            embed = disnake.Embed(
                title="User Blacklisted",
                description=f"**{user.name}** has been successfully added to the blacklist",
                color=0x9C84EF,
            )

            embed.set_footer(text=f"There are now {len(self.blacklist)} users in the blacklist")
            await ctx.send(embed=embed)
        except Exception as exception:
            embed = disnake.Embed(
                title="Error!",
                description=f"An unknown error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            await ctx.send(embed=embed)
            print(exception)

    @blacklist_group.sub_command(
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
            self.blacklist.remove(user.id)
            embed.set_footer(text=f"There are now {len(self.list['ids'])} users in the blacklist")
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
    bot.add_cog(Admin(bot))
