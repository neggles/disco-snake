import logging
from asyncio import sleep

import disnake
from disnake import (
    ApplicationCommandInteraction,
    Colour,
    DMChannel,
    Embed,
    GroupChannel,
    InteractionContextTypes,
    StageChannel,
    TextChannel,
)
from disnake.ext import commands

from disco_snake import checks
from disco_snake.blacklist import Blacklist

COG_NAME = "admin"
logger = logging.getLogger(__name__)


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

    @commands.slash_command(
        name="clear",
        guild_ids=[],
        contexts=InteractionContextTypes(guild=True, bot_dm=True, private_channel=True),
    )
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
        channel = await self.bot.fetch_channel(ctx.channel.id)
        if not isinstance(channel, (DMChannel, TextChannel, GroupChannel, StageChannel)):
            # don't even bother
            await ctx.send("I can't delete messages in this channel type, sorry.", ephemeral=True)
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
            except Exception:
                continue
            finally:
                if (delet_self + delet_other) >= count:
                    # this silences exceptions, but that's on purpose
                    break  # noqa: B012
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

    @commands.slash_command(
        name="blacklist",
        description="Manage blacklisted users",
        auto_sync=False,
    )
    @checks.is_admin()
    async def blacklist_group(self, ctx: ApplicationCommandInteraction) -> None:
        return await ctx.send("You need to specify a subcommand!", ephemeral=True)

    @blacklist_group.sub_command(name="add", description="Add a user to the blacklist")
    @checks.is_admin()
    async def blacklist_add(
        self,
        ctx: ApplicationCommandInteraction,
        user: disnake.User = commands.Param(
            description="User to blacklist",
        ),
        reason: str = commands.Param(
            description="The reason for blacklisting the user.",
            min_length=4,
        ),
    ) -> None:
        """
        Blocks a user from being able to use or interact with the bot in any way.
        :param ctx: The application command interaction.
        :param user: The user that should be added to the blacklist.
        :param reason: The reason for blacklisting the user (required).
        """
        await ctx.response.defer(ephemeral=True)
        try:
            user_id = user.id
            if user_id in self.blacklist:
                entry = self.blacklist[user_id]
                embed = disnake.Embed(
                    title="Error!",
                    description=f"**{user.name}** is already in the blacklist.",
                    color=0xE02B2B,
                )
                embed.add_field(name="Timestamp", value=entry.timestamp.strftime("%d/%m/%Y %H:%M:%S"))
                embed.add_field(name="Reason", value=entry.reason)
                return await ctx.send(embed=embed, ephemeral=True)

            self.blacklist.add_id(user_id, reason)
            embed = disnake.Embed(
                title="User Blacklisted",
                description=f"**{user.name}** has been successfully added to the blacklist",
                color=0x9C84EF,
            )
            embed.add_field(name="Reason", value=reason)

            embed.set_footer(text=f"There are now {len(self.blacklist)} users in the blacklist")
            return await ctx.send(embed=embed, ephemeral=True)
        except Exception as e:
            embed = disnake.Embed(
                title="Error!",
                description=f"An error occurred when trying to add **{user.name}** to the blacklist.",
                color=0xE02B2B,
            )
            embed.add_field(name="Error", value=e)
            await ctx.send(embed=embed, ephemeral=True)
            raise ValueError("Failed to add user to blacklist") from e

    @blacklist_group.sub_command(name="remove", description="Remove a user from the blacklist")
    @checks.is_admin()
    async def blacklist_remove(
        self,
        ctx: ApplicationCommandInteraction,
        user: disnake.User = commands.Param(
            name="user",
            default=None,
            description="Clear messages from a specific user",
        ),
    ):
        """
        Lets you remove a user from not being able to use the bot.
        :param interaction: The application command interaction.
        :param user: The user that should be removed from the blacklist.
        """
        embed = disnake.Embed(
            title="Error!",
            description="Unknown command error. Please try again.",
            color=0xE02B2B,
        )

        try:
            try:
                self.blacklist.remove_id(user.id)
                embed = disnake.Embed(
                    title="User removed from blacklist",
                    description=f"**{user.global_name}** has been successfully removed from the blacklist",
                    color=0x9C84EF,
                )
            except KeyError:
                embed.description = f"**{user.global_name}** is not in the blacklist."

            embed.set_footer(text=f"There are now {len(self.blacklist)} users in the blacklist")
        except Exception:
            embed.description = (
                f"An error occurred when trying to remove **{user.global_name}** from the blacklist."
            )
        finally:
            await ctx.send(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Admin(bot))
