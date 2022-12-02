import platform

import disnake
from disnake import ApplicationCommandInteraction
from disnake.ext import commands

from disco_snake.bot import DiscoSnake
from helpers import checks


class General(commands.Cog, name="general"):
    def __init__(self, bot):
        self.bot: DiscoSnake = bot

    @commands.slash_command(
        name="botinfo",
        description="Get some info about the bot.",
    )
    @checks.not_blacklisted()
    async def botinfo(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Get some useful (or not) information about the bot.
        :param ctx: The application command ctx.
        """
        embed = disnake.Embed(description="a questionably intelligent discord bot", color=0x9C84EF)
        embed.set_author(name="Info", icon_url=self.bot.user.avatar.url)
        embed.add_field(name="Owner:", value=self.bot.owner.mention, inline=True)
        embed.add_field(name="Source repo:", value=self.bot.config["repo_url"], inline=True)
        embed.add_field(name="Running on", value=f"Python {platform.python_version()}", inline=False)
        embed.add_field(name="Timezone:", value=str(self.bot.config["timezone"]), inline=True)
        embed.add_field(name="Uptime:", value=str(self.bot.status), inline=False)
        embed.set_footer(text=f"Requested by {ctx.author}", icon_url=ctx.author.avatar.url)
        await ctx.send(embed=embed)

    @commands.slash_command(
        name="serverinfo",
        description="Get some useful (or not) information about the server.",
    )
    @checks.not_blacklisted()
    async def serverinfo(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Get some useful (or not) information about the server.
        :param ctx: The application command ctx.
        """
        roles = [role.name for role in ctx.guild.roles]
        if len(roles) > 50:
            roles = roles[:50]
            roles.append(f">>>> Displaying[50/{len(roles)}] Roles")
        roles = ", ".join(roles)

        embed = disnake.Embed(title="**Server Name:**", description=f"{ctx.guild}", color=0x9C84EF)
        embed.set_thumbnail(url=ctx.guild.icon.url)
        embed.add_field(name="Server ID", value=ctx.guild.id)
        embed.add_field(name="Member Count", value=ctx.guild.member_count)
        embed.add_field(name="Text/Voice Channels", value=f"{len(ctx.guild.channels)}")
        embed.add_field(name=f"Roles ({len(ctx.guild.roles)})", value=roles)
        embed.set_footer(text=f"Created at: {ctx.guild.created_at}")
        await ctx.send(embed=embed)

    @commands.slash_command(name="ping", description="ping the bot")
    @checks.not_blacklisted()
    async def ping(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Check if the bot is alive.
        :param ctx: The application command ctx.
        """
        embed = disnake.Embed(
            title="🏓 Pong!",
            description=f"Current API latency is {round(self.bot.latency * 1000)}ms",
            color=0x9C84EF,
        )
        await ctx.send(embed=embed)

    @commands.slash_command(
        name="get-invite", description="Get the bot's invite link to add it to your server."
    )
    @checks.not_blacklisted()
    async def invite(self, ctx: ApplicationCommandInteraction) -> None:
        """
        Get the invite link of the bot to be able to invite it.
        :param ctx: The command context.
        """
        embed = disnake.Embed(
            description=f"Invite me by clicking [here](https://discordapp.com/oauth2/authorize?&client_id={self.bot.config['application_id']}&scope=bot+applications.commands&permissions={self.bot.config['permissions']}).",
            color=0xD75BF4,
        )
        try:
            # To know what permissions to give to your bot, please see here: https://discordapi.com/permissions.html and remember to not give Administrator permissions.
            await ctx.author.send(embed=embed)
            await ctx.send("I sent you a private message!")
        except disnake.Forbidden:
            await ctx.send(embed=embed)


def setup(bot):
    bot.add_cog(General(bot))
