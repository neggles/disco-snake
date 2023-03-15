import logging
from datetime import datetime

from disnake import ApplicationCommandInteraction, Embed
from disnake.ext import commands

import logsnake
from disco_snake import LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

COG_UID = "notes"

# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.INFO,
    maxBytes=2 * (2**20),
    backupCount=2,
)


class Notes(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot

    @commands.slash_command(name="note", description="Commands for managing notes.")
    @checks.not_blacklisted()
    async def notes_group(self, ctx: ApplicationCommandInteraction):
        pass

    @notes_group.sub_command(name="add", description="Add a note.")
    async def note_add(
        self,
        ctx: ApplicationCommandInteraction,
        name: str = commands.Param(description="Name of the note", max_length=64),
        note: str = commands.Param(description="Content of the note", max_length=4000),
    ):
        """
        Add a note.
        :param ctx: The application command interaction.
        :param name: Name of the note.
        :param note: Content of the note.
        """
        user_notes = self.bot.get_userdata_key(ctx.author, "notes", {})
        if name in user_notes.keys():
            await ctx.send(f"Note with name `{name}` already exists.", ephemeral=True)
            return

        user_notes[name] = note
        self.bot.set_userdata_key(ctx.author, "notes", user_notes)

        embed = Embed(
            title="Note added",
            description=f"Note `{name}` added to store.",
            timestamp=datetime.now(),
        )
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url)
        await ctx.send(embed=embed, ephemeral=True)

    @notes_group.sub_command(name="remove", description="Remove a note.")
    async def note_remove(
        self,
        ctx: ApplicationCommandInteraction,
        name: str = commands.Param(description="Name of the note", max_length=64),
    ):
        """
        Remove a note.
        :param ctx: The application command interaction.
        :param name: Name of the note.
        """
        user_notes = self.bot.get_userdata_key(ctx.author, "notes", {})
        if name not in user_notes.keys():
            await ctx.send(f"Note with name `{name}` does not exist.", ephemeral=True)
            return

        user_notes.pop(name)
        self.bot.set_userdata_key(ctx.author, "notes", user_notes)

        embed = Embed(
            title="Note removed",
            description=f"Note `{name}` removed from store.",
            timestamp=datetime.now(),
        )
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url)
        await ctx.send(embed=embed, ephemeral=True)

    @notes_group.sub_command(name="list", description="List all notes.")
    async def note_list(
        self,
        ctx: ApplicationCommandInteraction,
    ):
        user_notes = self.bot.get_userdata_key(ctx.author, "notes", {})
        if len(user_notes.keys()) == 0:
            await ctx.send("You have no notes.", ephemeral=True)
            return

        note_names = user_notes.keys()
        embed_description = f"You have {len(note_names)} notes:\n" + "\n".join(note_names)

        embed = Embed(
            title="Your notes",
            color=ctx.author.accent_color,
            description=embed_description,
            timestamp=datetime.now(),
        )
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url)
        embed.set_footer(text="Use /note remove <name> to remove a note.")

        await ctx.send(embed=embed, ephemeral=True)

    @notes_group.sub_command(name="show", description="Show a note.")
    async def note_read(
        self,
        ctx: ApplicationCommandInteraction,
        name: str = commands.Param(description="Name of the note", max_length=64),
    ):
        """
        Show a note.
        :param ctx: The application command interaction.
        :param name: Name of the note.
        """
        user_notes = self.bot.get_userdata_key(ctx.author, "notes", {})
        if name not in user_notes.keys():
            await ctx.send(f"Note with name `{name}` does not exist.", ephemeral=True)
            return

        embed = Embed(
            title=f"Note `{name}`",
            description=user_notes[name],
            timestamp=datetime.now(),
        )
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url)
        await ctx.send(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(Notes(bot))
