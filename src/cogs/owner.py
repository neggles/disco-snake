import logging

from disnake import (
    Colour,
    Embed,
    Message,
    MessageCommandInteraction,
    ModalInteraction,
    TextInputStyle,
)
from disnake.ext import commands
from disnake.ui import Modal, TextInput

from disco_snake import checks

logger = logging.getLogger(__name__)


class EditMessageModal(Modal):
    def __init__(self, ctx: MessageCommandInteraction):
        self.message: Message = ctx.target
        components = [
            TextInput(
                label="Content",
                custom_id="content",
                style=TextInputStyle.paragraph,
                value=self.message.content,
                required=True,
                max_length=2048,
            )
        ]
        super().__init__(title="Edit Message", components=components, custom_id=f"edit_message_{ctx.id}")

    async def callback(self, ctx: ModalInteraction):
        await ctx.response.defer(ephemeral=True)
        embed = Embed(title="Edit Message", description="Success!", colour=Colour.purple())
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
        embed.add_field(name="Message ID", value=self.message.id, inline=True)
        try:
            content = ctx.text_values.get("content", "").strip()
            if len(content) == 0:
                raise ValueError("No content entered")
            embed.add_field(name="Original Content", value=self.message.content[:1024], inline=False)
            await self.message.edit(content=content)
        except Exception as e:
            embed.color = Colour.red()
            embed.description = "Failed!"
            if isinstance(e, ValueError):
                if e.args[0] == "No content entered":
                    embed.add_field(name="Error", value="You didn't enter any content...", inline=False)
            else:
                logger.exception(e)
                embed.add_field(name="Error", value="An unknown error occurred", inline=False)
                embed.add_field(name="Exception", value=e, inline=False)
        finally:
            await ctx.send(embed=embed, ephemeral=True)


class Owner(commands.Cog, name="owner"):
    def __init__(self, bot: commands.Bot):
        self.bot: commands.Bot = bot

    @commands.message_command(
        name="Edit Message",
        dm_permission=True,
    )
    @commands.default_member_permissions(manage_messages=True)
    @commands.is_owner()
    async def message_edit(self, ctx: MessageCommandInteraction):
        logger.debug(f"Received edit message command for message {ctx.target.id}")
        if ctx.target.author.id != self.bot.user.id:
            await ctx.send("I can only edit my own messages!", ephemeral=True)
            return
        try:
            modal = EditMessageModal(ctx)
            await ctx.response.send_modal(modal)
        except Exception as e:
            logger.exception(e)
            raise e


def setup(bot):
    bot.add_cog(Owner(bot))
