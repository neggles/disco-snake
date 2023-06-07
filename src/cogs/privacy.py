import logging
from typing import Union

from disnake import (
    ApplicationCommandInteraction,
    ButtonStyle,
    Colour,
    Embed,
    Guild,
    Invite,
    Member,
    MessageCommandInteraction,
    User,
    ui,
)
from disnake.ext import commands
from disnake.ui import Button, View

from db import DiscordUser, Session
from disco_snake import checks
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__package__)

COG_UID = "privacy"

POLICY_TEXT = """This bot is an AI chatbot made by {org_name}.

When you interact with {org_name} chatbots, we collect data about your interactions to enhance your experience. This includes your Discord user account ID, username, and display names. We keep a record of your messages, timestamps, conversation contexts, hashes of received images and their auto-generated captions, and generated images. This data is used for troubleshooting, bot memory, feature and functionality improvements, abuse detection, analytics, and model training.

Typically, we retain this data for 30 days, but it may be stored indefinitely. Rest assured, no data is shared with any third party unless required by law. If you wish to have all data associated with your user ID removed from our systems, you can contact {org_name} through the support server linked in each bot's profile and we will remove your data within 30 days.

If you choose to reject these terms, we will only store your user ID to record your rejection and prevent any additional data collection. You can change your decision at any time by invoking this slash command again. You do not need to accept or reject these terms per bot; your decision applies to all {org_name} chatbots.

To agree to these terms and enable full bot functionality, please click the **Accept** button below.
If you do not agree, you can click **Reject**. Please note that this will disable all bot functionality except for this slash command.
"""


def get_policy_text(bot: DiscoSnake) -> str:
    return POLICY_TEXT.format(org_name=bot.support_guild.name)


class PrivacyEmbed(Embed):
    def __init__(self, author: Union[User, Member], support_guild: Guild, user: DiscordUser, invite: Invite):
        super().__init__(
            title="Privacy Policy",
            colour=Colour.purple(),
            description="Settings for data collection and privacy.",
        )
        self.set_author(name=author.name, icon_url=author.display_avatar.url)
        self.add_field(name="User Name", value=author.display_name)
        self.add_field(name="User ID", value=author.id)
        self.add_field(
            name="Policy State",
            value="Accepted" if user.tos_accepted else "Rejected" if user.tos_rejected else "Unset",
        )
        if user.tos_accepted or user.tos_rejected:
            self.add_field(name="Timestamp", value=user.tos_timestamp)
        self.add_field(name="Support Server", value=f"[{support_guild}]({invite})", inline=False)
        self.set_footer(text="Privacy Policy v1.0")


class PrivacyView(View):
    def __init__(self, user: DiscordUser):
        self.user: DiscordUser = user
        super().__init__(timeout=180)

    @ui.button(label="Accept", style=ButtonStyle.green, custom_id=f"{COG_UID}:PrivacyView:accept")
    async def accept(self, button: Button, ctx: ApplicationCommandInteraction):
        await ctx.response.defer()
        try:
            async with Session() as session:
                async with session.begin():
                    user: DiscordUser = await session.get(DiscordUser, self.user.id)
                    user.tos_accepted = True
                    user.tos_rejected = False
                    user.tos_timestamp = ctx.created_at
                async with session.begin():
                    user: DiscordUser = await session.get(DiscordUser, self.user.id)
                    if user.tos_accepted is not True:
                        raise RuntimeError("Failed to accept privacy policy")
                    logger.debug(f"User ID {user.id} accepted the privacy policy")
                    await ctx.send(
                        (
                            "Thank you for accepting the privacy policy. You can now use the bot normally."
                            + "\nThis message will be deleted in 30 seconds."
                        ),
                        delete_after=30,
                    )
        except Exception as e:
            await ctx.send(
                f"An error occurred while accepting the privacy policy: {e}",
                delete_after=60,
            )
        finally:
            await ctx.edit_original_response(view=self)
            self.stop()

    @ui.button(label="Reject", style=ButtonStyle.red, custom_id=f"{COG_UID}:PrivacyView:reject")
    async def reject(self, button: Button, ctx: ApplicationCommandInteraction):
        await ctx.response.defer()
        try:
            async with Session() as session:
                async with session.begin():
                    user: DiscordUser = await session.get(DiscordUser, self.user.id)
                    user.tos_accepted = False
                    user.tos_rejected = True
                    user.tos_timestamp = ctx.created_at
                    await session.commit()
                async with session.begin():
                    user: DiscordUser = await session.get(DiscordUser, self.user.id)
                    if user.tos_rejected is not True:
                        raise RuntimeError("Failed to reject privacy policy")
                    logger.debug(f"User ID {user.id} rejected the privacy policy")
                    await ctx.send(
                        "You have rejected the privacy policy and will not be able to use the bot."
                        + "\nThis message will be deleted in 30 seconds.",
                        delete_after=30,
                    )
        except Exception as e:
            await ctx.send(
                f"An error occurred while rejecting the privacy policy: {e}",
                delete_after=60,
            )
        finally:
            await ctx.edit_original_response(view=self)
            self.stop()


class Privacy(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot

    @commands.slash_command(
        name="privacy",
        description="View the privacy policy",
        dm_permission=True,
    )
    @checks.not_blacklisted()
    async def privacy(self, ctx: ApplicationCommandInteraction):
        await ctx.response.defer(ephemeral=True)
        logger.debug(f"Received privacy command for user {ctx.author} ({ctx.author.id})")
        try:
            async with Session.begin() as session:
                user: DiscordUser = await session.get(DiscordUser, ctx.author.id)
                if user is None:
                    user = DiscordUser.from_discord(ctx.author)
                    await session.add(user)
                    await session.commit()
                    user = await session.get(DiscordUser, ctx.author.id)

            invite = await self.bot.support_invite()
            embed = PrivacyEmbed(
                author=ctx.author,
                support_guild=self.bot.support_guild,
                user=user,
                invite=invite,
            )
            view = PrivacyView(user=user)
            content = get_policy_text(self.bot)
            # send the form in a DM
            await ctx.author.send(content=content, embed=embed, view=view)
            # and send an ephemeral message in the channel
            await ctx.send(
                embed=Embed(
                    title="Privacy Policy",
                    description="Please check your DMs for the privacy policy.",
                    colour=Colour.purple(),
                ),
                ephemeral=True,
            )
        except Exception as e:
            logger.exception(e)
            raise e


def setup(bot):
    bot.add_cog(Privacy(bot=bot))
