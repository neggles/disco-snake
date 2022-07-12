import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from disnake import Message
from disnake.ext import commands

from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__package__)


class Canned(commands.Cog, name="canned-messages"):
    def __init__(self, bot: DiscoSnake) -> None:
        self.bot: DiscoSnake = bot

    @commands.Cog.listener("on_message")
    async def on_message(self, message: Message):
        if message.author == self.bot.user or message.author.bot:
            return

        autoreplies = self.bot.userstate["autoreplies"]
        daily_autoreplies = [x for x in autoreplies if x["type"] == "daily"]

        if message.author.id in [x["user"] for x in daily_autoreplies]:
            autoreply = next(val for val in daily_autoreplies if val["user"] == message.author.id)
            index = daily_autoreplies.index(autoreply)

            logger.debug(f"Got message from autoreply target {message.author.name}")
            last_reply: date = (
                datetime.fromisoformat(autoreply["last_reply"]).replace(tzinfo=self.bot.timezone).date()
            )
            message_created: date = message.created_at.astimezone(self.bot.timezone).date()

            if last_reply < message_created:
                logger.debug(f"{message.author.name} has not had their daily autoreply yet")
                await message.reply(autoreply["message"])
                logger.info(f"Replied to {message.author.name} with their daily autoreply")
                self.bot.userstate["autoreplies"][index]["last_reply"] = message_created.isoformat()
                self.bot.save_userstate()
                return


# And then we finally add the cog to the bot so that it can load, unload, reload and use it's content.
def setup(bot):
    bot.add_cog(Canned(bot))
