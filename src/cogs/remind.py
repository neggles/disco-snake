import datetime as dt
import logging
from typing import List, Union

import dateparser as dp
from disnake import Member, ApplicationCommandInteraction, Interaction, Role, Embed, Option, OptionType, User
from disnake.ext import commands, tasks

import logsnake
from disco_snake import LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

COG_UID = "remind"


# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=1,
)


class Reminder(object):
    def __init__(
        self,
        inter: Interaction,
        targets: List[Union[Member, Role]],
        message: str,
        time: dt.datetime,
    ):
        self.inter = inter
        self.targets = targets
        self.time = time
        self.message = message
        self.embed: Embed = Embed(
            title="Reminder created!",
            color=0x00FF00,
            description=f"Scheduled for {self.time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        self.embed.add_field(name="Targets", value=", ".join(f"<@{x.id}>" for x in self.targets))
        self.embed.add_field(name="Message", value=self.message)

    async def send_msg(self):
        refids = ", ".join(f"<@{x.id}>" for x in self.targets)
        if self.message == "":
            await self.inter.send(f"{refids} Here's your reminder")
        else:
            await self.inter.send(f"{refids} {self.message}")

    def is_ready(self):
        return self.time < dt.datetime.now()


class Remind(commands.Cog, name="remind"):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.reminders = []

    async def cog_load(self) -> None:
        logger.info("Remind cog loaded!")
        if not self.send_reminders.is_running():
            logger.info("Starting reminder loop...")
            self.send_reminders.start()
        pass

    def cog_unload(self) -> None:
        logger.info("Unloading remind cog...")

    @tasks.loop(seconds=21.0)
    async def send_reminders(self):
        for reminder in self.reminders:
            if reminder.is_ready():
                self.reminders.remove(reminder)
                await reminder.send_msg()

        if len(self.reminders) == 0:
            self.send_reminders.stop()

    @commands.slash_command(
        name="remind",
        description="Send a message to a list of targets at a set time",
    )
    @checks.not_blacklisted()
    async def add_reminder(
        self,
        inter: ApplicationCommandInteraction,
        time: str = Option(
            name="time",
            description="When to send the reminder",
            type=OptionType.string,
            required=True,
        ),
        target: User = Option(
            name="target",
            description="Who to send the reminder to",
            type=OptionType.mentionable,
            required=True,
        ),
        message: str = Option(
            name="message",
            description="Message to send with the reminder",
            type=OptionType.string,
            required=True,
        ),
    ):
        """
        Reminds the list of targets of a custom message at a set time.
        'time info' follows example format: "Jan 02 01:30 AM"
        """
        logger.info(f"Adding reminder for {target.name} at {time} with message: {message}")
        rtime = dp.parse(time)
        self.reminders.append(Reminder(inter, [target], message, rtime))
        logger.info("Reminder added! sending confirmation message...")

        await inter.send(f"Reminder set for {rtime.strftime('%b %d %I:%M %p')}")
        if not self.send_reminders.is_running():
            self.send_reminders.start()


def setup(bot):
    bot.add_cog(Remind(bot))
