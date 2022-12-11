import logging
import os
from types import MethodType

import disnake
import dotsi
import psutil
from disnake import ApplicationCommandInteraction, Object
from disnake.ext import commands, tasks
from pynvml.smi import nvidia_smi

from helpers import checks

logger = logging.getLogger(__package__)


class Stats(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.index = 0
        self.bot = bot
        self.nvsmi = nvidia_smi.getInstance()
        self.gpustats = dotsi.Dict(self.nvsmi.DeviceQuery()["gpu"][0])

        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self.bot.status_task = MethodType(self.status_task, bot)
        self.update.start()

    def cog_unload(self):
        self.update.cancel()
        if self.status_task.is_running():
            self.status_task.cancel()

    @property
    def cpu_percent(self):
        return self.process.cpu_percent() / psutil.cpu_count()

    @property
    def memory_percent(self):
        return self.process.memory_percent()

    @property
    def memory_mb(self):
        return self.process.memory_info().rss / 1024 / 1024

    @property
    def gpu_percent(self):
        return self.gpustats.utilization.gpu_util

    @tasks.loop(seconds=15.0)
    async def update(self):
        self.gpustats = dotsi.Dict(self.nvsmi.DeviceQuery()["gpu"][0])
        self.gpuproc = [proc for proc in self.gpustats.processes if proc["pid"] == self.pid][0]

    @update.before_loop
    async def update_before(self):
        logger.info("waiting for bot to be ready...")
        await self.bot.wait_until_ready()

    @tasks.loop(seconds=30)
    async def status_task(self) -> None:
        """
        Set up the bot's status task
        """
        activity = disnake.Activity(name="butts", type=disnake.ActivityType.unknown)
        await self.bot.change_presence(activity=activity)


def setup(bot: commands.Bot):
    bot.add_cog(Stats(bot))
