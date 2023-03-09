import logging
from dataclasses import dataclass

from disnake import Guild, Role, Colour
from disnake.ext import commands, tasks

import logsnake
from disco_snake import LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake

COG_UID = "freedom"


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


@dataclass
class PatriotColors:
    red = Colour(0xF20C3E)
    white = Colour(0xF8F8F2)
    blue = Colour(0x0059E8)


class Freedom(commands.Cog, name="freedom"):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.guild: Guild = None
        self.role: Role = None

        self._guild_id = bot.config[COG_UID]["guild_id"] or None
        self._role_id = bot.config[COG_UID]["role_id"] or None

        if any([self._guild_id is None, self._role_id is None]):
            raise ValueError("freedom.role_id and freedom.guild_id must be set in config.json!")

        self.colours = [PatriotColors.red, PatriotColors.white, PatriotColors.blue]
        self.index = 0

    @property
    def is_patriotic(self) -> bool:
        return self.role.colour.to_rgb() in [x.to_rgb() for x in self.colours]

    async def cog_load(self) -> None:
        logger.info("FREEDOM IS COMING! WILL YOU HEED THE CALL?")
        pass

    def cog_unload(self):
        if self.military_industrial_complex.is_running():
            logger.info("FREEDOM'S WORK IS NEVER DONE!")
            self.military_industrial_complex.cancel()
        logger.info("FREEDOM IS NEVER TRULY LOST! WE SHALL RETURN!")

    @commands.Cog.listener("on_ready")
    async def on_ready(self):
        logger.info("FREEDOM RINGS ACROSS THE LAND!")
        if self.guild is None:
            self.guild = self.bot.get_guild(self._guild_id)
            logger.info(f"GUILD OF THE FREE: {self.guild}")

        if self.role is None:
            self.role = self.guild.get_role(self._role_id)
            logger.info(f"ROLE OF THE FREE: {self.role}")

        if not self.military_industrial_complex.is_running():
            self.military_industrial_complex.start()
            logger.info("FREEDOM'S WORK HAS ONLY JUST BEGUN!")
        logger.info("FREEDOM IS HERE! FIGHT ON, BROTHERS!")

    @tasks.loop(seconds=61.0)
    async def military_industrial_complex(self):
        logger.debug("THE TIME HAS COME TO FIGHT FOR FREEDOM!")
        if not self.is_patriotic:
            logger.debug(f"FREEDOM IS NOT FREE! {self.role.colour}")
            self.role = await self.role.edit(colour=PatriotColors.red)
            self.index = 0
        else:
            try:
                self.index = (self.index + 1) % len(self.colours)
                self.role = await self.role.edit(colour=self.colours[self.index])
                logger.debug(f"FREEDOM REMAINS FREE! NOW IN {self.role.colour}!")
            except Exception as e:
                logger.error("ERROR IN FREEDOM'S WORK!")
                logger.error(e)
                raise e
        logger.debug(f"ROLE OF THE FREE: {self.role}")


def setup(bot):
    bot.add_cog(Freedom(bot))
