import logging
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from disnake import Embed, GuildCommandInteraction, User, File
from disnake.ext import commands

import logsnake
from disco_snake import DATADIR_PATH, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

# setup package logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=__package__,
    formatter=logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{Path(__file__).stem}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=5,
)

SD_DATADIR = DATADIR_PATH.joinpath("sd", "waifu")
SD_DATADIR.mkdir(parents=True, exist_ok=True)

SD_MODEL = "waifu-diffusion"


class WaifuEmbed(Embed):
    def __init__(self, prompt: str, image_file: File, requestor: User, *args, **kwargs):
        super().__init__(title=f"{prompt}:", *args, **kwargs)

        self.colour = int(requestor.accent_colour) if requestor.accent_colour is not None else 0xFFD01C

        self.set_image(file=image_file)
        self.set_author(name=requestor.display_name, icon_url=requestor.avatar.url)
        self.set_footer(text="Powered by Huggingface Diffusers 🤗🧨")


# Here we name the cog and create a new class for the cog.
class Journey(commands.Cog, name="waifu"):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        logger.info(f"Loaded {self.qualified_name} cog.")

    async def cog_load(self) -> None:
        await self.pipe_init()
        return await super().cog_load()

    async def pipe_init(
        self,
        model_name: str = SD_MODEL,
        torch_dtype: torch.dtype = torch.float32,
    ):
        model_dir = DATADIR_PATH.joinpath("models", model_name)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        logger.info(f"Loading diffusers model from {model_dir}")
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch_dtype, local_files_only=True
        )
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        self.pipe.to("cuda")
        logger.info(f"Loaded diffusers model {model_name} successfully.")
        return

    # Cog slash command group
    @commands.slash_command(
        name="waifu", description=f"Generate images with {SD_MODEL}. WARNING: NO CONTENT FILTER"
    )
    @checks.not_blacklisted()
    @commands.cooldown(1, 60.0, commands.BucketType.user)
    async def generate(
        self,
        ctx: GuildCommandInteraction,
        prompt: str = commands.Param(description="The prompt to generate an image from."),
    ):
        """
        make an image from a prompt
        """
        if not prompt:
            await ctx.send("i can't generate an image from nothing...", ephemeral=True)
            return
        await ctx.response.defer()

        logger.info(f"Generating image for {ctx.user.name} from prompt '{prompt}'")

        try:
            result: StableDiffusionPipelineOutput = self.pipe(prompt.strip(), guidance_scale=6)
        except Exception as e:
            raise e

        save_path = SD_DATADIR.joinpath(f"{ctx.author.id}_{round(datetime.utcnow().timestamp())}.png")
        image = result.images[0]
        if result.nsfw_content_detected[0] is True:
            logger.info(f"NSFW content detected for {ctx.user.name} from prompt '{prompt}'")
            save_path = save_path.with_suffix(".nsfw.png")
            # logger.info(f"Saving NSFW image to {save_path}")
            # image.save(save_path)
            # await ctx.send("that prompt was too spicy for me to handle...")
            # return

        logger.info(f"Saving image to {save_path}")
        image.save(save_path)
        image_file = File(save_path)
        await ctx.send(embed=WaifuEmbed(prompt, image_file, ctx.author))
        return


def setup(bot):
    bot.add_cog(Journey(bot))
