import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial as partial_func
from io import BytesIO, FileIO

import requests
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging as d2logging
from disnake import (
    ApplicationCommandInteraction,
    ButtonStyle,
    Colour,
    Embed,
    File,
    Member,
    MessageInteraction,
    User,
    ui,
)
from disnake.ext import commands

import logsnake
from cogs.common import Upscaler
from disco_snake import DATADIR_PATH, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

COG_UID = "waifu"

# set diffusers logger to info
d2logging.set_verbosity_info()

# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=2,
)

SD_DATADIR = DATADIR_PATH.joinpath("sd", COG_UID)
SD_DATADIR.mkdir(parents=True, exist_ok=True)
SD_MODEL = "waifu-diffusion"


class SDEmbed(Embed):
    def __init__(self, prompt: str, image_file: File | str, requestor: User | Member, *args, **kwargs):
        super().__init__(title=f"{prompt}:", *args, **kwargs)
        self.colour = requestor.colour if isinstance(requestor, Member) else Colour(0xFFD01C)

        if isinstance(image_file, File):
            self.set_image(file=image_file)
            self._image_filename = image_file.filename
        else:
            self.set_image(url=image_file)
            self._image_filename = image_file.split("/")[-1]
        self.set_author(name=requestor.display_name, icon_url=requestor.avatar.url)
        self.set_footer(text="Powered by Huggingface Diffusers 🤗🧨")


class ImageView(ui.View):
    def __init__(self, bot: DiscoSnake, upscaler: Upscaler = None):
        super().__init__(timeout=None)
        self.bot = bot
        self.upscaler = upscaler
        if upscaler is None:
            self.upscale_button.disabled = True
            self.upscale_button.label = "❌ No Upscaler"

    @ui.button(label="Upscale", style=ButtonStyle.primary, custom_id=f"{COG_UID}_ImageView:upscale")
    async def upscale_button(self, button: ui.Button, ctx: MessageInteraction):
        # disable the upscale button
        button.disabled = True
        button.label = "Upscaling..."
        await ctx.response.edit_message(view=self)

        embed: SDEmbed = ctx.message.embeds[0]
        src_url = embed.image.url
        src_filename = embed.image.url.split("/")[-1]

        try:
            upscaled_url = await self.bot.do(self.upscaler.upscale, src_url)
            button.label = "Upscaled!"

            res = requests.get(upscaled_url)
            res.raise_for_status()
            image_file = File(BytesIO(res.content), filename=src_filename)
            embed.set_image(file=image_file)

            prompt = embed.title[:-1] if embed.title.endswith(":") else embed.title
            embed.title = f"{prompt} (Upscaled):"
            embed.set_footer(text="Powered by Huggingface Diffusers 🤗🧨 and Replicate.com 🧬")
        except Exception as e:
            await ctx.followup.send(e)
            logger.error(e)
        finally:
            await ctx.edit_original_response(embed=embed, view=None)
            self.stop()
            return


# Here we name the cog and create a new class for the cog.
class Waifu(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.pipe: StableDiffusionPipeline = None  # type: ignore
        self.loading = True  # set to false once the pipe is loaded

        self.executor = ThreadPoolExecutor(
            max_workers=5, thread_name_prefix=f"{COG_UID}"
        )  # thread pool for blocking code
        self.gpu_executor = bot.gpu_executor  # thread "pool" for GPU operations

        # Retrieve the replicate token from the config for upscaling
        self.replicate_token = self.bot.config["replicate_token"] or None
        if self.replicate_token is None:
            logger.warning("No replicate token found in config, disabling upscaling.")
            self.upscaler = None
        else:
            logger.info("Replicate token found, enabling upscaling...")
            self.upscaler = Upscaler(token=self.replicate_token, logger=logger)

        logger.info(f"Loaded {self.qualified_name} cog.")

    async def do(self, func, *args, **kwargs):
        funcname = getattr(func, "__name__", None)
        if funcname is None:
            funcname = getattr(func.__class__, "__name__", "unknown")
        logger.info(f"Running {funcname} in background thread...")
        return await self.bot.loop.run_in_executor(self.executor, partial_func(func, *args, **kwargs))

    async def do_gpu(self, func, *args, **kwargs):
        funcname = getattr(func, "__name__", None)
        if funcname is None:
            funcname = getattr(func.__class__, "__name__", "unknown")
        logger.info(f"Running {funcname} on GPU...")
        res = await self.bot.loop.run_in_executor(self.gpu_executor, partial_func(func, *args, **kwargs))
        return res

    async def cog_load(self) -> None:
        logger.info("Loading diffusers model...")
        await self.pipe_init(SD_MODEL, torch.float32)
        logger.info("Loaded diffusers model successfully.")
        self.loading = False
        return await super().cog_load()

    async def pipe_init(self, model_name: str, torch_dtype: torch.dtype):
        model_dir = DATADIR_PATH.joinpath("models", model_name)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        logger.info(f"Loading diffusers model from {model_dir}")

        self.pipe: StableDiffusionPipeline = await self.do_gpu(
            StableDiffusionPipeline.from_pretrained, model_dir, torch_dtype=torch_dtype, local_files_only=True
        )
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        self.pipe.to("cuda")
        logger.info(f"Loaded diffusers model {model_name} successfully.")

    # Cog slash command group
    @commands.slash_command(
        name="waifu", description=f"Generate waifus with {SD_MODEL}. WARNING: NO CONTENT FILTER"
    )
    @checks.not_blacklisted()
    @commands.cooldown(1, 40.0, commands.BucketType.user)
    async def generate(
        self,
        ctx: ApplicationCommandInteraction,
        prompt: str = commands.Param(
            description="List of Danbooru tags to generate your waifu with", max_length=240
        ),
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
            result: StableDiffusionPipelineOutput = await self.do_gpu(
                self.pipe, prompt.strip(), guidance_scale=6
            )
        except Exception as e:
            raise e

        save_path = SD_DATADIR.joinpath(f"{ctx.author.id}_{round(datetime.utcnow().timestamp())}.png")
        image = result.images[0]
        if result.nsfw_content_detected[0] is True:
            logger.info(f"NSFW content detected for {ctx.user.name} from prompt '{prompt}'")
            save_path = save_path.with_suffix(".nsfw.png")

        logger.info(f"Saving image to {save_path}")
        image.save(save_path)
        image_file = File(save_path)
        await ctx.send(embed=SDEmbed(prompt, image_file, ctx.author), view=ImageView(self.bot, self.upscaler))
        return


def setup(bot):
    bot.add_cog(Waifu(bot))
