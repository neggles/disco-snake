import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial as partial_func
from pathlib import Path
from time import perf_counter

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
TORCH_DTYPE = torch.float32


class SDEmbed(Embed):
    def __init__(
        self,
        prompt: str,
        image: File,
        author: User | Member,
        nsfw: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            description=prompt,
            colour=author.colour if isinstance(author, Member) else Colour(0xFFD01C),
            *args,
            **kwargs,
        )
        logger.debug(
            f"Creating {COG_UID} SDEmbed with image {image.filename} and nsfw={nsfw}"
            + f"for user {author.display_name} ({author.id})"
        )
        try:
            self._imagename = image.filename
            image.spoiler = nsfw
            self.set_image(file=image)
            self.set_author(name=author.display_name, icon_url=author.display_avatar.url)
            self.set_footer(text="Powered by Huggingface Diffusers 🤗🧨")
        except Exception as e:
            logger.error(e)
            raise e


class ImageView(ui.View):
    def __init__(self, bot: DiscoSnake, author: User | Member, upscaler: Upscaler = None, **kwargs):
        super().__init__(timeout=None)
        self.bot: DiscoSnake = bot
        self.upscaler = upscaler
        self.author = author
        self.kwargs = kwargs
        if upscaler is None:
            self.upscale_button.disabled = True
            self.upscale_button.label = "❌ No Upscaler"

    @ui.button(label="Upscale", style=ButtonStyle.blurple, custom_id=f"{COG_UID}_ImageView:upscale")
    async def upscale_button(self, button: ui.Button, ctx: MessageInteraction):
        await ctx.response.defer()

        self.upscale_button.disabled = True
        self.upscale_button.label = "Upscaling..."
        await ctx.edit_original_response(view=self)

        embed: SDEmbed = ctx.message.embeds[0]
        src_url = embed.image.url
        src_name = Path(embed.image.url.split("/")[-1])

        try:
            upscaled = await self.bot.do(self.upscaler.upscale, url=src_url, download=True)
            upscaled_name = str(src_name.stem + "-upscaled" + src_name.suffix)
            SD_DATADIR.joinpath(str(ctx.author.id), upscaled_name).write_bytes(upscaled.read())
            upscaled.seek(0)

            image = File(upscaled, filename=upscaled_name)
            embed.set_image(file=image).set_footer(
                text="Powered by Huggingface Diffusers 🤗🧨 and Replicate.com 🧬"
            )
            self.upscale_button.label = "✔️ Upscaled"
        except Exception as e:
            await ctx.followup.send(f"Upscale failed: {e}")
            self.upscale_button.label = "❌ Failed"
            self.upscale_button.style = ButtonStyle.red
            logger.error(e)
        finally:
            await ctx.edit_original_response(embed=embed, attachments=None, view=self)
            return

    @ui.button(label="Retry", style=ButtonStyle.green, custom_id=f"{COG_UID}_ImageView:retry")
    async def retry_button(self, button: ui.Button, ctx: MessageInteraction):
        await ctx.response.defer()
        try:
            upscale_state = self.upscale_button.disabled
            self.upscale_button.disabled = True
            self.retry_button.disabled = True
            self.retry_button.label = "Retrying..."
            self.retry_button.style = ButtonStyle.grey
            await ctx.edit_original_response(view=self)

            # restore upscale button in case of failure
            self.upscale_button.disabled = upscale_state

            # Generate new embed
            logger.info(f"Retrying {COG_UID} generation for {ctx.author.display_name} ({ctx.author.id})")
            logger.debug(f"ctx: {ctx.__dict__}")
            logger.debug(f"kwargs: {self.kwargs}")
            embed = await self.bot.cogs[COG_UID].generate_embed(author=self.author, **self.kwargs)

            self.retry_button.label = "Complete"
            self.retry_button.style = ButtonStyle.green
            await ctx.followup.send(
                embed=embed,
                view=ImageView(bot=self.bot, upscaler=self.upscaler, author=ctx.author, **self.kwargs),
            )
        except Exception as e:
            await ctx.followup.send(f"Retry failed: {e}")
            logger.error(e)
        finally:
            await ctx.edit_original_response(view=self)
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
        await self.pipe_init(SD_MODEL, TORCH_DTYPE)
        self.loading = False
        return await super().cog_load()

    async def pipe_init(self, model_name: str, torch_dtype: torch.dtype):
        model_dir = DATADIR_PATH.joinpath("models", model_name)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        logger.info(f"Loading diffusers model from {model_dir}")

        self.pipe: StableDiffusionPipeline = await self.do_gpu(
            StableDiffusionPipeline.from_pretrained,
            model_dir,
            torch_dtype=torch_dtype,
            local_files_only=True,
            safety_checker=lambda images, **kwargs: (images, [False] * len(images)),
        )
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
        logger.info(f"Loaded diffusers model {model_name} successfully.")

    async def generate_embed(self, prompt, steps, author: User | Member, **kwargs):
        if not prompt:
            raise ValueError("i can't generate an image from nothing...")
        if self.pipe is None:
            raise ValueError("Pipeline is not ready yet, please try again in a few seconds.")

        try:
            start_time = perf_counter()
            result: StableDiffusionPipelineOutput = await self.do_gpu(
                self.pipe,
                prompt=f"mdjrny-v4 style {prompt.strip()}",
                num_inference_steps=round(steps),
                **kwargs,
            )
            run_duration = perf_counter() - start_time
            logger.info(f"Generated in {run_duration:.2f}s")
        except Exception as e:
            logger.error(e)
            raise e

        SD_DATADIR.joinpath(str(author.id)).mkdir(parents=True, exist_ok=True)
        save_path = SD_DATADIR.joinpath(str(author.id), f"{round(datetime.utcnow().timestamp())}.png")
        image = result.images[0]
        nsfw = result.nsfw_content_detected[0]
        if nsfw is True:
            logger.info(f"NSFW content detected for {author.name} from prompt '{prompt}'")
            save_path = save_path.with_suffix(".nsfw.png")

        logger.info(f"Saving image to {save_path}")
        image.save(save_path)
        image = File(fp=save_path, filename=save_path.name)
        embed = SDEmbed(prompt, image, author, nsfw)
        return embed

    # Cog slash command group
    @commands.slash_command(
        name="waifu", description=f"Generate waifus with {SD_MODEL}. WARNING: NO CONTENT FILTER"
    )
    @checks.not_blacklisted()
    @commands.cooldown(1, 40.0, commands.BucketType.user)
    async def generate_command(
        self,
        ctx: ApplicationCommandInteraction,
        prompt: str = commands.Param(description="Prompt to generate an image from.", max_length=240),
        steps: float = commands.Param(
            description="Number of steps to run the model for.", default=50.0, min_value=25.0, max_value=100.0
        ),
        guidance: float = commands.Param(
            description="Higher values follow the prompt more closely at the expense of image quality.",
            default=7.5,
            min_value=1.0,
            max_value=25.0,
        ),
        negative: str = commands.Param(
            description="Negative prompt to steer the model away from", max_length=240, default=None
        ),
    ):
        """
        make an image from a prompt
        """
        if not prompt:
            await ctx.send("i can't generate an image from nothing...", ephemeral=True)
            return
        if self.pipe is None:
            await ctx.send("Pipeline is not ready yet, please try again in a few seconds.", ephemeral=True)
            return

        await ctx.response.defer()
        logger.info(f"Generating image for {ctx.user.name} from prompt '{prompt}'")

        generate_args = {
            "prompt": prompt,
            "steps": steps,
            "author": ctx.author,
            "guidance_scale": guidance,
        }
        if negative is not None:
            generate_args["negative_prompt"] = negative

        embed = await self.generate_embed(**generate_args)
        await ctx.edit_original_response(
            embed=embed,
            view=ImageView(bot=self.bot, upscaler=self.upscaler, **generate_args),
        )
        return


def setup(bot):
    bot.add_cog(Waifu(bot))
