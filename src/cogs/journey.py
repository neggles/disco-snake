import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial as partial_func
from pathlib import Path
from random import uniform as rand_float
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
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

try:
    import xformers
except ImportError:
    xformers = None

COG_UID = "journey"
GUIDANCE_DEFAULT = 9.1

# set diffusers logger to info
d2logging.set_verbosity_info()

# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=COG_UID,
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{COG_UID}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=2,
)

# set up the diffusers logger
d2logger = d2logging.get_logger("diffusers")
for handler in logger.handlers:
    d2logger.addHandler(handler)

SD_DATADIR = DATADIR_PATH.joinpath("sd", COG_UID)
SD_DATADIR.mkdir(parents=True, exist_ok=True)
SD_MODEL = "openjourney"

PARAM_DISPLAY = {
    "model": "Model",
    "prompt": "Prompt",
    "negative_prompt": "Antiprompt",
    "num_inference_steps": "Steps",
    "guidance_scale": "Guidance",
    "width": "Width",
    "height": "Height",
}


class SDEmbed(Embed):
    def __init__(
        self,
        prompt: str,
        image: File,
        author: User | Member,
        nsfw: bool,
        model_params: dict,
        run_duration: float = None,
        **kwargs,
    ):
        super().__init__(
            description=prompt,
            colour=author.colour if isinstance(author, Member) else Colour(0xFFD01C),
            **kwargs,
        )
        logger.debug(
            f"Creating {COG_UID} SDEmbed with image {image.filename} and nsfw={nsfw} "
            + f"for user {author.display_name} ({author.id})"
        )
        try:
            image.spoiler = nsfw

            if model_params is not None:
                for key, val in model_params.items():
                    if val is None:
                        self.add_field(name=PARAM_DISPLAY[key], value="None", inline=True)
                    elif isinstance(val, (int, float, str)):
                        self.add_field(name=PARAM_DISPLAY[key], value=val, inline=True)

            if run_duration is not None:
                self.add_field(name="Runtime", value=f"{run_duration:.2f}s", inline=True)

            self.set_image(file=image)
            self.set_author(name=author.display_name, icon_url=author.display_avatar.url)
            self.set_footer(text="Powered by Huggingface Diffusers 🤗🧨")
        except Exception as e:
            logger.error(e)
            raise e


class ImageView(ui.View):
    def __init__(
        self,
        bot: DiscoSnake,
        author: User | Member,
        prompt: str,
        model_params: dict = None,
        upscaler: Upscaler = None,
    ):
        super().__init__(timeout=None)
        self.bot: DiscoSnake = bot
        self.upscaler = upscaler
        self.author = author
        self.prompt = prompt
        self.model_params = model_params
        if upscaler is None:
            self.upscale_button.disabled = True
            self.upscale_button.label = "❌ No Upscaler"

    @ui.button(label="Upscale", style=ButtonStyle.blurple, custom_id=f"{COG_UID}_ImageView:upscale")
    async def upscale_button(self, button: ui.Button, ctx: MessageInteraction):
        await ctx.response.defer()
        try:
            # Disable the upscale button while the upscale is in progress
            self.upscale_button.disabled = True
            self.retry_button.disabled = True
            self.upscale_button.label = "Upscaling..."
            await ctx.edit_original_response(view=self)

            # Restore the retry button in case the upscale fails
            self.retry_button.disabled = False

            embed: SDEmbed = ctx.message.embeds[0]
            src_url = embed.image.url
            src_name = Path(embed.image.url.split("/")[-1])

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
            # Switch into disabled states to prevent double-clicking and race conditions
            upscale_state = self.upscale_button.disabled
            self.upscale_button.disabled = True
            self.retry_button.disabled = True
            self.retry_button.label = "Retrying..."
            self.retry_button.style = ButtonStyle.grey
            await ctx.edit_original_response(view=self)

            # restore upscale button value in case of failure
            self.upscale_button.disabled = upscale_state

            # Generate new embed
            logger.info(f"Retrying {COG_UID} generation for {ctx.author.display_name} ({ctx.author.id})")
            embed = await self.bot.cogs[COG_UID].generate_embed(
                prompt=self.prompt, author=ctx.author, model_params=self.model_params
            )

            # Update button state to reflect completion
            self.retry_button.label = "Complete"
            self.retry_button.style = ButtonStyle.green

            # Send new message with new embed
            await ctx.followup.send(
                embed=embed,
                view=ImageView(
                    bot=self.bot,
                    upscaler=self.upscaler,
                    author=ctx.author,
                    prompt=self.prompt,
                    model_params=self.model_params,
                ),
            )
        except Exception as e:
            await ctx.followup.send(f"Retry failed: {e}")
            logger.error(e)
        finally:
            await ctx.edit_original_response(view=self)
            return


# Here we name the cog and create a new class for the cog.
class Journey(commands.Cog, name=COG_UID):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.pipe: StableDiffusionPipeline = None  # type: ignore
        self.loading = True  # set to false once the pipe is loaded

        self.executor = ThreadPoolExecutor(
            max_workers=5, thread_name_prefix=f"{COG_UID}"
        )  # thread pool for blocking code
        self.gpu_executor = bot.gpu_executor  # thread "pool" for GPU operations

        self.torch_device = self.bot.config["diffusers"]["torch_device"] or "cpu"
        dtype_str = self.bot.config["diffusers"]["torch_dtype"] or "float32"
        self.torch_dtype: torch.dtype = getattr(torch, dtype_str)

        logger.debug(f"{COG_UID} cog using torch device {self.torch_device} and dtype {self.torch_dtype}")
        logger.debug(f"Diffusers model: models/{SD_MODEL}")

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
        await self.pipe_init(SD_MODEL)
        self.loading = False
        return await super().cog_load()

    async def pipe_init(self, model_name: str):
        model_dir = DATADIR_PATH.joinpath("models", model_name)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        logger.info(f"Loading diffusers model from {model_dir}")

        self.pipe: StableDiffusionPipeline = await self.do_gpu(
            StableDiffusionPipeline.from_pretrained,
            model_dir,
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            safety_checker=lambda images, **kwargs: (images, [False] * len(images)),
        )
        self.pipe = self.pipe.to(self.torch_device)
        if xformers is not None:
            await self.do_gpu(self.pipe.enable_xformers_memory_efficient_attention)

        logger.info(f"Loaded diffusers model {model_name} successfully.")

    async def generate_embed(self, prompt: str, author: User | Member, model_params: dict):
        if not prompt:
            raise ValueError("i can't generate an image from nothing...")
        if self.pipe is None:
            raise ValueError("Pipeline is not ready yet, please try again in a few seconds.")

        logger.info(
            "\n".join(
                [f"Generating image for {author.name}:", f"Prompt: {prompt}", f"Params: {model_params}"]
            )
        )

        try:
            start_time = perf_counter()
            result: StableDiffusionPipelineOutput = await self.do_gpu(
                self.pipe, prompt=f"mdjrny-v4 style {prompt.strip()}", **model_params
            )
            run_duration = perf_counter() - start_time
            logger.info(f"Generated in {run_duration:.2f}s")
            image = result.images[0]
            nsfw = result.nsfw_content_detected[0]
        except Exception as e:
            logger.error(e)
            raise e

        SD_DATADIR.joinpath(str(author.id)).mkdir(parents=True, exist_ok=True)
        save_path = SD_DATADIR.joinpath(str(author.id), f"{round(datetime.utcnow().timestamp())}.png")
        if nsfw is True:
            logger.info(f"NSFW content detected for {author.name} from prompt '{prompt}'")
            save_path = save_path.with_suffix(".nsfw.png")
        logger.info(f"Saving image to {save_path}")
        image.save(save_path)

        image = File(fp=save_path, filename=save_path.name)
        embed = SDEmbed(
            prompt=prompt,
            image=image,
            author=author,
            nsfw=nsfw,
            model_params=model_params,
            run_duration=run_duration,
        )
        return embed

    # Cog slash command group
    @commands.slash_command(
        name="journey", description=f"Generate images with {SD_MODEL}. WARNING: NO CONTENT FILTER"
    )
    @checks.not_blacklisted()
    @commands.cooldown(1, 35.0, commands.BucketType.user)
    async def generate_command(
        self,
        ctx: ApplicationCommandInteraction,
        prompt: str = commands.Param(
            description="Prompt to generate an image from.",
            max_length=240,
        ),
        steps: float = commands.Param(
            description="Number of steps to run the model for.",
            default=50.0,
            min_value=25.0,
            max_value=100.0,
        ),
        guidance: float = commands.Param(
            description="Higher values follow the prompt more closely at the expense of image quality.",
            default=GUIDANCE_DEFAULT,
            min_value=1.0,
            max_value=30.0,
        ),
        negative: str = commands.Param(
            description="Negative prompt to steer the model away from",
            default="",
            max_length=240,
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

        # send thinking notification
        await ctx.response.defer()

        # randomize guidance if it's set to default
        if guidance == GUIDANCE_DEFAULT:
            guidance = rand_float(6.0, 12.0)

        model_params = {
            "num_inference_steps": round(steps),
            "guidance_scale": round(guidance, 2),
        }

        if negative != "":
            model_params["negative_prompt"]: negative

        embed = await self.generate_embed(prompt=prompt, author=ctx.author, model_params=model_params)
        await ctx.edit_original_response(
            embed=embed,
            view=ImageView(
                bot=self.bot,
                upscaler=self.upscaler,
                author=ctx.author,
                prompt=prompt,
                model_params=model_params,
            ),
        )
        return


def setup(bot):
    bot.add_cog(Journey(bot))
