import logging
from datetime import datetime
from pathlib import Path
from io import FileIO
from asyncio import sleep as async_sleep

import replicate
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from disnake import ApplicationCommandInteraction, Embed, File, MessageInteraction, User, ButtonStyle, ui
from disnake.ext import commands

import logsnake
from disco_snake import DATADIR_PATH, LOGDIR_PATH
from disco_snake.bot import DiscoSnake
from helpers import checks

# setup package logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name=Path(__file__).stem,
    formatter=logsnake.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath(f"{Path(__file__).stem}.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=2 * (2**20),
    backupCount=5,
)

SD_DATADIR = DATADIR_PATH.joinpath("sd")
SD_DATADIR.mkdir(parents=True, exist_ok=True)

COG_UID = "journey"


class Upscaler:
    def __init__(self, token: str):
        self._client = replicate.Client(api_token=token)
        self._model = self._client.models.get("jingyunliang/swinir")
        self._upscaler = self._model.versions.get(
            "660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"
        )

    async def upscale(self, image: FileIO, large: bool = False) -> str:
        if large is True:
            task_type = "Real-World Image Super-Resolution-Large"
        else:
            task_type = "Real-World Image Super-Resolution-Medium"
        result = self._upscaler.predict(image=image, task_type=task_type)

        # wait for prediction to finish
        while result.status not in ["succeeded", "failed", "canceled"]:
            async_sleep(0.5)
            result.reload()

        if result.status in ["failed", "canceled"]:
            raise RuntimeError(f"Upscaling returned result '{result.status}' :(")
        return result


class SDEmbed(Embed):
    def __init__(self, prompt: str, image_file: File | str, requestor: User, *args, **kwargs):
        super().__init__(title=f"{prompt}:", *args, **kwargs)

        self.colour = int(requestor.accent_colour) if requestor.accent_colour is not None else 0xFFD01C

        if isinstance(image_file, File):
            self.set_image(file=image_file)
        else:
            self.set_image(url=image_file)
        self.set_author(name=requestor.display_name, icon_url=requestor.avatar.url)
        self.set_footer(text="Powered by Huggingface Diffusers 🤗🧨")


class ImageView(ui.View):
    def __init__(self, upscaler: Upscaler):
        super().__init__(timeout=180.0)
        self.upscaler = upscaler

    @ui.button(label="Upscale", style=ButtonStyle.primary, custom_id=f"{COG_UID}_ImageView:upscale")
    async def upscale_button(self, button: ui.Button, ctx: MessageInteraction):
        await ctx.response.defer()
        image = ctx.message.attachments[0].url
        try:
            image = await self.upscaler.upscale(image)
        except RuntimeError as e:
            await ctx.response.send_message(e)
            return

        prompt = ctx.message.embeds[0].title
        prompt = prompt[:-1] if prompt.endswith(":") else prompt
        self.upscale_button.disabled = True
        self.upscale_button.label = "Upscaled!"
        await ctx.response.edit_message(embed=SDEmbed(prompt, image, ctx.message.author), view=self)
        pass


# Here we name the cog and create a new class for the cog.
class Journey(commands.Cog, name="journey"):
    def __init__(self, bot: DiscoSnake):
        self.bot: DiscoSnake = bot
        self.replicate_token = self.bot.config["replicate_token"] or None
        if self.replicate_token is None:
            logger.warning("No replicate token found in config, disabling upscaling.")
            self.upscaler = None
        else:
            self.upscaler = Upscaler(self.replicate_token)

        logger.info(f"Loaded {self.qualified_name} cog.")

    async def cog_load(self) -> None:
        await self.pipe_init()
        return await super().cog_load()

    async def pipe_init(
        self,
        model_name: str = "openjourney",
        torch_dtype: torch.dtype = torch.float32,
    ):
        model_dir = Path(DATADIR_PATH) / "models" / model_name
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
    @commands.slash_command(name="journey", description="Generate and edit images with DALL·E")
    @commands.cooldown(1, 60.0, commands.BucketType.user)
    async def generate(
        self,
        ctx: ApplicationCommandInteraction,
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
            result: StableDiffusionPipelineOutput = self.pipe(f"{prompt.strip()}, mdjrny-v4 style")
        except Exception as e:
            raise e

        save_path = SD_DATADIR.joinpath(f"{ctx.author.id}_{round(datetime.utcnow().timestamp())}.png")
        image = result.images[0]
        if result.nsfw_content_detected[0] is True:
            logger.info(f"NSFW content detected for {ctx.user.name} from prompt '{prompt}'")
            save_path = save_path.with_suffix(".nsfw.png")
            logger.info(f"Saving NSFW image to {save_path}")
            image.save(save_path)
            await ctx.send("that prompt was too spicy for me to handle...")
            return

        logger.info(f"Saving image to {save_path}")
        image.save(save_path)
        image_file = File(save_path)
        await ctx.send(embed=SDEmbed(prompt, image_file, ctx.author), view=ImageView(self.upscaler))
        return


def setup(bot):
    bot.add_cog(Journey(bot))
