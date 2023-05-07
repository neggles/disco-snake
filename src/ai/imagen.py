import logging
import json
from base64 import b64decode
from io import BytesIO
from typing import Dict, Any, Optional, List
from pathlib import Path

import aiohttp
from dacite import from_dict
from PIL import Image

import logsnake
from ai.config import ImagenConfig, ImagenApiParams, ImagenParams, ImagenSDPrompt, ImagenLMPrompt
from ai.utils import any_in_text, get_current_time, get_image_dimensions, get_image_time_tag
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH

# setup cog logger
logger = logsnake.setup_logger(
    level=logging.DEBUG,
    isRootLogger=False,
    name="imagen",
    formatter=logsnake.LogFormatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"),
    logfile=LOGDIR_PATH.joinpath("imagen.log"),
    fileLoglevel=logging.DEBUG,
    maxBytes=1 * (2**20),
    backupCount=2,
)

IMAGEN_IMG_DIR = DATADIR_PATH.joinpath("ai", "images")
IMAGEN_CFG_PATH = DATADIR_PATH.joinpath("ai", "imagen.json")


class Imagen:
    def __init__(self, config: Optional[ImagenConfig] = None):
        if config is None:
            config = from_dict(ImagenConfig, json.loads(IMAGEN_CFG_PATH.read_text()))

        self.config: ImagenConfig = config
        self.params: ImagenParams = config.params
        self.lm_prompt: ImagenLMPrompt = config.lm_prompt
        self.sd_prompt: ImagenSDPrompt = config.sd_prompt
        self.timezone: str = config.params.timezone
        self.api_host: str = config.params.api_host.rstrip("/")
        self.api_params: ImagenApiParams = config.api_params

        self.api_endpoint = "/sdapi/v1/txt2img"
        IMAGEN_IMG_DIR.mkdir(exist_ok=True, parents=True)

        logger.info("Initialized Imagen. Dump config:")
        logger.info(json.dumps(self.config.asdict(), indent=2))

    async def close(self):
        await self.session.close()

    def get_lm_prompt(self, user_request: str) -> str:
        return self.lm_prompt.prompt(user_request)

    def get_lm_stopping_strings(self) -> List[str]:
        return self.lm_prompt.stopping_strings

    def build_request(self, llm_tags: str, user_prompt: str) -> Dict[str, Any]:
        time_tag = get_image_time_tag()
        user_prompt = user_prompt.lower()
        llm_tags = llm_tags.lower()
        format_tags = ""

        if len(user_prompt) > 4 and len(llm_tags.strip()) > 2:
            if "selfie" in user_prompt:
                format_tags = ", looking into the camera, "
            elif any_in_text(
                ["person", "you as", "yourself as", "you cosplaying", "yourself cosplaying"],
                user_prompt,
            ):
                format_tags = ""  # no 'standing next to' etc. when it's just the bot
            elif "with a" in user_prompt:
                format_tags = ", she has"
            elif "of you with" in user_prompt:
                format_tags = ", she is with"

            if "holding" in user_prompt:
                format_tags = f"{format_tags}, holding"

        if len(llm_tags) > 0:
            llm_tags = f"({llm_tags}:1.2)"

        image_prompt = self.sd_prompt.prompt(f"{time_tag}, {format_tags} {llm_tags}")

        # Generate at random aspect ratios, but same total pixels
        width, height = get_image_dimensions()

        # make sure we do portrait if the user asks for a selfie or portrait
        if "selfie" in user_prompt or "portrait" in user_prompt:
            if width > height:
                width, height = height, width

        # Build the request and return it
        gen_request: Dict[str, Any] = self.api_params.get_request(
            prompt=image_prompt,
            negative=self.sd_prompt.negative_prompt(),
            width=width,
            height=height,
        )
        logger.debug(f"Generated request: {gen_request}")
        return gen_request

    async def submit_request(self, request: Dict[str, Any]) -> Path:
        # make a tag for the filename
        req_string: str = request["prompt"][:50]
        req_string = req_string.replace(" ", "-").replace(",", "")
        # submit the request and save the image
        async with aiohttp.ClientSession(base_url=self.api_host) as session:
            async with session.post("/sdapi/v1/txt2img", json=request) as r:
                if r.status == 200:
                    response = await r.json()
                    image_data = response["images"][0]
                    image_info = json.loads(response["info"])
                    image = Image.open(BytesIO(b64decode(image_data)))
                    imagefile_path = IMAGEN_IMG_DIR.joinpath(
                        f"{image_info['job_timestamp']}_{image_info['seed']}-{req_string}.png"
                    )
                    image.save(imagefile_path)
                    return imagefile_path
                else:
                    r.raise_for_status()
