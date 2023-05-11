import json
import logging
import re
from asyncio import Lock
from base64 import b64decode
from datetime import datetime
from io import BytesIO
from pathlib import Path
from random import choice, randint
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
from dacite import from_dict
from PIL import Image

import logsnake
from ai.config import ImagenApiParams, ImagenConfig, ImagenLMPrompt, ImagenParams, ImagenSDPrompt
from ai.utils import any_in_text
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

# IMAGE_SIZE_STEPS = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960]
# IMAGE_TARGET_PIXELS = 524288  # no more than this many pixels in pre-upscale image

# Just using hardcoded options for now
IMAGE_SIZE_OPTS = [
    (512, 1024),
    (544, 960),
    (576, 896),
    (608, 864),
    (640, 832),
    (672, 768),
    (720, 744),
    (768, 688),
    (576, 576),
    (800, 640),
    (848, 544),
    (864, 600),
    (896, 585),
    (928, 564),
    (960, 480),
]


re_take_pic = re.compile(
    r".*(how about|another|capture|create|display|draw|give|make|message|paint|post|provide|see|send|send|share|shoot|show|snap|take)"
    + r"\b(.+)?\b(image|pic(ture)?|photo(graph)?|screen(shot|ie)|(paint|draw)ing|portrait|selfie)s?",
    flags=re.I + re.M,
)
re_take_pic_alt = re.compile(
    r".*(send|mail|message|me a)\b.+?\b(image|pic(ture)?|photo|snap(shot)?|selfie)s?\b",
    flags=re.I + re.M,
)
take_pic_regexes = [re_take_pic, re_take_pic_alt]

re_surrounded = re.compile(r"\*[^*]*?(\*|$)", flags=re.I + re.M)


class Imagen:
    sd_api_path = "/sdapi/v1/txt2img"

    def __init__(self, lm_api_host: str, config: Optional[ImagenConfig] = None):
        if config is None:
            config = from_dict(ImagenConfig, json.loads(IMAGEN_CFG_PATH.read_text()))

        self.config: ImagenConfig = config
        self.params: ImagenParams = config.params
        self.timezone: str = config.params.timezone

        self.sd_prompt: ImagenSDPrompt = config.sd_prompt
        self.sd_api_host: str = config.params.api_host.rstrip("/")
        self.sd_api_params: ImagenApiParams = config.api_params

        self.lm_prompt: ImagenLMPrompt = config.lm_prompt
        self.lm_api_host: str = lm_api_host.rstrip("/")

        IMAGEN_IMG_DIR.mkdir(exist_ok=True, parents=True)

        logger.info("Initialized Imagen.")

    def get_lm_prompt(self, user_request: str) -> str:
        if len(user_request.split("of", 1)) > 1:
            user_request: str = user_request.split("of", 1)[1]
            user_request = "a photo of" + user_request.strip()

        for word in ["yourself ", "you "]:
            user_request = user_request.replace(word, "a girl ")

        if "your " in user_request:
            user_request = user_request.replace("your ", "her ")

        user_request = user_request.strip("?.!")
        prompt = self.lm_prompt.prompt(user_request)
        if len(prompt.strip()) == 0:
            prompt = self.lm_prompt.default_prompt
        return prompt

    async def submit_lm_prompt(self, prompt: Optional[str] = None) -> str:
        request = self.lm_prompt.get_request(prompt)
        try:
            async with aiohttp.ClientSession(self.lm_api_host) as session:
                async with session.post("/api/v1/generate", json=request.asdict()) as resp:
                    if resp.status == 200:
                        ret = await resp.json()
                        result: str = ret["results"][0]["text"]
                        return (
                            result.replace(self.sd_prompt.get_leading(), "")
                            .replace(", ,", "")
                            .replace(",,", ",")
                            .strip()
                        )
                    else:
                        resp.raise_for_status()
        except Exception as e:
            raise Exception(f"Could not generate response. Error: {await resp.text()}") from e

    def get_lm_stopping_strings(self) -> List[str]:
        return self.lm_prompt.gensettings["stopping_strings"]

    def build_request(self, lm_tags: str, user_prompt: str) -> Dict[str, Any]:
        time_tag = get_time_tag()
        user_prompt = user_prompt.lower()
        lm_tags = lm_tags.lower()
        format_tags = ""

        if len(user_prompt) > 4 and len(lm_tags.strip()) > 2:
            if "selfie" in user_prompt:
                format_tags = ", looking at viewer"
            elif any_in_text(
                ["person", "you as", "yourself as", "you cosplaying", "yourself cosplaying"],
                user_prompt,
            ):
                format_tags = ""  # no 'standing next to' etc. when it's just the bot
            elif "with a" in user_prompt:
                format_tags = ", she has"
            elif any_in_text(["you with", "yourself with", "a selfie with"], user_prompt):
                format_tags = ", she is with "

            if "holding" in user_prompt:
                format_tags = f"{format_tags}, holding"

        if len(lm_tags) > 0:
            lm_tags = f", ({lm_tags}:{self.sd_prompt.lm_weight})"

        image_prompt = self.sd_prompt.prompt(f"{time_tag}{format_tags}{lm_tags}")

        # Generate at random aspect ratios, but same total pixels
        width, height = get_image_dimensions()

        # make sure we do portrait if the user asks for a portrait
        if "portrait" in user_prompt:
            if width > height:
                width, height = height, width

        # Build the request and return it
        gen_request: Dict[str, Any] = self.sd_api_params.get_request(
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
        async with aiohttp.ClientSession(base_url=self.sd_api_host) as session:
            async with session.post(self.sd_api_path, json=request) as r:
                if r.status == 200:
                    response: dict = await r.json()
                else:
                    r.raise_for_status()

                try:
                    # throw an exception if we got no image
                    if response["images"] is None or len(response["images"]) == 0:
                        raise ValueError("No image data returned from Imagen API", args=response)

                    # load and decode the image
                    image = Image.open(BytesIO(b64decode(response["images"][0])))
                    response.pop("images")  # don't need this anymore
                    # info is a recursively encoded json string, so we need to decode it
                    response["info"] = json.loads(response["info"])

                    # work out the path to save the image to, then save it and the job info
                    imagefile_path = IMAGEN_IMG_DIR.joinpath(
                        f'{response["info"]["job_timestamp"]}_{response["info"]["seed"]}-{req_string}.png'
                    )
                    image.save(imagefile_path)
                    imagefile_path.with_suffix(".json").write_text(
                        json.dumps(response, indent=2, skipkeys=True, default=str)
                    )
                    # this could return the image object, but we're saving it anyway and it's easier to
                    # load a disnake File() from a path, so, uh... memory leak prevention? :sweat_smile:
                    return imagefile_path
                except Exception as e:
                    logger.exception("Error saving image")
                    raise e

    def should_take_pic(self, message: str) -> bool:
        """
        Dumb regex matcher to detect photo requests
        TODO: Use a model instead of regexes? RoBERTa in multi-choice mode?
        """
        message = remove_surrounded_chars(message)
        for regex in take_pic_regexes:
            if bool(regex.search(message)) is True:
                logger.debug("Message matched take pic regex")
                return True
        return False

    def strip_take_pic(self, message: str) -> str:
        """
        Strip everything before "take a (photo|etc)" from a message
        """
        message = remove_surrounded_chars(message)
        for regex in take_pic_regexes:
            message = regex.sub("", message)
        return message.strip()


# picks a random image size from the above list, swapping width and height 50% of the time
def get_image_dimensions() -> Tuple[int, int]:
    """
    Pick a random image size from <this>.IMAGE_SIZE_OPTS, swap the width and height 50% of the time.
    Used to decide what resolution the diffusion model should generate at.
    """
    dims = choice(IMAGE_SIZE_OPTS)
    if randint(0, 1) == 0:
        return dims
    else:
        return (dims[1], dims[0])


def get_time_tag(time: Optional[datetime] = None, tz=ZoneInfo("Asia/Tokyo")) -> str:
    """
    Get a natural language word for the time of day it is. We use this to prompt the diffusion model.
    """
    hour = time.hour if time is not None else datetime.now(tz=tz).hour
    if hour in range(5, 12):
        return "morning"
    elif hour in range(12, 17):
        return "afternoon"
    elif hour in range(18, 21):
        return "evening"
    elif hour == 21:
        return "dusk"
    else:
        return "night"


def remove_surrounded_chars(string):
    """
    Removes as few symbols as possible between any asterisk pairs, or everything from the last
    asterisk to the end of the string (if there's an odd number of asterisks)

    I think. Regexes are weird.
    """
    return re_surrounded.sub("", string)
