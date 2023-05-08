import json
import logging
import re
from base64 import b64decode
from datetime import datetime
from io import BytesIO
from pathlib import Path
from random import choice, randint
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
import logsnake
from dacite import from_dict
from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from PIL import Image

from ai.config import ImagenApiParams, ImagenConfig, ImagenLMPrompt, ImagenParams, ImagenSDPrompt
from ai.utils import any_in_text, get_current_time

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

IMAGE_SIZE_OPTS = [
    (1024, 576),
    (512, 1024),
    (544, 960),
    (576, 1024),
    (576, 896),
    (600, 864),
    (640, 832),
    (720, 800),
    (768, 768),
]


re_take_pic = re.compile(
    r"(another|capture|create|display|draw|give|make|message|paint|post|provide|see|send|send|share|shoot|show|snap|take)"
    + r"\b(.+)?\b(image|pic(ture)?|photo(graph)?|screen(shot|ie)|(paint|draw)ing|portrait|selfie)s?",
    flags=re.I + re.M,
)
re_take_pic_alt = re.compile(
    r"(send|mail|message|me a)\b.+?\b(image|pic(ture)?|photo|snap(shot)?|selfie)s?\b",
    flags=re.I + re.M,
)
take_pic_regexes = [re_take_pic, re_take_pic_alt]

re_surrounded = re.compile(r"\*[^*]*?(\*|$)", flags=re.I + re.M)


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

        logger.info("Initialized Imagen.")
        # logger.info(json.dumps(self.config.asdict(), indent=2))

    def get_lm_prompt(self, user_request: str) -> str:
        return self.lm_prompt.prompt(user_request)

    def get_lm_stopping_strings(self) -> List[str]:
        return self.lm_prompt.stopping_strings

    def build_request(self, llm_tags: str, user_prompt: str) -> Dict[str, Any]:
        time_tag = get_time_tag()
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

    def should_take_pic(self, message: str) -> bool:
        message = remove_surrounded_chars(message)
        for regex in take_pic_regexes:
            if bool(regex.search(message)) is True:
                logger.debug(f"Matched take pic regex: {regex.pattern}")
                return True
        return False

    def strip_take_pic(self, message: str) -> str:
        message = remove_surrounded_chars(message)
        for regex in take_pic_regexes:
            message = regex.sub("", message)
        return message.strip()


# picks a random image size from the above list, swapping width and height 50% of the time
def get_image_dimensions() -> Tuple[int, int]:
    dims = choice(IMAGE_SIZE_OPTS)
    if randint(0, 1) == 0:
        return dims
    else:
        return (dims[1], dims[0])


def get_time_tag(time: Optional[datetime] = None, tz=ZoneInfo("Asia/Tokyo")) -> str:
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
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re_surrounded.sub("", string)
