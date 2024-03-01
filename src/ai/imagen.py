import json
import logging
import re
from asyncio import Lock
from base64 import b64decode
from datetime import datetime
from io import BytesIO
from math import ceil, sqrt
from pathlib import Path
from random import choice, randint
from typing import TYPE_CHECKING, Any, Optional, Tuple
from zoneinfo import ZoneInfo

import aiohttp
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ai.settings import (
    IMAGES_DIR,
    ImagenApiParams,
    ImagenLMPrompt,
    ImagenParams,
    ImagenSDPrompt,
    ImagenSettings,
    get_imagen_settings,
)
from ai.utils import any_in_text

if TYPE_CHECKING:
    from ai import Ai

# setup cog logger
logger = logging.getLogger(__name__)

# Just using hardcoded options for now
IMAGE_SIZE_OPTS = [
    # 576s
    (576, 576),
    (576, 640),
    (576, 768),
    (576, 832),
    (576, 896),
    # 640s
    (640, 640),
    (640, 704),
    (640, 768),
    (640, 832),
    (640, 896),
    # 704s
    (704, 704),
    (704, 768),
    (704, 832),
    (704, 896),
    # 768s
    (768, 768),
    (768, 832),
]


re_take_pic = re.compile(
    r".*(another|capture|create|display|draw|give|make|message|paint|post|provide|see|send|send|share|shoot|show|snap|take)"
    + r"\b(.+)?\b(image|pic(ture)?|photo(graph)?|screen(shot|ie)|(paint|draw)ing|portrait|selfie)s?",
    flags=re.I + re.M,
)
re_take_pic_alt = re.compile(
    r".*(send|mail|message)\b(.+)?\b(image|pic(ture)?|photo|snap(shot)?|selfie)s?\b",
    flags=re.I + re.M,
)
take_pic_regexes = [re_take_pic, re_take_pic_alt]
re_surrounded = re.compile(r"\*[^*]*?(\*|$)", flags=re.I + re.M)
re_clean_filename = re.compile(r"[^a-zA-Z0-9_\- ]+")  # for removing non-alphanumeric characters
re_single_dash = re.compile(r"-+")  # for removing multiple dashes
re_fix_commas = re.compile(r",[,\s]*", re.I + re.M)

re_eyes_hair = re.compile(
    r"(red|blue|brown|black|grey|silver|gold|yellow|orange|green|purple|violet|indigo|pink)"
    + r" (eye|hair)s?",
    re.I + re.M,
)


re_image_description = re.compile(r"\[\s*image:? ([^\]]+)\s*\]?", re.I + re.M)
re_send_image = re.compile(r"[\[(].?(?:send|sends|sending) a? ?([^)\]]+)[)\]]", re.I + re.M)
send_pic_regexes = [re_image_description, re_send_image]


def ceil_to_interval(val: int, interval: int = 64) -> int:
    """Rounds up to the nearest multiple of `interval` (default 64)"""
    return ceil(val / interval) * interval


class Imagen:
    SD_API_PATH = "/sdapi/v1/txt2img"

    def __init__(self, cog: "Ai") -> None:
        self.ai: "Ai" = cog
        self.config: ImagenSettings = get_imagen_settings()
        self.params: ImagenParams = self.config.params
        self.timezone: str = self.params.timezone

        self.sd_prompt: ImagenSDPrompt = self.config.sd_prompt
        self.sd_api_host: str = self.params.api_host.rstrip("/")
        self.sd_api_params: ImagenApiParams = self.config.api_params

        self.lm_prompt: ImagenLMPrompt = self.config.lm_prompt
        self.lm_api_host: str = self.ai.provider_config.endpoint.rstrip("/")

        IMAGES_DIR.mkdir(exist_ok=True, parents=True)

        self._lock = Lock()

        self._last_request: str = ""
        self._last_llm_tags: str = ""

        logger.info("Initialized Imagen.")

    @property
    def enabled(self):
        return self.params.enabled

    def get_lm_prompt(self, user_request: str) -> str:
        if len(user_request.split("of", 1)) > 1:
            user_request: str = user_request.split("of", 1)[1]
            user_request = "a photo of " + user_request.strip()

        for word in ["yourself ", "you "]:
            user_request = user_request.replace(word, f"a {self.params.chartype} ")

        if "your " in user_request:
            user_request = user_request.replace("your ", "her ")

        user_request = user_request.strip("?.!")
        self._last_request = user_request
        prompt = self.lm_prompt.wrap_prompt(user_request)
        if len(prompt.strip()) == 0:
            prompt = self.lm_prompt.default_prompt
        return prompt, user_request

    async def submit_lm_prompt(self, prompt: Optional[str] = None) -> str:
        request = self.lm_prompt.get_request(prompt)
        payload = request.dict(exclude_none=True)
        logger.debug(f"Sending request: {json.dumps(payload, default=str, ensure_ascii=False)}")

        try:
            async with aiohttp.ClientSession(base_url=self.lm_api_host) as session:
                async with session.post("/v1/completions", json=payload) as resp:
                    if resp.status == 200:
                        ret = await resp.json(encoding="utf-8")
                        result: str = ret["choices"][0]["text"]
                        for tag in self.lm_prompt.tags:
                            result = result.replace(f"{tag},", "").replace(f"{tag}", "").strip()
                        result = re_fix_commas.sub(", ", result)
                        return result
                    else:
                        resp.raise_for_status()
        except Exception as e:
            raise Exception(f"Could not generate response. Error: {await resp.text()}") from e

    def get_lm_stopping_strings(self) -> list[str]:
        return self.lm_prompt.gensettings.stop

    def build_request(self, lm_tag_string: str, user_prompt: str) -> dict[str, Any]:
        user_prompt = user_prompt.lower()
        lm_tag_string = lm_tag_string.lower()
        prompt_tags = [get_time_tag()]
        lm_tags = []

        if len(user_prompt) > 4 and len(lm_tag_string.strip()) > 2:
            if any_in_text(
                ["portrait", "vertical", "of you ", "of yourself ", "selfie"],
                user_prompt,
            ):
                prompt_tags.append("looking at viewer")
            elif any_in_text(
                ["person", "you as", "yourself as", "you cosplaying", "yourself cosplaying"],
                user_prompt,
            ):
                prompt_tags.append("cowboy shot")

            if "holding" in user_prompt:
                prompt_tags.append("holding")

            # filter out banned tags from the LM's generated tags
        if len(lm_tag_string) > 0:
            # split tags, strip whitespace, remove banned tags, rejoin
            lm_tags_split = [x.strip().lower() for x in lm_tag_string.split(",") if len(x) > 3]
            for tag in lm_tags_split:
                if any((re.search(x, tag, re.I) for x in self.sd_prompt.banned_tags)):
                    logger.debug(f"Removing banned tag '{tag}'")
                    continue
                if self.sd_prompt.cleanup_desc and re_eyes_hair.search(tag, re.I):
                    logger.debug(f"Removing eye/hair description tag '{tag}'")
                    continue
                lm_tags.append(tag)

            removed = len(lm_tags_split) - len(lm_tags)
            lm_tags = list(set(lm_tags))  # remove duplicates
            duplicate = removed - (len(lm_tags_split) - len(lm_tags))
            logger.debug(f"Removed {removed} banned and {duplicate} duplicate LM tags, remaining: {lm_tags}")

        lm_tags = [x.replace(" ", self.sd_prompt.word_sep) for x in lm_tags]
        prompt_tags = [x.replace(" ", self.sd_prompt.word_sep) for x in prompt_tags]

        self._last_llm_tags = self.sd_prompt.tag_sep.join(lm_tags)
        if len(lm_tags) > 0:
            prompt_tags.append(f"({self._last_llm_tags}:{self.sd_prompt.lm_weight})")
        logger.debug(f"Final prompt tags: {prompt_tags}")

        # Generate at random aspect ratios, but same total pixels
        width, height = get_image_dimensions()

        if any_in_text(["portrait", "vertical", "selfie"], user_prompt):
            # do portrait if portrait is requested
            width, height = min(width, height), max(width, height)
        if any_in_text(["landscape", "horizontal", "the view"], user_prompt):
            # do landscape if landscape requested
            width, height = max(width, height), min(width, height)
        if any_in_text(["instagram", "square"], user_prompt):
            # if square, do square (1:1)
            pixels = width * height
            # but round to nearest 64px
            width = height = ceil_to_interval(sqrt(pixels))

        # Build the request and return it
        gen_request: dict[str, Any] = self.sd_api_params.get_request(
            prompt=self.sd_prompt.wrap_prompt(prompt_tags),
            negative=self.sd_prompt.get_negative(join=True),
            width=width,
            height=height,
        )
        logger.debug(f"Generated request: {gen_request}")
        return gen_request

    async def submit_request(self, request: dict[str, Any]) -> Path:
        # make a tag for the filename
        req_string: str = request["prompt"][:50]
        req_string = req_string.replace(" ", "-").replace(",", "")  # remove spaces and commas
        req_string = re_clean_filename.sub("", req_string)  # trim fs-unfriendly characters
        req_string = re_single_dash.sub("-", req_string)  # collapse consecutive dashes
        req_string = req_string.rstrip("-")[:64]  # remove trailing dashes, truncate to 64 chars

        # submit the request and save the image
        if self._lock.locked():
            raise RuntimeError("Cannot submit request while another request is in progress")
        async with self._lock:  # one at a time, people
            async with aiohttp.ClientSession(base_url=self.sd_api_host) as session:
                async with session.post(self.SD_API_PATH, json=request) as r:
                    if r.status == 200:
                        response: dict = await r.json()
                    else:
                        r.raise_for_status()
                    try:
                        # throw an exception if we got no image
                        if response["images"] is None or len(response["images"]) == 0:
                            raise ValueError("No image data returned from Imagen API", args=response)

                        # glue the LLM prompt onto the response
                        response["llm_prompt"] = self._last_request
                        response["llm_tags"] = self._last_llm_tags

                        # load and decode the image
                        image: Image.Image = Image.open(BytesIO(b64decode(response["images"][0])))
                        response.pop("images")  # don't need this anymore
                        image.format = "PNG"  # unsure why this is necessary but whatever

                        # decode the info dict and attach it to the response
                        info_dict = json.loads(response.pop("info"))

                        # get the pnginfo string and glue the LLM prompt onto it
                        infotext = info_dict.pop("infotexts", None)
                        if infotext is not None:
                            infotext = infotext[0] if isinstance(infotext, list) else infotext
                            infotext = f'{infotext}, LLM prompt: "{self._last_request}"'
                        else:
                            infotext = f"LLM prompt: {self._last_request}"

                        # reattach the infotext to the info dict, reattach info to the response
                        info_dict.update({"infotexts": [infotext]})
                        response["info"] = info_dict

                        # make a pnginfo object for attaching to the image
                        pnginfo = PngInfo()
                        pnginfo.add_text("parameters", infotext)

                        # work out the path to save the image to, then save it and the job info
                        imagefile_path = IMAGES_DIR.joinpath(
                            f'{response["info"]["job_timestamp"]}_{response["info"]["seed"]}_{req_string}.png'
                        )
                        image.save(imagefile_path, format="PNG", pnginfo=pnginfo)

                        # save the job info
                        imagefile_path.with_suffix(".json").write_text(
                            json.dumps(
                                {"request": request, "response": response},
                                indent=2,
                                skipkeys=True,
                                default=str,
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
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

    def find_image_desc(self, message: str) -> tuple[str | Any, str] | None:
        for regex in send_pic_regexes:
            matches = regex.search(message)
            if matches is not None:
                logger.debug("Message matched image description regex")
                return matches.group(1), regex.sub("", message).strip()
        return None, message

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
