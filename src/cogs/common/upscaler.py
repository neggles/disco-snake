import asyncio
from enum import Enum
from io import BytesIO, FileIO
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Union

import replicate
import requests
from PIL import Image
from disco_snake import DATADIR_PATH, LOGDIR_PATH

UPSCALER_MODEL = "jingyunliang/swinir"
UPSCALER_VERSION = "660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"

MOD_DATADIR = DATADIR_PATH.joinpath("sd", "upscaler")


class UpscaleType(Enum):
    MEDIUM = "Real-World Image Super-Resolution-Medium"
    LARGE = "Real-World Image Super-Resolution-Large"


def download_file(url: str) -> BytesIO:
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)


class Upscaler:
    def __init__(self, token: str, logger: Logger):
        self.client = replicate.Client(api_token=token)
        self.upscaler = self.client.models.get(UPSCALER_MODEL).versions.get(UPSCALER_VERSION)
        self.logger = logger

    def _download(url: str) -> BytesIO:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    def upscale(
        self, url: str, type: UpscaleType = UpscaleType.LARGE, download: bool = False
    ) -> Union[FileIO, str]:
        self.logger.info(f"Upscaling {url}...")

        prediction = self.client.predictions.create(
            version=self.upscaler, input={"image": url, "task_type": type.value}
        )
        self.logger.info("Prediction submitted, waiting for it to finish...")
        prediction.wait()

        if prediction.status in ["failed", "canceled"]:
            failure = f"Upscaling returned result '{prediction.status}' :("
            self.logger.error(failure)
            raise RuntimeError(failure)
        self.logger.info(f"Upscaling successful, result URL = {prediction.output}")

        if download is True:
            self.logger.info(f"Retrieving {prediction.output.split('/')[-1]}...")
            ret = self._download(prediction.output)
        else:
            ret = prediction.output

        return ret
