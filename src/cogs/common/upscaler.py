import asyncio
from enum import Enum
from io import BytesIO, FileIO
from logging import Logger

import replicate
import requests
from PIL import Image

UPSCALER_MODEL = "jingyunliang/swinir"
UPSCALER_VERSION = "660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a"


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

    def upscale(self, image_file: str, type: UpscaleType = UpscaleType.LARGE) -> str:
        # with Image.open(image_file) as image:
        #     self.logger.info(f"Upscaling {image.size} image with task {type.name}")

        prediction = self.client.predictions.create(
            version=self.upscaler, input={"image": image_file, "task_type": type.value}
        )
        self.logger.info("Prediction submitted, waiting for it to finish...")
        prediction.wait()

        if prediction.status in ["failed", "canceled"]:
            failure = f"Upscaling returned result '{prediction.status}' :("
            self.logger.error(failure)
            raise RuntimeError(failure)
        self.logger.info(f"Upscaling finished successfully: {prediction.output}")
        return prediction.output


class AsyncUpscaler:
    def __init__(self, token: str, logger: Logger):
        self.client = replicate.Client(api_token=token)
        self.upscaler = self.client.models.get(UPSCALER_MODEL).versions.get(UPSCALER_VERSION)
        self.logger = logger

    async def upscale(self, image_file: FileIO, type: UpscaleType = UpscaleType.LARGE) -> str:
        with Image.open(image_file) as image:
            self.logger.info(f"Upscaling {image.size} image with task {type.name}")

        prediction = self.client.predictions.create(
            version=self.upscaler, input={"image": image_file, "task_type": type.value}
        )

        # wait for prediction to finish
        self.logger.info("Prediction submitted, waiting for it to finish...")
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            await asyncio.sleep(1.0)
            prediction.reload()

        if prediction.status in ["failed", "canceled"]:
            failure = f"Upscaling returned result '{prediction.status}' :("
            self.logger.error(failure)
            raise RuntimeError(failure)
        self.logger.info(f"Upscaling finished successfully : {prediction.output}")
        return prediction.output
