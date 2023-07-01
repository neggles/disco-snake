import logging
from enum import Enum
from typing import Iterator

from pydantic import BaseModel, BaseSettings, Field

from disco_snake.settings import DEF_DATA_PATH, JsonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigInfo(BaseModel):
    name: str = Field(...)
    config: str = Field(...)


class ConfigList(BaseModel):
    __root__: list[ConfigInfo] = []

    def __iter__(self) -> Iterator[ConfigInfo]:
        return iter(self.__root__)

    def __getitem__(self, key) -> ConfigInfo:
        return self.__root__[key]

    def get_enum(self) -> Enum:
        """Create an enum from the list of settings"""
        # make a dict of config names to config values, backwards Because Click
        configs = {info.config: info.name for info in self.__root__}
        # add an "all" option to the enum
        configs.update({"all": "All"})
        return Enum(
            value="ConfigName",
            names=configs,
            module=__name__,
            qualname="ConfigName",
        )


class MultisnakeSettings(BaseSettings):
    configs: ConfigList = Field(...)

    class Config(JsonConfig):
        json_config_path = DEF_DATA_PATH.joinpath("multisnake.json")


settings: MultisnakeSettings = MultisnakeSettings()
ConfigName: Enum = settings.configs.get_enum()
