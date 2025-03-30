import logging
from enum import Enum
from typing import Annotated, Iterator

from pydantic import BaseModel, Field, RootModel
from pydantic_settings import SettingsConfigDict

from disco_snake.settings import DEF_DATA_PATH, JsonSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigInfo(BaseModel):
    name: str = Field(...)
    config: str = Field(...)


class ConfigList(RootModel):
    root: list[ConfigInfo] = []

    def __iter__(self) -> Iterator[ConfigInfo]:  # type: ignore
        return self.root.__iter__()

    def __getitem__(self, key) -> ConfigInfo:
        return self.root.__getitem__(key)

    def get_enum(self) -> type[Enum]:
        """Create an enum from the list of settings"""
        # make a dict of config names to config values, backwards Because Click
        configs = {info.config: info.name for info in self.root}
        # add an "all" option to the enum
        configs.update({"all": "All"})
        return Enum(
            value="ConfigName",
            names=configs,
            module=__name__,
            qualname="ConfigName",
        )


class MultisnakeSettings(JsonSettings):
    configs: Annotated[ConfigList, Field(default_factory=ConfigList)]

    model_config = SettingsConfigDict(
        json_file=DEF_DATA_PATH.joinpath("multisnake.json"),
    )


settings: MultisnakeSettings = MultisnakeSettings()  # type: ignore
ConfigName: type[Enum] = settings.configs.get_enum()
