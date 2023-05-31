import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, BaseSettings, Field, validator

from disco_snake import DATADIR_PATH, LOG_FORMAT, LOGDIR_PATH
from disco_snake.settings import JsonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TG_DATA_DIR = DATADIR_PATH.joinpath("telegram")
TG_CONFIG_PATH = DATADIR_PATH.joinpath("telegram.json")


class RedisSettings(BaseModel):
    host: str = Field(...)
    port: int = Field(...)
    db: int = Field(...)
    prefix: str = Field("aiogram")
    pool_size: int = Field(10)


class ClientSettings(BaseModel):
    api_id: int = Field(...)
    api_hash: str = Field(...)
    phone: str = Field(...)
    database_encryption_key: str = Field(...)
    files_directory: Path = Field("files")

    @validator("files_directory", pre=True, always=True)
    def _files_directory(cls, v: Union[str, Path]) -> Path:
        return TG_DATA_DIR.joinpath(v).absolute()


class TelegramChat(BaseModel):
    id: int = Field(...)
    name: Optional[str] = Field(None)
    enable: bool = Field(False)


class TelegramSettings(BaseSettings):
    # Project config
    token: str = Field(...)
    debug: bool = Field(False)
    client: ClientSettings = Field(...)
    redis: RedisSettings = Field(...)
    chats: List[TelegramChat] = Field(...)

    class Config(JsonConfig):
        json_config_path = TG_CONFIG_PATH

    @property
    def enabled_chats(self):
        return [x.id for x in self.chats if x.enable is True]


@lru_cache(maxsize=1)
def get_tg_settings() -> TelegramSettings:
    settings = TelegramSettings(json_config_path=TG_CONFIG_PATH)
    return settings
