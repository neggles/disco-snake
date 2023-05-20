import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from pydantic import BaseSettings, Field, PostgresDsn, validator
from pydantic.env_settings import SettingsSourceCallable

from disco_snake import CONFIG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_settings(settings: BaseSettings) -> Dict[str, Any]:
    encoding = settings.__config__.env_file_encoding
    config_paths = [
        CONFIG_PATH,
        Path.cwd().joinpath("config.json"),
    ]
    for path in config_paths:
        if path.exists() and path.is_file():
            logger.info(f"Loading JSON config from {path}")
            return json.loads(path.read_text(encoding=encoding))
    logger.warning("No JSON config found")
    return {}


class Settings(BaseSettings):
    app_id: int = Field(...)
    bot_token: str = Field(...)
    permissions: int = Field(1642253970515)

    timezone: ZoneInfo = Field("UTC")
    owner: str = Field("N/A")
    repo_url: str = Field("N/A")

    owner_ids: List[int] = Field(..., unique_items=True)
    home_guild: int

    status_type: str = Field("playing")
    statuses: List[str] = Field(["with your heart"])
    log_level: str = Field("INFO")
    debug: bool = Field(False)
    reload: bool = Field(False)

    db_uri: PostgresDsn = Field(..., env="DB_URI")
    disable_cogs: List[str] = Field([])

    @validator("timezone", pre=True, always=True)
    def validate_timezone(cls, v) -> ZoneInfo:
        return ZoneInfo(v)

    class Config:
        env_file_encoding = "utf-8"
        json_encoders = {
            ZoneInfo: lambda v: str(v),
        }

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                env_settings,
                json_settings,
                file_secret_settings,
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
