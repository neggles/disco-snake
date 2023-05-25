import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseConfig, BaseSettings, Field, PostgresDsn, validator
from pydantic.env_settings import SettingsSourceCallable

from disco_snake import CONFIG_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_settings(settings: BaseSettings) -> Dict[str, Any]:
    encoding = settings.__config__.env_file_encoding
    config_path: Path = settings.__config__.json_config_path
    classname = settings.__class__.__name__

    if config_path.exists() and config_path.is_file():
        logger.info(f"Loading {classname} from JSON: {config_path}")
        return json.loads(config_path.read_text(encoding=encoding))
    logger.warning(f"No {classname} JSON found at {config_path}")
    return {}


class JsonConfig(BaseConfig):
    json_config_path = Path.cwd().joinpath("config.json")
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
    ) -> Tuple[SettingsSourceCallable, ...]:
        return (
            env_settings,
            json_settings,
            file_secret_settings,
        )


class Settings(BaseSettings):
    app_id: int = Field(...)
    bot_token: str = Field(...)
    permissions: int = Field(1642253970515)

    timezone: ZoneInfo = Field("UTC")
    owner: str = Field("N/A")
    repo_url: str = Field("N/A")

    owner_id: int = Field(...)
    admin_ids: List[int] = Field(..., unique_items=True)
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

    class Config(JsonConfig):
        json_config_path = CONFIG_PATH


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    return settings
