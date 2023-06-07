import json
import logging
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import AnyUrl, BaseConfig, BaseSettings, Field, PostgresDsn, validator
from pydantic.env_settings import (
    EnvSettingsSource,
    InitSettingsSource,
    SecretsSettingsSource,
    SettingsSourceCallable,
)

from disco_snake import DEF_CONFIG_PATH, get_suffix_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonSettingsSource:
    __slots__ = "json_config_path"

    def __init__(self, json_config_path: Optional[PathLike] = None) -> None:
        self.json_config_path: Optional[Path] = (
            Path(json_config_path) if json_config_path is not None else None
        )

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:  # noqa C901
        classname = settings.__class__.__name__
        encoding = settings.__config__.env_file_encoding
        if self.json_config_path is None:
            pass
        elif self.json_config_path.exists() and self.json_config_path.is_file():
            logger.info(f"Loading {classname} config from path: {self.json_config_path}")
            return json.loads(self.json_config_path.read_text(encoding=encoding))
        logger.warning(f"No {classname} config found at {self.json_config_path}")
        return {}

    def __repr__(self) -> str:
        return f"JsonSettingsSource(json_config_path={self.json_config_path!r})"


class JsonConfig(BaseConfig):
    json_config_path: Optional[Path] = None
    env_file_encoding = "utf-8"

    @classmethod
    def customise_sources(
        cls,
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[SettingsSourceCallable, ...]:
        # pull json_config_path from init_settings if passed, otherwise use the class var
        json_config_path = init_settings.init_kwargs.pop("json_config_path", cls.json_config_path)
        # create a JsonSettingsSource with the json_config_path (which may be None)
        json_settings = JsonSettingsSource(json_config_path=json_config_path)
        # return the new settings sources
        return (
            init_settings,
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
    db_uri: PostgresDsn = Field(..., env="DB_URI")
    ai_conf_name: Optional[str] = Field(None)

    owner_id: int = Field(...)
    admin_ids: List[int] = Field(..., unique_items=True)
    home_guild: int = Field(...)
    support_guild: int = Field(...)
    support_invite: AnyUrl = Field(...)

    status_type: str = Field("playing")
    statuses: List[str] = Field(["with your heart"])
    log_level: str = Field("INFO")
    debug: bool = Field(False)
    reload: bool = Field(False)

    disable_cogs: List[str] = Field([])

    @validator("timezone", pre=True, always=True)
    def validate_timezone(cls, v) -> ZoneInfo:
        return ZoneInfo(v)

    class Config(JsonConfig):
        json_config_path = DEF_CONFIG_PATH


@lru_cache(maxsize=2)
def get_settings(config_path: Optional[Path] = None) -> Settings:
    if config_path is None:
        config_path = DEF_CONFIG_PATH.with_stem(get_suffix_name(DEF_CONFIG_PATH.stem))
    settings = Settings(json_config_path=config_path)
    return settings
