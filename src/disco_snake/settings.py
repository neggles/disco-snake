import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional
from zoneinfo import ZoneInfo

from pydantic import Field, PostgresDsn
from pydantic_extra_types.timezone_name import TimeZoneName, timezone_name_settings
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from disco_snake import DEF_DATA_PATH, per_config_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timezone_name_settings(strict=False)
class TZNonStrict(TimeZoneName):
    pass


class JsonSettings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            JsonConfigSettingsSource(settings_cls),
            init_settings,
            env_settings,
            dotenv_settings,
        )


class BotSettings(JsonSettings):
    app_id: int
    bot_token: str
    permissions: int = Field(1642253970515)

    timezone: Annotated[ZoneInfo, Field("UTC")]
    owner: str = Field("N/A")
    repo_url: str = Field("N/A")
    db_uri: PostgresDsn = Field()
    ai_conf_name: Optional[str] = Field(None)

    owner_id: int
    admin_ids: Annotated[list[int], Field()]
    retcon_ids: Annotated[list[int], Field([])]
    home_guild: int
    support_guild: int
    support_channel: int

    status_type: str = Field("playing")
    statuses: list[str] = Field(["with your heart"])
    log_level: str = Field("INFO")
    debug: bool = Field(False)
    reload: bool = Field(False)

    disable_cogs: list[str] = Field([])

    model_config = SettingsConfigDict(
        json_file=[
            DEF_DATA_PATH.joinpath("config.json"),
            DEF_DATA_PATH.joinpath(per_config_name("config.json")),
        ],
    )


@lru_cache(maxsize=2)
def get_settings(config_path: Optional[Path] = None) -> BotSettings:
    if config_path is not None:
        if config_path.is_file():
            settings = BotSettings.model_validate_json(config_path.read_text())
        elif config_path.is_dir():
            settings = BotSettings.model_validate_json(config_path.joinpath("imagen.json").read_text())
        else:
            raise ValueError(f"Invalid config path: {config_path}")
    else:
        settings = BotSettings()  # type: ignore
    return settings
