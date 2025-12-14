import logging
from collections.abc import Mapping
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import Field, PostgresDsn
from pydantic_extra_types.timezone_name import TimeZoneName, timezone_name_settings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import JsonConfigSettingsSource, PathType, PydanticBaseSettingsSource

from disco_snake import DEF_DATA_PATH, per_config_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timezone_name_settings(strict=False)
class TZNonStrict(TimeZoneName):
    pass


def update_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class MultiJsonConfigSettingsSource(JsonConfigSettingsSource):
    def _read_files(self, files: PathType | None) -> dict[str, Any]:
        if files is None:
            return {}
        if isinstance(files, (str, PathLike)):
            files = [files]
        vars: dict[str, Any] = {}
        for file in files:
            file_path = Path(file).expanduser()
            if file_path.is_file():
                update_recursive(vars, self._read_file(file_path))
        return vars


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
            MultiJsonConfigSettingsSource(settings_cls),
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
    ai_conf_name: str | None = None

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
def get_settings(config_path: Path | None = None) -> BotSettings:
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
