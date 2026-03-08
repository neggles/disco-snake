from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import SettingsConfigDict

from disco_snake.settings import JsonSettings
from rattlesnake import RATTLESNAKE_CONFIG_PATH


class RattlesnakeSettings(JsonSettings):
    db_uri: PostgresDsn
    log_level: str = Field("INFO")
    debug: bool

    model_config = SettingsConfigDict(
        json_file=RATTLESNAKE_CONFIG_PATH,
        json_file_encoding="utf-8",
        nested_model_default_partial_update=True,
        extra="forbid",
    )


@lru_cache(maxsize=1)
def get_settings() -> RattlesnakeSettings:
    return RattlesnakeSettings()  # type: ignore
