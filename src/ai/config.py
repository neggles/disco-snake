from dataclasses import dataclass
from typing import List


@dataclass
class ModelProviderConfig:
    endpoint: str
    username: str
    password: str
    gensettings: dict


# configuration dataclasses
@dataclass
class MemoryStoreConfig:
    database_uri: str
    model: str
    model_layer: int
    short_term_amount: int
    long_term_amount: int


@dataclass
class BotParameters:
    conditional_response: bool
    idle_messaging: bool
    idle_messaging_interval: int
    nicknames: List[str]
    context_size: int
    logging_channel_id: int
    activity_channels: List[int]
    debug: bool
    memory_enable: bool


@dataclass
class ChatbotConfig:
    name: str
    prompt: str
    params: BotParameters
    memory_store: MemoryStoreConfig
    model_provider: ModelProviderConfig
