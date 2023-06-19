from datetime import datetime
from typing import Annotated, Optional

import disnake
import sqlalchemy as sa
from disnake.ext import commands
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym_for

from db.base import Base, CreateTimestamp, Timestamp, UpdateTimestamp
from db.discord import DiscordName, DiscordSnowflake, DiscordUser


class DiscordChatLog(Base):
    __tablename__ = "discord_logs"
