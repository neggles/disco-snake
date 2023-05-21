from lib2to3.fixes.fix_idioms import TYPE
import logging
from typing import TYPE_CHECKING

from disnake import (
    Colour,
    Embed,
    Member,
    User,
)
from shimeji.model_provider import OobaGenRequest

if TYPE_CHECKING:
    from ai import Ai

logger = logging.getLogger(__name__)
PURPLE = Colour(0x9966FF)


class AiStatusEmbed(Embed):
    def __init__(self, cog: "Ai", user: User | Member, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gensettings: OobaGenRequest = cog.model_provider_cfg.gensettings

        vision_model = cog.eyes.config.model_name if cog.eyes is not None else "N/A"
        imagen_model = cog.imagen.config.api_params.checkpoint if cog.imagen is not None else "N/A"

        self.description = "**AI Engine Status**" if not self.description else self.description
        self.colour = PURPLE
        self.set_author(name=cog.name, icon_url=cog.bot.user.avatar.url)

        self.add_field(name="Provider", value=cog.model_provider.__class__.__name__)
        self.add_field(name="Context Length", value=cog.params.context_size, inline=True)
        self.add_field(name="Context Messages", value=cog.params.context_messages, inline=True)
        self.add_field(name="Self-trigger", value=cog.params.conditional_response, inline=False)
        self.add_field(name="Idle Messaging", value=cog.params.idle_messaging, inline=True)
        self.add_field(name="Idle Messaging Interval", value=cog.params.idle_messaging_interval, inline=True)

        self.add_field(name="Vision Enabled", value=(cog.eyes is not None), inline=False)
        self.add_field(name="Vision Model", value=vision_model, inline=True)

        self.add_field(name="Imagen Enabled", value=(cog.imagen is not None), inline=False)
        self.add_field(name="Imagen Model", value=imagen_model, inline=True)

        self.add_field(name="Temperature", value=gensettings.temperature, inline=True)
        self.add_field(name="Top P", value=gensettings.top_p, inline=True)
        self.add_field(name="Top K", value=gensettings.top_k, inline=True)
        self.add_field(name="Typical P", value=gensettings.typical_p, inline=True)
        self.add_field(name="Rep P", value=gensettings.repetition_penalty, inline=True)
        self.add_field(name="Enc. Rep P", value=gensettings.encoder_repetition_penalty, inline=True)
        self.add_field(name="Min Length", value=gensettings.min_length, inline=True)
        self.add_field(name="Max Length", value=gensettings.max_new_tokens, inline=True)
        self.add_field(name="No Repeat Size", value=gensettings.no_repeat_ngram_size, inline=True)

        self.add_field(name="Penalty Alpha", value=gensettings.penalty_alpha, inline=True)
        self.add_field(name="Num Beams", value=gensettings.num_beams, inline=True)
        self.add_field(name="Length Penalty", value=gensettings.length_penalty, inline=True)
        self.add_field(name="Early Stopping", value=gensettings.early_stopping, inline=True)

        self.add_field(name="Add BOS Token", value=gensettings.add_bos_token, inline=False)
        self.add_field(name="Ban EOS Token", value=gensettings.ban_eos_token, inline=True)
        self.add_field(name="Skip Special Tokens", value=gensettings.ban_eos_token, inline=True)
        self.add_field(name="Truncation Length", value=gensettings.truncation_length, inline=True)

        self.set_footer(text=f"Requested by {user.name}", icon_url=user.avatar.url)
