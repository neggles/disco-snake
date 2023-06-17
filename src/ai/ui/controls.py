import logging
from lib2to3.fixes.fix_idioms import TYPE
from typing import TYPE_CHECKING

from disnake import (
    ApplicationCommandInteraction,
    Colour,
    Embed,
    Member,
    User,
)
from disnake.ui import Modal, StringSelect, TextInput, View
from shimeji.model_provider import OobaGenRequest

if TYPE_CHECKING:
    from ai import Ai

logger = logging.getLogger(__name__)
PURPLE = Colour(0x9966FF)


class AiStatusEmbed(Embed):
    def __init__(self, cog: "Ai", user: User | Member, *args, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        gensettings: OobaGenRequest = cog.provider_config.gensettings

        vision_model = cog.eyes.config.modeltype if cog.eyes is not None else "N/A"
        imagen_model = cog.imagen.config.api_params.checkpoint if cog.imagen is not None else "N/A"

        self.description = "**AI Engine Status**" if not self.description else self.description
        self.colour = PURPLE
        self.set_author(name=cog.name, icon_url=cog.bot.user.avatar.url)

        if verbose is True:
            self.add_field(name="Provider", value=cog.model_provider.__class__.__name__)
            self.add_field(name="Context Tokens", value=cog.params.context_size)
            self.add_field(name="Max Messages", value=cog.params.context_messages)
            self.add_field(name="Autoresponse", value=cog.params.autoresponse)
            self.add_field(name="Idle Messaging", value=cog.params.idle_enable)
            self.add_field(name="Vision Enabled", value=(cog.eyes is not None))
            self.add_field(name="Vision Model", value=vision_model)
            self.add_field(name="Imagen Enabled", value=(cog.imagen is not None))
            self.add_field(name="Imagen Model", value=imagen_model)

        self.add_field(name="Temperature", value=gensettings.temperature)
        self.add_field(name="Top P", value=gensettings.top_p)
        self.add_field(name="Top K", value=gensettings.top_k)
        self.add_field(name="Typical P", value=gensettings.typical_p)
        self.add_field(name="Rep. Penalty", value=gensettings.repetition_penalty)
        self.add_field(name="Eta Cutoff", value=gensettings.eta_cutoff)
        self.add_field(name="Epsilon Cutoff", value=gensettings.epsilon_cutoff)

        if verbose is True:
            self.add_field(name="Min Tokens", value=gensettings.min_length)
            self.add_field(name="Max Tokens", value=gensettings.max_new_tokens)
            self.add_field(name="Encoder Rep. P", value=gensettings.encoder_repetition_penalty)
            self.add_field(name="No Repeat Ngram", value=gensettings.no_repeat_ngram_size)
            self.add_field(name="Penalty Alpha", value=gensettings.penalty_alpha)
            self.add_field(name="Num Beams", value=gensettings.num_beams)
            self.add_field(name="Length Penalty", value=gensettings.length_penalty)
            self.add_field(name="Early Stopping", value=gensettings.early_stopping)

            self.add_field(name="Add BOS Token", value=gensettings.add_bos_token)
            self.add_field(name="Ban EOS Token", value=gensettings.ban_eos_token)
            self.add_field(name="Skip Special Tokens", value=gensettings.ban_eos_token)
            self.add_field(name="Truncation Length", value=gensettings.truncation_length)

        self.set_footer(text=f"Requested by {user.name}", icon_url=user.avatar.url)
