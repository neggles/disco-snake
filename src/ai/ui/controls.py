import logging
from typing import TYPE_CHECKING, Callable

from disnake import Colour, Embed, Member, OptionChoice, User
from pydantic import BaseModel

from shimeji.model_provider import OobaGenRequest

if TYPE_CHECKING:
    from ai import Ai

logger = logging.getLogger(__name__)
PURPLE = Colour(0x9966FF)


class AiParam(BaseModel):
    name: str
    id: str
    kind: Callable


settable_params: list[AiParam] = [
    AiParam(name="Temperature", id="temperature", kind=float),
    AiParam(name="Top P", id="top_p", kind=float),
    AiParam(name="Top K", id="top_k", kind=float),
    AiParam(name="Min P", id="min_p", kind=float),
    AiParam(name="Rep Penalty", id="repetition_penalty", kind=float),
    AiParam(name="Rep Penalty Range", id="repetition_penalty_range", kind=int),
    AiParam(name="Min Length", id="min_length", kind=int),
    AiParam(name="Max Length", id="max_tokens", kind=int),
    AiParam(name="Eta C", id="eta_cutoff", kind=float),
    AiParam(name="Eps C", id="epsilon_cutoff", kind=float),
    AiParam(name="Apply Temp Last", id="temperature_last", kind=bool),
]

set_choices = [OptionChoice(name=param.name, value=param.id) for param in settable_params]


class AiStatusEmbed(Embed):
    def __init__(self, cog: "Ai", user: User | Member, *args, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        gensettings: OobaGenRequest = cog.provider_config.gensettings

        vision_enabled = cog.eyes.enabled if cog.eyes is not None else False
        vision_model = cog.eyes.api_info.model_type if vision_enabled else "N/A"

        imagen_enabled = cog.imagen.enabled if cog.imagen is not None else False
        imagen_model = cog.imagen.config.api_params.checkpoint if imagen_enabled else "N/A"

        self.description = self.description if self.description is not None else "**AI Status**"
        self.colour = PURPLE
        self.set_author(name=cog.bot.user.name, icon_url=cog.bot.user.avatar.url)

        if verbose is True:
            self.add_field(name="Message Context", value=cog.params.context_messages)
            self.add_field(name="Vision Enabled", value=vision_enabled)
            if vision_enabled:
                self.add_field(name="Vision Model", value=vision_model)
            self.add_field(name="Imagen Enabled", value=imagen_enabled)
            if imagen_enabled:
                self.add_field(name="Imagen Model", value=imagen_model)

        self.add_field(name="Temperature", value=gensettings.temperature)
        self.add_field(name="Temperature Last", value=gensettings.temperature_last)
        self.add_field(name="Top P", value=gensettings.top_p)
        self.add_field(name="Top K", value=gensettings.top_k)
        self.add_field(name="Min P", value=gensettings.min_p)

        self.add_field(name="Rep Penalty", value=gensettings.repetition_penalty)
        self.add_field(name="Rep Pen. Range", value=gensettings.repetition_penalty_range)

        if gensettings.eta_cutoff > 0 or gensettings.epsilon_cutoff > 0:
            self.add_field(name="Eta Cutoff", value=gensettings.eta_cutoff)
            self.add_field(name="Eps Cutoff", value=gensettings.epsilon_cutoff)

        self.add_field(name="Max Length", value=gensettings.max_tokens)
        if verbose is True:
            self.add_field(name="Min Length", value=gensettings.min_length)
            if gensettings.penalty_alpha > 0:
                self.add_field(name="Penalty Alpha", value=gensettings.penalty_alpha)
                self.add_field(name="Num Beams", value=gensettings.num_beams)
                self.add_field(name="Length Penalty", value=gensettings.length_penalty)
                self.add_field(name="Early Stopping", value=gensettings.early_stopping)
            self.add_field(name="Add BOS Token", value=gensettings.add_bos_token)
            self.add_field(name="Ban EOS Token", value=gensettings.ban_eos_token)
            self.add_field(name="Skip Special Tokens", value=gensettings.ban_eos_token)
            self.add_field(name="Truncation Length", value=gensettings.truncation_length)

        self.set_footer(text=f"Requested by {user.name}", icon_url=user.avatar.url)
