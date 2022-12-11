from random import choice as random_choice
from typing import Union, List

from disnake import Colour, Embed, File, Member, User
from humanize import naturaldelta as fuzzydelta

from disco_snake import DATADIR_PATH

COOLDOWN_LINES = [
    "slow down, cowboy!",
    "2 fast 2 furious",
    "Easy, tiger",
    "Hold your horses, cowboy!",
    "Take it down a notch",
    "Pace yourself, champ",
    "Settle down, cowboy",
    "Not so fast, my friend",
    "Whoa, whoa, whoa, not so fast!",
    "Take a deep breath, comrade",
    "Chill out, my dude",
    "Take it easy, friend",
    "Simmer down, soldier",
    "Cool your jets, Maverick",
    "Don't get ahead of yourself, okay?",
    "Ease up on the throttle",
    "Wait just a minute, pardner",
    "Keep your shirt on, pal",
    "Unrustle your jimmies, friendo",
]


def cooldown_msg() -> str:
    return random_choice(COOLDOWN_LINES)


class CooldownEmbed(Embed):
    def __init__(self, cooldown: int, target: Union[User, Member, None] = None, *args, **kwargs):
        kwargs.update(
            title=cooldown_msg(),
            colour=target.colour if isinstance(target, Member) else Colour(0xFF6600),
            description=f"Please wait {fuzzydelta(cooldown)} to use this command again",
        )
        super().__init__(*args, **kwargs)
        if target is not None:
            self.set_footer(text=f"Requested by {target.display_name}", icon_url=target.display_avatar.url)


DENIED_GIF_PATH = DATADIR_PATH.joinpath("misc", "magic-word-2.gif")
DENIED_GIF = File(DENIED_GIF_PATH, filename=DENIED_GIF_PATH.name)


class PermissionEmbed(Embed):
    def __init__(self, author: Union[User, Member] = None, permissions: List[str] = None):
        super().__init__(
            title="ah ah ah!",
            description="you didn't say the magic word!",
            color=0xE02B2B,
        )
        self.set_image(file=DENIED_GIF)

        if author is not None:
            self.set_footer(text=f"Requested by {author.display_name}", icon_url=author.display_avatar.url)
        if permissions is not None:
            self.add_field(name="Required permissions", value=", ".join(permissions))
