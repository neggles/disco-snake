from disnake.ext.commands import CommandError


class UserBlacklisted(CommandError):
    def __init__(self, message: str = "You have been blacklisted and cannot interact with this bot.") -> None:
        self.message = message
        super().__init__(self.message)


class UserNotAdmin(CommandError):
    def __init__(self, message: str = "Did you really think that would work?") -> None:
        self.message = message
        super().__init__(self.message)


class UserNotOwner(UserNotAdmin):
    def __init__(self, message: str = "Did you really think that would work?") -> None:
        self.message = message
        super().__init__(self.message)
