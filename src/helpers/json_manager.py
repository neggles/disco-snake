import json

from disco_snake.cli import DATADIR_PATH


def add_user_to_blacklist(user_id: int) -> None:
    """
    This function will add a user based on its ID in the blacklist.json file.
    :param user_id: The ID of the user that should be added into the blacklist.json file.
    """
    with DATADIR_PATH.joinpath("blacklist.json").open("r+") as file:
        file_data = json.load(file)
        file_data["ids"].append(user_id)
    with DATADIR_PATH.joinpath("blacklist.json").open("w") as file:
        file.seek(0)
        json.dump(file_data, file, indent=4)


def remove_user_from_blacklist(user_id: int) -> None:
    """
    This function will remove a user based on its ID from the blacklist.json file.
    :param user_id: The ID of the user that should be removed from the blacklist.json file.
    """
    with DATADIR_PATH.joinpath("blacklist.json").open("r") as file:
        file_data = json.load(file)
        file_data["ids"].remove(user_id)
    with DATADIR_PATH.joinpath("blacklist.json").open("w") as file:
        file.seek(0)
        json.dump(file_data, file, indent=4)
