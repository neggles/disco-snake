from ai.constants import AI_DATA_DIR

TOOLS_DATA_DIR = AI_DATA_DIR.joinpath("tools")
if not TOOLS_DATA_DIR.is_dir():
    TOOLS_DATA_DIR.mkdir(parents=True, exist_ok=True)
