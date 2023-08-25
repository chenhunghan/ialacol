import logging

from get_env import get_env


LOGGING_LEVEL = get_env("LOGGING_LEVEL", "INFO")

log = logging.getLogger("uvicorn")
try:
    log.setLevel(LOGGING_LEVEL)
except ValueError:
    log.setLevel("INFO")
