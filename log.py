import logging

from get_env import get_env
from const import DEFAULT_LOG_LEVEL

LOGGING_LEVEL = get_env("LOGGING_LEVEL", DEFAULT_LOG_LEVEL)

log = logging.getLogger("uvicorn")
try:
    log.setLevel(LOGGING_LEVEL)
except ValueError:
    log.setLevel(DEFAULT_LOG_LEVEL)
