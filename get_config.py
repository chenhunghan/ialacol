import logging
from ctransformers import Config

from request_body import ChatCompletionRequestBody, CompletionRequestBody
from get_env import get_env
from get_default_thread import get_default_thread

LOGGING_LEVEL = get_env("LOGGING_LEVEL", "INFO")

log = logging.getLogger(__name__)
try:
    log.setLevel(LOGGING_LEVEL)
except ValueError:
    log.setLevel("INFO")

THREADS = int(get_env("THREADS", str(get_default_thread())))


def get_config(body: CompletionRequestBody | ChatCompletionRequestBody) -> Config:
    # ggml only, follow ctransformers defaults
    TOP_K = int(get_env("TOP_K", "40"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
    TOP_P = float(get_env("TOP_P", "1.0"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
    TEMPERATURE = float(get_env("TEMPERATURE", "1"))
    # ggml only, follow ctransformers defaults
    REPETITION_PENALTY = float(get_env("REPETITION_PENALTY", "1.1"))
    # ggml only, follow ctransformers defaults
    LAST_N_TOKENS = int(get_env("LAST_N_TOKENS", "64"))
    # ggml only, follow ctransformers defaults
    SEED = int(get_env("SEED", "-1"))
    # ggml only, follow ctransformers defaults
    BATCH_SIZE = int(get_env("BATCH_SIZE", "8"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
    MAX_TOKENS = int(get_env("MAX_TOKENS", "9999999"))
    # ggml only, follow ctransformers defaults
    CONTEXT_LENGTH = int(get_env("CONTEXT_LENGTH", "-1"))

    log.info("BATCH_SIZE: %s", BATCH_SIZE)
    log.info("CONTEXT_LENGTH: %s", CONTEXT_LENGTH)
    log.info("LAST_N_TOKENS: %s", LAST_N_TOKENS)
    log.info("MAX_TOKENS: %s", MAX_TOKENS)

    config = Config(
        top_k=body.top_k if body.top_k else TOP_K,
        top_p=body.top_p if body.top_p else TOP_P,
        temperature=body.temperature if body.temperature else TEMPERATURE,
        repetition_penalty=body.repetition_penalty
        if body.repetition_penalty
        else REPETITION_PENALTY,
        last_n_tokens=LAST_N_TOKENS,
        seed=SEED,
        batch_size=BATCH_SIZE,
        threads=THREADS,
        max_new_tokens=body.max_tokens if body.max_tokens else MAX_TOKENS,
        stop=body.stop,
        context_length=CONTEXT_LENGTH,
    )
    return config
