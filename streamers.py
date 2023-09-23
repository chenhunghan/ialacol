import json
from os import times
from ctransformers import LLM, Config

from log import log
from get_env import get_env
from const import DEFAULT_CONTEXT_LENGTH, DEFAULT_LOG_LEVEL


def completions_streamer(
    prompt: str,
    model_name: str,
    llm: LLM,
    config: Config,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    top_k = config.top_k
    log.debug("top_k: %s", top_k)
    top_p = config.top_p
    log.debug("top_p: %s", top_p)
    temperature = config.temperature
    log.debug("temperature: %s", temperature)
    repetition_penalty = config.repetition_penalty
    log.debug("repetition_penalty: %s", repetition_penalty)
    last_n_tokens = config.last_n_tokens
    log.debug("last_n_tokens: %s", last_n_tokens)
    seed = config.seed
    log.debug("seed: %s", seed)
    batch_size = config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = config.threads
    log.debug("threads: %s", threads)
    max_new_tokens = config.max_new_tokens
    log.debug("max_new_tokens: %s", max_new_tokens)
    stop = config.stop
    log.debug("stop: %s", stop)
    log.debug("prompt: %s", prompt)
    CONTEXT_LENGTH = int(get_env("CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
    LOGGING_LEVEL = get_env("LOGGING_LEVEL", DEFAULT_LOG_LEVEL)

    log.debug("Streaming from ctransformer instance!")
    total_tokens = 0
    for token in llm(
        prompt,
        stream=True,
        reset=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        max_new_tokens=max_new_tokens,
        stop=stop,
    ):
        if LOGGING_LEVEL == "DEBUG":
            # Only track token length if we're in debug mode to avoid overhead
            total_tokens = total_tokens + len(token)
            # tokens are not necessarily characters, but this is a good enough approximation
            if total_tokens > CONTEXT_LENGTH:
                log.debug(
                    "Total token length %s exceeded context length %s",
                    total_tokens,
                    CONTEXT_LENGTH,
                )
                log.debug(
                    "Try to increase CONTEXT_LENGTH that is currently set to %s to your model's context length",
                    CONTEXT_LENGTH,
                )
                log.debug(
                    "Alternatively, increse REPETITION_PENALTY %s and LAST_N_TOKENS %s AND/OR adjust temperature %s top_k %s top_p %s",
                    repetition_penalty,
                    last_n_tokens,
                    temperature,
                    top_k,
                    top_p,
                )
        log.debug("Streaming token %s", token)
        data = json.dumps(
            {
                "id": "id",
                "object": "text_completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": token,
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
        )
        yield f"data: {data}" + "\n\n"

    stop_data = json.dumps(
        {
            "id": "id",
            "object": "text_completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield f"data: {stop_data}" + "\n\n"
    log.debug("Streaming ended")


def chat_completions_streamer(
    prompt: str,
    model_name: str,
    llm: LLM,
    config: Config,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    top_k = config.top_k
    log.debug("top_k: %s", top_k)
    top_p = config.top_p
    log.debug("top_p: %s", top_p)
    temperature = config.temperature
    log.debug("temperature: %s", temperature)
    repetition_penalty = config.repetition_penalty
    log.debug("repetition_penalty: %s", repetition_penalty)
    last_n_tokens = config.last_n_tokens
    log.debug("last_n_tokens: %s", last_n_tokens)
    seed = config.seed
    log.debug("seed: %s", seed)
    batch_size = config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = config.threads
    log.debug("threads: %s", threads)
    max_new_tokens = config.max_new_tokens
    log.debug("max_new_tokens: %s", max_new_tokens)
    stop = config.stop
    log.debug("stop: %s", stop)
    log.debug("prompt: %s", prompt)
    CONTEXT_LENGTH = int(get_env("CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
    LOGGING_LEVEL = get_env("LOGGING_LEVEL", DEFAULT_LOG_LEVEL)

    log.debug("Streaming from ctransformer instance")
    total_tokens = 0
    for token in llm(
        prompt,
        stream=True,
        reset=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        max_new_tokens=max_new_tokens,
        stop=stop,
    ):
        if LOGGING_LEVEL == "DEBUG":
            # Only track token length if we're in debug mode to avoid overhead
            total_tokens = total_tokens + len(token)
            # tokens are not necessarily characters, but this is a good enough approximation
            if total_tokens > CONTEXT_LENGTH:
                log.debug(
                    "Total token length %s exceeded context length %s",
                    total_tokens,
                    CONTEXT_LENGTH,
                )
                log.debug(
                    "Try to increase CONTEXT_LENGTH that is currently set to %s to your model's context length",
                    CONTEXT_LENGTH,
                )
                log.debug(
                    "Alternatively, increse REPETITION_PENALTY %s and LAST_N_TOKENS %s AND/OR adjust temperature %s top_k %s top_p %s",
                    repetition_penalty,
                    last_n_tokens,
                    temperature,
                    top_k,
                    top_p,
                )
        log.debug("Streaming token %s", token)
        data = json.dumps(
            {
                "id": "id",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": token},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
        )
        yield f"data: {data}" + "\n\n"

    stop_data = json.dumps(
        {
            "id": "id",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield f"data: {stop_data}" + "\n\n"
    log.debug("Streaming ended")
