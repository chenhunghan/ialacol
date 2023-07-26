import json
from logging import Logger
from os import times

from ctransformers import LLM, Config


def completions_streamer(
    prompt: str,
    model_name: str,
    llm: LLM,
    config: Config,
    log: Logger,
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
    max_new_tokens = config.max_new_tokens
    log.debug("max_new_tokens: %s", max_new_tokens)
    stop = config.stop
    log.debug("stop: %s", stop)
    batch_size = config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = config.threads
    log.debug("thread: %s", threads)
    log.debug("prompt: %s", prompt)

    log.debug("Streaming from ctransformer instance!")
    for token in llm(
        prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        stop=stop,
        batch_size=batch_size,
        threads=threads,
        stream=True,
        reset=True,
        max_new_tokens=max_new_tokens,
    ):
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
    log: Logger,
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
    stop = config.stop
    log.debug("stop: %s", stop)
    batch_size = config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = config.threads
    log.debug("threads: %s", threads)
    log.debug("prompt: %s", prompt)

    log.debug("Streaming from ctransformer instance")
    for token in llm(
        prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        stop=stop,
        batch_size=batch_size,
        threads=threads,
        stream=True,
        reset=True,
    ):
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
