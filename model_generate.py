from time import time
from logging import Logger
from ctransformers import LLM, Config


def model_generate(
    prompt: str,
    model_name: str,
    llm: LLM,
    config: Config,
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

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
    batch_size = config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = config.threads
    log.debug("thread: %s", threads)
    log.debug("prompt: %s", prompt)

    log.debug("Getting from ctransformer instance")
    result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        max_new_tokens=max_new_tokens,
    )
    http_response = {
        "id": "id",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": result,
                "logprobs": None,
                "finish_reason": "end_of_token",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    log.debug("http_response:%s ", http_response)
    return http_response


def chat_model_generate(
    prompt: str,
    model_name: str,
    llm: LLM,
    config: Config,
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

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
    log.debug("thread: %s", threads)
    log.debug("prompt: %s", prompt)

    log.debug("Getting from ctransformer instance")
    result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
    )
    http_response = {
        "id": "id",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result,
                },
                "finish_reason": "end_of_token",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    log.debug("http_response:%s ", http_response)
    return http_response
