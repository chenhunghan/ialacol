import json
from logging import Logger
from os import times

from ctransformers import LLM
from llm_rs.base_model import Model

from llm_rs.config import (  # pylint: disable=no-name-in-module,import-error
    GenerationConfig,
    SessionConfig,
)


def completions_streamer(
    prompt: str,
    model_name: str,
    llm_model: LLM | Model,
    lib: str,
    generation_config: GenerationConfig,
    session_config: SessionConfig,
    log: Logger,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    top_k = generation_config.top_k
    log.debug("top_k: %s", top_k)
    top_p = generation_config.top_p
    log.debug("top_p: %s", top_p)
    temperature = generation_config.temperature
    log.debug("temperature: %s", temperature)
    repetition_penalty = generation_config.repetition_penalty
    log.debug("repetition_penalty: %s", repetition_penalty)
    last_n_tokens = generation_config.repetition_penalty_last_n
    log.debug("last_n_tokens: %s", last_n_tokens)
    seed = generation_config.seed
    log.debug("seed: %s", seed)
    stop = generation_config.stop_words
    log.debug("stop: %s", stop)
    max_new_tokens = generation_config.max_new_tokens
    log.debug("max_new_tokens: %s", max_new_tokens)
    batch_size = session_config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = session_config.threads
    log.debug("thread: %s", threads)

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Streaming from ctransformer instance!")
        log.debug("Prompt: %s", prompt)
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

    # the llm_model is a llm-rs instance
    if lib == "llm-rs":
        log.debug("Streaming from llm-rs instance")
        model: Model = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        for token in model.stream(prompt, generation_config=generation_config):
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
    llm_model: LLM | Model,
    lib: str,
    generation_config: GenerationConfig,
    session_config: SessionConfig,
    log: Logger,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    top_k = generation_config.top_k
    log.debug("top_k: %s", top_k)
    top_p = generation_config.top_p
    log.debug("top_p: %s", top_p)
    temperature = generation_config.temperature
    log.debug("temperature: %s", temperature)
    repetition_penalty = generation_config.repetition_penalty
    log.debug("repetition_penalty: %s", repetition_penalty)
    last_n_tokens = generation_config.repetition_penalty_last_n
    log.debug("last_n_tokens: %s", last_n_tokens)
    seed = generation_config.seed
    log.debug("seed: %s", seed)
    stop = generation_config.stop_words
    log.debug("stop: %s", stop)
    batch_size = session_config.batch_size
    log.debug("batch_size: %s", batch_size)
    threads = session_config.threads
    log.debug("threads: %s", threads)

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Streaming from ctransformer instance")
        log.debug("Prompt: %s", prompt)
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

    # the llm_model is a llm-rs instance
    if lib == "llm-rs":
        log.debug("Streaming from llm-rs instance")
        model: Model = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        for token in model.stream(prompt, generation_config=generation_config):
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
