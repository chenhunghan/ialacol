import json
from logging import Logger
from os import times

from ctransformers import LLM
from llm_rs.base_model import Model

from llm_rs.config import GenerationConfig, SessionConfig # pylint: disable=no-name-in-module,import-error

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

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Streaming from ctransformer instance")
        for token in llm(
            prompt,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            repetition_penalty=generation_config.repetition_penalty,
            last_n_tokens=generation_config.repetition_penalty_last_n,
            seed=generation_config.seed,
            stop=generation_config.stop_words,
            batch_size=session_config.batch_size,
            threads=session_config.threads,
            stream=True,
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

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Streaming from ctransformer instance")
        for token in llm(
            prompt,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            repetition_penalty=generation_config.repetition_penalty,
            last_n_tokens=generation_config.repetition_penalty_last_n,
            seed=generation_config.seed,
            stop=generation_config.stop_words,
            batch_size=session_config.batch_size,
            threads=session_config.threads,
            stream=True,
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
