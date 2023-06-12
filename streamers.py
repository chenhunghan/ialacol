import json
from logging import Logger
from os import times

from typing import Any

from llm_rs.config import GenerationConfig


def chat_completions_streamer(
    prompt: str,
    model_name: str,
    llm_model: Any,
    lib: str,
    generation_config: GenerationConfig,
    log: Logger,
):
    created = times()

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        log.debug("Streaming from ctransformer instance")
        for token in llm_model(prompt, stream=True):
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
        for token in llm_model.stream(prompt, generation_config=generation_config):
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
