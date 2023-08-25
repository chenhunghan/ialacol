import json
from logging import Logger
from os import times

from ctransformers import LLM


def completions_streamer(
    prompt: str,
    model_name: str,
    llm: LLM,
    log: Logger,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    log.debug("prompt: %s", prompt)

    log.debug("Streaming from ctransformer instance!")
    for token in llm(
        prompt,
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
    log: Logger,
):
    """_summary_
    returns a generator that yields a stream of responses
    """
    created = times()

    log.debug("prompt: %s", prompt)

    log.debug("Streaming from ctransformer instance")
    for token in llm(
        prompt,
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
