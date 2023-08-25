from time import time
from logging import Logger
from ctransformers import LLM


def model_generate(
    prompt: str,
    model_name: str,
    llm: LLM,
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

    log.debug("prompt: %s", prompt)

    log.debug("Getting from ctransformer instance")
    result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
        prompt=prompt,
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
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

    log.debug("prompt: %s", prompt)

    log.debug("Getting from ctransformer instance")
    result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
        prompt=prompt,
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
