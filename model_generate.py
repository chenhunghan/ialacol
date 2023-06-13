from time import time
from logging import Logger
from typing import List
from ctransformers import LLM
from llm_rs.base_model import Model
from llm_rs.config import (  # pylint: disable=no-name-in-module,import-error
    GenerationConfig,
    SessionConfig,
)
from llm_rs.results import (  # pylint: disable=no-name-in-module,import-error
    GenerationResult,
)

def model_generate(
    prompt: str,
    model_name: str,
    llm_model: LLM | Model,
    lib: str,
    generation_config: GenerationConfig,
    session_config: SessionConfig,
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Getting from ctransformer instance")
        result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
            prompt=prompt,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            repetition_penalty=generation_config.repetition_penalty,
            last_n_tokens=generation_config.repetition_penalty_last_n,
            seed=generation_config.seed,
            batch_size=session_config.batch_size,
            threads=session_config.threads,
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

    # the llm_model is a llm-rs instance
    if lib == "llm-rs":
        log.debug("Getting from llm-rs instance")
        model: Model = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        generation_result: GenerationResult | List[float] = model.generate(
            prompt,
            generation_config=generation_config,  # pyright: ignore [reportGeneralTypeIssues]
        )
        if not isinstance(generation_result, GenerationResult):
            return {"error": "unknown generation_result"}
        stop_reason = generation_result.stop_reason
        finnish_reason = "unknown"
        if stop_reason == 0:
            finnish_reason = "end_of_token"
        elif stop_reason == 1:
            finnish_reason = "max_length"
        elif stop_reason == 2:
            finnish_reason = "user_cancelled"

        http_response = {
            "id": "id",
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": generation_result.text,
                    "logprobs": None,
                    "finish_reason": finnish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        log.debug("http_response:%s ", http_response)
        return http_response

def chat_model_generate(
    prompt: str,
    model_name: str,
    llm_model: LLM | Model,
    lib: str,
    generation_config: GenerationConfig,
    session_config: SessionConfig,
    log: Logger,
):
    """_summary_
    returns the response body for /chat/completions
    """
    created = time()

    # the llm_model is a ctransformer model instance
    if lib == "ctransformer":
        llm: LLM = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        log.debug("Getting from ctransformer instance")
        result: str = llm(  # pyright: ignore [reportGeneralTypeIssues]
            prompt=prompt,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
            repetition_penalty=generation_config.repetition_penalty,
            last_n_tokens=generation_config.repetition_penalty_last_n,
            seed=generation_config.seed,
            batch_size=session_config.batch_size,
            threads=session_config.threads,
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

    # the llm_model is a llm-rs instance
    if lib == "llm-rs":
        log.debug("Getting from llm-rs instance")
        model: Model = llm_model  # pyright: ignore [reportGeneralTypeIssues]
        generation_result: GenerationResult | List[float] = model.generate(
            prompt,
            generation_config=generation_config,  # pyright: ignore [reportGeneralTypeIssues]
        )
        if not isinstance(generation_result, GenerationResult):
            return {"error": "unknown generation_result"}
        stop_reason = generation_result.stop_reason
        finnish_reason = "unknown"
        if stop_reason == 0:
            finnish_reason = "end_of_token"
        elif stop_reason == 1:
            finnish_reason = "max_length"
        elif stop_reason == 2:
            finnish_reason = "user_cancelled"

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
                        "content": generation_result.text,
                    },
                    "finish_reason": finnish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        log.debug("http_response:%s ", http_response)
        return http_response
