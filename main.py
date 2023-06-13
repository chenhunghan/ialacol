"""_summary_

This module contains the main FastAPI application.
"""
import logging
import os

from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Union,
    Annotated,
)
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from llm_rs.auto import AutoModel
from llm_rs.config import (  # pylint: disable=no-name-in-module,import-error
    GenerationConfig,
    Precision,
    SessionConfig,
)
from llm_rs.base_model import Model
from ctransformers import LLM, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from request_body import ChatCompletionRequestBody, CompletionRequestBody
from response_body import ChatCompletionResponseBody, CompletionResponseBody
from streamers import chat_completions_streamer, completions_streamer
from model_generate import chat_model_generate, model_generate

DEFAULT_MODEL_HG_REPO_ID = os.environ.get("DEFAULT_MODEL_HG_REPO_ID", None)
DEFAULT_MODEL_FILE = os.environ.get("DEFAULT_MODEL_FILE", None)
DEFAULT_MODEL_META = os.environ.get("DEFAULT_MODEL_META", None)
DOWNLOAD_DEFAULT_MODEL = os.environ.get("DOWNLOAD_DEFAULT_MODEL", "True") == "True"
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
MODELS_FOLDER = os.environ.get("MODELS_FOLDER", "models")
CACHE_FOLDER = os.environ.get("MODELS_FOLDER", "cache")

THREADS = int(os.environ.get("THREADS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "1024"))

log = logging.getLogger("uvicorn")

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]

session_config = SessionConfig(
    threads=THREADS,
    batch_size=BATCH_SIZE,
    context_length=CONTEXT_LENGTH,
    # https://github.com/ggerganov/llama.cpp/discussions/1593
    keys_memory_type=Precision.FP16,
    values_memory_type=Precision.FP16,
    prefer_mmap=True,
)


async def get_llm_model(
    body: ChatCompletionRequestBody,
) -> dict[Union[Literal["lib"], Literal["llm_model"]], Any]:
    """_summary_

    Args:
        body (ChatCompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """
    verbose = LOGGING_LEVEL == "DEBUG"
    # use ctransformer if the model is a `starcoder`/`starchat`/`starcoderplus` model
    # as llm-rs does not support these models (yet) https://github.com/rustformers/llm/issues/304
    if "star" in body.model:
        log.debug("Using ctransformer model as the mode is %s", body.model)

        return dict(
            lib="ctransformer",
            llm_model=AutoModelForCausalLM.from_pretrained(
                f"./{MODELS_FOLDER}/{body.model}",
                model_type="starcoder",
            ),
        )

    log.debug("Using llm-rs model as the mode is %s", body.model)
    return dict(
        lib="llm-rs",
        llm_model=AutoModel.from_pretrained(
            model_path_or_repo_id=f"./{MODELS_FOLDER}/{body.model}",
            session_config=session_config,
            verbose=verbose,
        ),
    )


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.
    """
    log.info("Starting up...")
    log.setLevel(LOGGING_LEVEL)
    log.info("Log level set to %s", LOGGING_LEVEL)
    if DOWNLOAD_DEFAULT_MODEL is True:
        if DEFAULT_MODEL_FILE is not None and DEFAULT_MODEL_HG_REPO_ID is not None:
            log.info(
                "Downloading model... %s/%s",
                DEFAULT_MODEL_HG_REPO_ID,
                DEFAULT_MODEL_FILE,
            )
            hf_hub_download(
                repo_id=DEFAULT_MODEL_HG_REPO_ID,
                cache_dir=CACHE_FOLDER,
                local_dir=MODELS_FOLDER,
                filename=DEFAULT_MODEL_FILE,
            )
        if DEFAULT_MODEL_META is not None and DEFAULT_MODEL_HG_REPO_ID is not None:
            log.info(
                "Downloading meta... %s/%s",
                DEFAULT_MODEL_HG_REPO_ID,
                DEFAULT_MODEL_META,
            )
            hf_hub_download(
                repo_id=DEFAULT_MODEL_HG_REPO_ID,
                cache_dir=CACHE_FOLDER,
                local_dir=MODELS_FOLDER,
                filename=DEFAULT_MODEL_META,
            )
            log.info(
                "Default model/meta %s/%s downloaded to %s",
                DEFAULT_MODEL_FILE,
                DEFAULT_MODEL_META,
                MODELS_FOLDER,
            )


@app.get("/ping")
async def ping():
    """_summary_

    Returns:
        _type_: pong!
    """
    return {"ialacol": "pong"}


@app.post("/v1/completions", response_model=CompletionResponseBody)
async def completions(
    body: CompletionRequestBody,
    model_data: Annotated[dict[str, Any], Depends(get_llm_model)],
):
    """_summary_
        Compatible with https://platform.openai.com/docs/api-reference/completions
    Args:
        body (CompletionRequestBody): parsed request body

    Returns:
        StreamingResponse: streaming response
    """
    log.debug("Body:%s", str(body))
    if (
        (body.n is not None)
        or (body.logit_bias is not None)
        or (body.user is not None)
        or (body.presence_penalty is not None)
        or (body.frequency_penalty is not None)
    ):
        log.warning(
            "n, logit_bias, user, presence_penalty and frequency_penalty are not supporte."
        )
    prompt = body.prompt
    generation_config = GenerationConfig(
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        repetition_penalty=body.repeat_penalty,
        max_new_tokens=body.max_tokens,
        stop_words=body.stop,
    )
    llm_model: LLM | Model = model_data["llm_model"]
    llm_model_lib: str = model_data["lib"]
    model_name = body.model
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            completions_streamer(
                prompt,
                model_name,
                llm_model,
                llm_model_lib,
                generation_config,
                session_config,
                log,
            ),
            media_type="text/event-stream",
        )
    return model_generate(
        prompt,
        model_name,
        llm_model,
        llm_model_lib,
        generation_config,
        session_config,
        log,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponseBody)
async def chat_completions(
    body: ChatCompletionRequestBody,
    model_data: Annotated[dict[str, Any], Depends(get_llm_model)],
):
    """_summary_
        Compatible with https://platform.openai.com/docs/api-reference/chat
    Args:
        body (ChatCompletionRequestBody): parsed request body

    Returns:
        StreamingResponse: streaming response
    """
    log.debug("Body:%s", str(body))
    if (
        (body.n is not None)
        or (body.logit_bias is not None)
        or (body.user is not None)
        or (body.presence_penalty is not None)
        or (body.frequency_penalty is not None)
    ):
        log.warning(
            "n, logit_bias, user, presence_penalty and frequency_penalty are not supporte."
        )
    user_message = next(
        (message for message in body.messages if message.role == "user"), None
    )
    user_message_content = user_message.content if user_message else ""
    assistant_message = next(
        (message for message in body.messages if message.role == "assistant"), None
    )
    assistant_message_content = (
        f"### Assistant: {assistant_message.content}" if assistant_message else ""
    )
    system_message = next(
        (message for message in body.messages if message.role == "system"), None
    )
    system_message_content = system_message.content if system_message else ""

    prompt = f"{system_message_content} {assistant_message_content} ### Human: {user_message_content} ### Assistant:"
    generation_config = GenerationConfig(
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        repetition_penalty=body.repeat_penalty,
        repetition_penalty_last_n=body.repeat_penalty_last_n,
        seed=body.seed,
        max_new_tokens=body.max_tokens,
        stop_words=body.stop,
    )
    llm_model: LLM | Model = model_data["llm_model"]
    llm_model_lib: str = model_data["lib"]
    model_name = body.model
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            chat_completions_streamer(
                prompt,
                model_name,
                llm_model,
                llm_model_lib,
                generation_config,
                session_config,
                log,
            ),
            media_type="text/event-stream",
        )
    return chat_model_generate(
        prompt,
        model_name,
        llm_model,
        llm_model_lib,
        generation_config,
        session_config,
        log,
    )
