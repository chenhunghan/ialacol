"""_summary_

This module contains the main FastAPI application.
"""
import logging
import os

from typing import (
    Awaitable,
    Callable,
    Union,
    Annotated,
)
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from ctransformers import LLM, AutoModelForCausalLM, Config
from huggingface_hub import hf_hub_download

from request_body import ChatCompletionRequestBody, CompletionRequestBody
from response_body import ChatCompletionResponseBody, CompletionResponseBody
from streamers import chat_completions_streamer, completions_streamer
from model_generate import chat_model_generate, model_generate

DEFAULT_MODEL_HG_REPO_ID = os.environ.get(
    "DEFAULT_MODEL_HG_REPO_ID", "TheBloke/Llama-2-7B-Chat-GGML"
)
DEFAULT_MODEL_FILE = os.environ.get(
    "DEFAULT_MODEL_FILE", "llama-2-7b-chat.ggmlv3.q4_0.bin"
)
DEFAULT_MODEL_META = os.environ.get("DEFAULT_MODEL_META", "")
DOWNLOAD_DEFAULT_MODEL = os.environ.get("DOWNLOAD_DEFAULT_MODEL", "True") == "True"
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
MODELS_FOLDER = os.environ.get("MODELS_FOLDER", "models")
CACHE_FOLDER = os.environ.get("MODELS_FOLDER", "cache")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "1024"))


def get_default_thread():
    """_summary_
    Automatically get the default number of threads to use for generation
    """
    count = os.cpu_count()
    if count is not None:
        return count / 2
    else:
        return 8


THREADS = int(os.environ.get("THREADS", get_default_thread()))

DOWNLOADING_MODEL = False


def set_downloading_model(boolean: bool):
    """_summary_

    Args:
        boolean (bool): the boolean value to set DOWNLOADING_MODEL to
    """
    globals()["DOWNLOADING_MODEL"] = boolean
    log.info("DOWNLOADING_MODEL set to %s", globals()["DOWNLOADING_MODEL"])


log = logging.getLogger("uvicorn")

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]


async def get_llm_model(
    body: ChatCompletionRequestBody,
) -> LLM:
    """_summary_

    Args:
        body (ChatCompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """

    # These are also in "starcoder" format
    # https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML
    # https://huggingface.co/TheBloke/minotaur-15B-GGML
    if (
        "star" in body.model
        or "starchat" in body.model
        or "WizardCoder" in body.model
        or "minotaur-15" in body.model
    ):
        ctransformer_model_type = "starcoder"

    ctransformer_model_type = "llama"
    if "llama" in body.model:
        ctransformer_model_type = "llama"
    if "mpt" in body.model:
        ctransformer_model_type = "mpt"
    if "replit" in body.model:
        ctransformer_model_type = "replit"
    if "falcon" in body.model:
        ctransformer_model_type = "falcon"
    if "dolly" in body.model:
        ctransformer_model_type = "dolly-v2"
    if "stablelm" in body.model:
        ctransformer_model_type = "gpt_neox"

    return AutoModelForCausalLM.from_pretrained(
        f"./{MODELS_FOLDER}/{body.model}", model_type=ctransformer_model_type
    )


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.
    """
    log.info("Starting up...")
    try:
        log.setLevel(LOGGING_LEVEL)
        log.info("Log level set to %s", LOGGING_LEVEL)
    except ValueError:
        log.setLevel("INFO")
        log.info("Unknown Log level %s, fallback to INFO", LOGGING_LEVEL)
    if DOWNLOAD_DEFAULT_MODEL is True:
        if DEFAULT_MODEL_FILE and DEFAULT_MODEL_HG_REPO_ID:
            set_downloading_model(True)
            log.info(
                "Downloading model... %s/%s",
                DEFAULT_MODEL_HG_REPO_ID,
                DEFAULT_MODEL_FILE,
            )
            try:
                hf_hub_download(
                    repo_id=DEFAULT_MODEL_HG_REPO_ID,
                    cache_dir=CACHE_FOLDER,
                    local_dir=MODELS_FOLDER,
                    filename=DEFAULT_MODEL_FILE,
                )
            except Exception as exception:
                log.error("Error downloading model: %s", exception)
            finally:
                set_downloading_model(False)

        if DEFAULT_MODEL_META and DEFAULT_MODEL_HG_REPO_ID:
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


@app.get("/v1/models")
async def models():
    """_summary_

    Returns:
        _type_: a list of models
    """
    if DOWNLOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Downloading model")
    return {
        "data": [
            {
                "id": DEFAULT_MODEL_FILE,
                "object": "model",
                "owned_by": "community",
                "permission": [],
            }
        ],
        "object": "list",
    }


@app.post("/v1/completions", response_model=CompletionResponseBody)
async def completions(
    body: CompletionRequestBody,
    llm: Annotated[LLM, Depends(get_llm_model)],
):
    """_summary_
        Compatible with https://platform.openai.com/docs/api-reference/completions
    Args:
        body (CompletionRequestBody): parsed request body

    Returns:
        StreamingResponse: streaming response
    """
    if DOWNLOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Downloading model")
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
    config = Config(
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        repetition_penalty=body.repeat_penalty,
        max_new_tokens=body.max_tokens,
        stop=body.stop,
    )

    model_name = body.model
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            completions_streamer(
                prompt,
                model_name,
                llm,
                config,
                log,
            ),
            media_type="text/event-stream",
        )
    return model_generate(
        prompt,
        model_name,
        llm,
        config,
        log,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponseBody)
async def chat_completions(
    body: ChatCompletionRequestBody,
    llm: Annotated[LLM, Depends(get_llm_model)],
):
    """_summary_
        Compatible with https://platform.openai.com/docs/api-reference/chat
    Args:
        body (ChatCompletionRequestBody): parsed request body

    Returns:
        StreamingResponse: streaming response
    """
    if DOWNLOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Downloading model")
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
    default_assistant_start = "### Assistant: "
    default_assistant_end = ""
    default_user_start = "### Human: "
    default_user_end = ""
    default_system = ""

    if "llama" in body.model:
        default_assistant_start = "ASSISTANT: \n"
        default_user_start = "USER: "
        default_user_end = "\n"
        default_system = "SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    # For most instruct fine-tuned models using  Alpaca prompt template
    # Although instruct fine-tuned models are not tuned for chat, they can be to generate response as if chatting, using Alpaca
    # prompt template likely gives better results than using the default prompt template
    # See https://github.com/tatsu-lab/stanford_alpaca#data-release
    if "instruct" in body.model:
        default_assistant_start = "### Response:"
        default_user_start = "### Instruction: "
        default_user_end = "\n\n"
        default_system = "Below is an instruction that describes a task. Write a response that appropriately completes the request\n\n"
    if "starchat" in body.model:
        # See https://huggingface.co/blog/starchat-alpha and https://huggingface.co/TheBloke/starchat-beta-GGML#prompt-template
        default_assistant_start = "<|assistant|>\n"
        default_assistant_end = " <|end|>\n"
        default_user_start = "<|user|>\n"
        default_user_end = " <|end|>\n"
        default_system = "<|system|>\nBelow is a dialogue between a human and an AI assistant called StarChat.<|end|>\n"
    if "airoboros" in body.model:
        # e.g. A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. USER: [prompt] ASSISTANT:
        # see https://huggingface.co/jondurbin/airoboros-mpt-30b-gpt4-1p4-five-epochs
        default_assistant_start = "ASSISTANT: "
        default_user_start = "USER: "
        default_system = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."
    # If it's a mpt-chat model, we need to add the default prompt
    # from https://huggingface.co/TheBloke/mpt-30B-chat-GGML#prompt-template
    # and https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py#L17
    if "mpt" in body.model and "chat" in body.model:
        default_assistant_start = "<|im_start|>assistant\n"
        default_assistant_end = "<|im_end|>\n"
        default_user_start = "<|im_start|>user\n"
        default_user_end = "<|im_end|>\n"
        default_system = "<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n"

    user_message = next(
        (message for message in body.messages if message.role == "user"), None
    )
    user_message_content = user_message.content if user_message else ""
    assistant_message = next(
        (message for message in body.messages if message.role == "assistant"), None
    )
    assistant_message_content = (
        f"{default_assistant_start}{assistant_message.content}{default_assistant_end}"
        if assistant_message
        else ""
    )
    system_message = next(
        (message for message in body.messages if message.role == "system"), None
    )
    system_message_content = (
        system_message.content if system_message else default_system
    )

    prompt = f"{system_message_content}{assistant_message_content} {default_user_start}{user_message_content}{default_user_end} {default_assistant_start}"
    config = Config(
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        repetition_penalty=body.repeat_penalty,
        last_n_tokens=body.repeat_penalty_last_n,
        seed=body.seed,
        max_new_tokens=body.max_tokens,
        stop=body.stop,
    )
    model_name = body.model
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            chat_completions_streamer(
                prompt,
                model_name,
                llm,
                config,
                log,
            ),
            media_type="text/event-stream",
        )
    return chat_model_generate(
        prompt,
        model_name,
        llm,
        config,
        log,
    )
