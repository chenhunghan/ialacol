"""_summary_

This module contains the main FastAPI application.
"""
import os

from typing import (
    Awaitable,
    Callable,
    Union,
    Annotated,
)
from fastapi import FastAPI, Depends, HTTPException, Body, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from ctransformers import LLM, AutoModelForCausalLM, Config
from huggingface_hub import hf_hub_download, snapshot_download
from get_config import get_config
from get_model_type import get_model_type

from request_body import ChatCompletionRequestBody, CompletionRequestBody
from response_body import ChatCompletionResponseBody, CompletionResponseBody
from streamers import chat_completions_streamer, completions_streamer
from model_generate import chat_model_generate, model_generate
from get_env import get_env
from log import log
from truncate import truncate
from const import DEFAULT_CONTEXT_LENGTH

DEFAULT_MODEL_HG_REPO_ID = get_env(
    "DEFAULT_MODEL_HG_REPO_ID", "TheBloke/Llama-2-7B-Chat-GGML"
)
DEFAULT_MODEL_HG_REPO_REVISION = get_env(
    "DEFAULT_MODEL_HG_REPO_REVISION", "main"
)
DEFAULT_MODEL_FILE = get_env("DEFAULT_MODEL_FILE", "llama-2-7b-chat.ggmlv3.q4_0.bin")

log.info("DEFAULT_MODEL_HG_REPO_ID: %s", DEFAULT_MODEL_HG_REPO_ID)
log.info("DEFAULT_MODEL_HG_REPO_REVISION: %s", DEFAULT_MODEL_HG_REPO_REVISION)
log.info("DEFAULT_MODEL_FILE: %s", DEFAULT_MODEL_FILE)

DOWNLOADING_MODEL = False
LOADING_MODEL = False


def set_downloading_model(boolean: bool):
    """_summary_

    Args:
        boolean (bool): the boolean value to set DOWNLOADING_MODEL to
    """
    globals()["DOWNLOADING_MODEL"] = boolean
    log.debug("DOWNLOADING_MODEL set to %s", globals()["DOWNLOADING_MODEL"])


def set_loading_model(boolean: bool):
    """_summary_

    Args:
        boolean (bool): the boolean value to set LOADING_MODEL to
    """
    globals()["LOADING_MODEL"] = boolean
    log.debug("LOADING_MODEL set to %s", globals()["LOADING_MODEL"])


Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]


app = FastAPI()

# https://github.com/tiangolo/fastapi/issues/3361
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    log.error("%s: %s", request, exc_str)
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.
    """
    log.info("Starting up...")
    model_type = get_model_type(DEFAULT_MODEL_FILE)
    if DEFAULT_MODEL_HG_REPO_ID:
        set_downloading_model(True)

        try:
            if model_type == "gptq":
                log.info(
                    "Downloading repo %s to %s/models",
                    DEFAULT_MODEL_HG_REPO_ID,
                    os.getcwd(),
                )
                snapshot_download(
                    repo_id=DEFAULT_MODEL_HG_REPO_ID,
                    revision=DEFAULT_MODEL_HG_REPO_REVISION,
                    cache_dir="models/.cache",
                    local_dir="models",
                    resume_download=True,
                )
            elif DEFAULT_MODEL_FILE:
                log.info(
                    "Downloading model... %s/%s to %s/models",
                    DEFAULT_MODEL_HG_REPO_ID,
                    DEFAULT_MODEL_FILE,
                    os.getcwd(),
                )
                hf_hub_download(
                    repo_id=DEFAULT_MODEL_HG_REPO_ID,
                    revision=DEFAULT_MODEL_HG_REPO_REVISION,
                    cache_dir="models/.cache",
                    local_dir="models",
                    filename=DEFAULT_MODEL_FILE,
                    resume_download=True,
                )
        except Exception as exception:
            log.error("Error downloading model: %s", exception)
        finally:
            set_downloading_model(False)

    # ggml only, follow ctransformers defaults
    CONTEXT_LENGTH = int(get_env("CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
    # the layers to offloading to the GPU
    GPU_LAYERS = int(get_env("GPU_LAYERS", "0"))

    log.debug("CONTEXT_LENGTH: %s", CONTEXT_LENGTH)
    log.debug("GPU_LAYERS: %s", GPU_LAYERS)

    config = Config(
        context_length=CONTEXT_LENGTH,
        gpu_layers=GPU_LAYERS,
    )

    log.info(
        "Creating llm singleton with model_type: %s",
        model_type,
    )
    set_loading_model(True)
    if model_type == "gptq":
        log.debug("Creating llm/gptq instance...")
        llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=f"{os.getcwd()}/models",
            model_type="gptq",
            local_files_only=True,
        )
        app.state.llm = llm
    else:
        log.debug("Creating llm/ggml instance...")
        llm = LLM(
            model_path=f"{os.getcwd()}/models/{DEFAULT_MODEL_FILE}",
            config=config,
            model_type=model_type,
        )
        app.state.llm = llm
    log.info("llm singleton created.")
    set_loading_model(False)


@app.get("/v1/models")
async def models():
    """_summary_

    Returns:
        _type_: a list of models
    """
    if DOWNLOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Downloading model")
    if LOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Loading model in memory")
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
    body: Annotated[CompletionRequestBody, Body()],
    config: Annotated[Config, Depends(get_config)],
    request: Request,
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

    model_name = body.model
    llm = request.app.state.llm
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            completions_streamer(prompt, model_name, llm, config),
            media_type="text/event-stream",
        )
    return model_generate(prompt, model_name, llm, config)


@app.post("/v1/engines/{engine}/completions")
async def engine_completions(
    # Can't use body as FastAPI require corrent context-type header
    # But copilot client maybe not send such header
    request: Request,
    # copilot client ONLY request param
    engine: str,
):
    """_summary_
        Similar to https://platform.openai.com/docs/api-reference/completions
        but with engine param and with /v1/engines
    Args:
        body (CompletionRequestBody): parsed request body
    Returns:
        StreamingResponse: streaming response
    """
    if DOWNLOADING_MODEL is True:
        raise HTTPException(status_code=503, detail="Downloading model")
    json = await request.json()
    log.debug("Body:%s", str(json))

    body = CompletionRequestBody(**json, model=engine)
    prompt = truncate(body.prompt)

    config = get_config(body)
    llm = request.app.state.llm
    if body.stream is True:
        log.debug("Streaming response from %s", engine)
        return StreamingResponse(
            completions_streamer(prompt, engine, llm, config),
            media_type="text/event-stream",
        )
    return model_generate(prompt, engine, llm, config)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponseBody)
async def chat_completions(
    body: Annotated[ChatCompletionRequestBody, Body()],
    config: Annotated[Config, Depends(get_config)],
    request: Request,
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
    system_start = ""
    system = "You are a helpful assistant."
    system_end = ""
    user_start = ""
    user_end = ""
    assistant_start = ""
    assistant_end = ""

    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/3
    if "llama-2" in body.model.lower() and "chat" in body.model.lower():
        system_start = "<s>[INST] <<SYS>>\n"
        system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        system_end = "<</SYS>>\n\n"
        assistant_start = " "
        assistant_end = " </s><s>[INST] "
        user_start = ""
        user_end = " [/INST]"
    # For most instruct fine-tuned models using  Alpaca prompt template
    # Although instruct fine-tuned models are not tuned for chat, they can be to generate response as if chatting, using Alpaca
    # prompt template likely gives better results than using the default prompt template
    # See https://github.com/tatsu-lab/stanford_alpaca#data-release
    if "instruct" in body.model.lower():
        system_start = ""
        system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        system_end = ""
        assistant_start = "### Response:"
        assistant_end = ""
        user_start = "### Instruction:\n"
        user_end = "\n\n"
    # For instruct fine-tuned models using mistral prompt template
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    if "mistral" in body.model.lower() and "instruct" in body.model.lower():
        system_start = "<s>"
        system = ""
        system_end = ""
        assistant_start = ""
        assistant_end = "</s> "
        user_start = "[INST] "
        user_end = " [/INST]"
    if "starchat" in body.model.lower():
        # See https://huggingface.co/blog/starchat-alpha and https://huggingface.co/TheBloke/starchat-beta-GGML#prompt-template
        system_start = "<|system|>"
        system = (
            "Below is a dialogue between a human and an AI assistant called StarChat."
        )
        system_end = " <|end|>\n"
        user_start = "<|user|>"
        user_end = " <|end|>\n"
        assistant_start = "<|assistant|>\n"
        assistant_end = " <|end|>\n"
    if "airoboros" in body.model.lower():
        # e.g. A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. USER: [prompt] ASSISTANT:
        # see https://huggingface.co/jondurbin/airoboros-mpt-30b-gpt4-1p4-five-epochs
        system_start = ""
        system = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."
        system_end = ""
        user_start = "USER: "
        user_end = ""
        assistant_start = "ASSISTANT: "
        assistant_end = ""
    # If it's a mpt-chat model, we need to add the default prompt
    # from https://huggingface.co/TheBloke/mpt-30B-chat-GGML#prompt-template
    # and https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py#L17
    if "mpt" in body.model.lower() and "chat" in body.model.lower():
        system_start = "<|im_start|>system\n"
        system = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
        system_end = "<|im_end|>\n"
        assistant_start = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>\n"
        user_start = "<|im_start|>user\n"
        user_end = "<|im_end|>\n"
    # orca mini https://huggingface.co/pankajmathur/orca_mini_3b
    if "orca" in body.model.lower() and "mini" in body.model.lower():
        system_start = "### System:\n"
        system = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
        system_end = "\n\n"
        assistant_start = "### Response:\n"
        assistant_end = ""
        # v3 e.g. https://huggingface.co/pankajmathur/orca_mini_v3_13b
        if "v3" in body.model.lower():
            assistant_start = "### Assistant:\n"
        user_start = "### User:\n"
        user_end = "\n\n"
    # openchat_3.5 https://huggingface.co/openchat/openchat_3.5
    if "openchat" in body.model.lower():
        system_start = ""
        system = ""
        system_end = ""
        assistant_start = "GPT4 Assistant: "
        assistant_end = "<|end_of_turn|>"
        user_start = "GPT4 User: "
        user_end = "<|end_of_turn|>"

    user_message = next(
        (message for message in reversed(body.messages) if message.role == "user"), None
    )
    user_message_content = user_message.content if user_message else ""
    assistant_message = next(
        (message for message in reversed(body.messages) if message.role == "assistant"), None
    )
    assistant_message_content = (
        f"{assistant_start}{assistant_message.content}{assistant_end}"
        if assistant_message
        else ""
    )
    system_message = next(
        (message for message in body.messages if message.role == "system"), None
    )
    system_message_content = system_message.content if system_message else system
    # avoid duplicate user start token in prompt if user message already includes it
    if len(user_start) > 0 and user_start in user_message_content:
        user_start = ""
    # avoid duplicate user end token in prompt if user message already includes it
    if len(user_end) > 0 and user_end in user_message_content:
        user_end = ""
    # avoid duplicate assistant start token in prompt if user message already includes it
    if len(assistant_start) > 0 and assistant_start in user_message_content:
        assistant_start = ""
    # avoid duplicate system_start token in prompt if system_message_content already includes it
    if len(system_start) > 0 and system_start in system_message_content:
        system_start = ""
    prompt = f"{system_start}{system_message_content}{system_end}{assistant_message_content}{user_start}{user_message_content}{user_end}{assistant_start}"
    model_name = body.model
    llm = request.app.state.llm
    if body.stream is True:
        log.debug("Streaming response from %s", model_name)
        return StreamingResponse(
            chat_completions_streamer(prompt, model_name, llm, config),
            media_type="text/event-stream",
        )
    return chat_model_generate(prompt, model_name, llm, config)
