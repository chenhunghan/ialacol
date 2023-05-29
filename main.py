"""_summary_

This module contains the main FastAPI application.
"""
import logging
import os
import json
from time import time
from typing import (
    Awaitable,
    Callable,
    Iterator,
    Literal,
    Union,
    List,
    Optional,
    Dict,
    Annotated,
)
from queue import Queue, Empty
from threading import Thread
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from llm_rs.auto import AutoModel
from llm_rs.base_model import Model
from llm_rs.config import GenerationConfig, Precision, SessionConfig
from llm_rs.results import GenerationResult
from starlette.types import Send
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "DEBUG")
MODELS_FOLDER = os.environ.get("MODELS_FOLDER", "models")
CACHE_FOLDER = os.environ.get("MODELS_FOLDER", "cache")
DEFAULT_MODEL_HG_REPO_ID = os.environ.get(
    "DEFAULT_MODEL_HG_REPO_ID", "rustformers/pythia-ggml"
)
DEFAULT_MODEL_FILE = os.environ.get("DEFAULT_MODEL_FILE", "pythia-70m-q4_0.bin")
DEFAULT_MODEL_META = os.environ.get("DEFAULT_MODEL_META", "pythia-70m-q4_0.meta")
DOWNLOAD_DEFAULT_MODEL = os.environ.get("DOWNLOAD_DEFAULT_MODEL", "True") == "True"

log = logging.getLogger("uvicorn")

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]


class EmptyIterator(Iterator[Union[str, bytes]]):
    """_summary_

    Args:
        Iterator (_type_): _description_
    """

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class ModelStreamingResponse(StreamingResponse):
    """Streaming response for model, inheritance from StreamingResponse."""

    sender: Sender
    DONE = object()
    ERROR = object()
    exception: Exception | None

    def __init__(
        self,
        prompt: str,
        model_name: str,
        _llm_model: Model,
        generation_config: GenerationConfig,
        status_code: int = 200,
        media_type: str = "text/event-stream",
    ) -> None:
        super().__init__(
            content=EmptyIterator(), status_code=status_code, media_type=media_type
        )
        self.prompt = prompt
        self.model_name = model_name
        self.llm_model = _llm_model
        self.queue = Queue()
        self.exception = None
        self.generation_config = generation_config

    def get_model_response(self):
        """Get model response and put it into queue."""

        def callback(token: str):
            self.queue.put(token)

        try:
            self.llm_model.generate(
                self.prompt, callback=callback, generation_config=self.generation_config
            )
        # pylint: disable=broad-except
        except Exception as exception:
            self.queue.put(self.ERROR)
            self.exception = exception

        self.queue.put(self.DONE)

    async def stream_response(self, send: Send) -> None:
        """Rewrite stream_response to send response to client."""
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        created = time()
        thread = Thread(target=self.get_model_response)
        thread.start()
        while True:
            try:
                token = self.queue.get()
                if token is self.DONE:
                    break
                if token is self.ERROR and self.exception is not None:
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                    raise self.exception
                data = json.dumps(
                    {
                        "id": "id",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": token},
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                chunk_body = bytes(f"data: {data}" + "\n\n", "utf-8")
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk_body,
                        "more_body": True,
                    }
                )
            except Empty:
                break
        thread.join()
        stop_data = json.dumps(
            {
                "id": "id",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": bytes(f"data: {stop_data}" + "\n\n", "utf-8"),
                "more_body": True,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": bytes("data: [DONE]" + "\n\n", "utf-8"),
                "more_body": True,
            }
        )
        # send empty body to client to close connection
        await send({"type": "http.response.body", "body": b"", "more_body": False})


model = Field(description="The model to use for generating completions.")

max_tokens = Field(
    default=None, ge=1, le=2048, description="The maximum number of tokens to generate."
)

temperature = Field(
    default=0.01,
    ge=0.00001,
    le=1.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

stop = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream = Field(
    default=False,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty = Field(
    default=1.3,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

repeat_penalty_last_n = Field(
    default=512,
    ge=0,
)

seed = Field(
    default=42,
    ge=0,
)

presence_penalty = Field(
    default=None,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty = Field(
    default=None,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)


# from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
class ChatCompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


# from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
class ChatCompletionRequestBody(BaseModel):
    """Request body for /chat/completions."""

    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = max_tokens
    temperature: float = temperature
    top_p: float = top_p
    stop: Optional[List[str]] = stop
    stream: bool = stream

    model: str = model

    # ignored or currently unsupported
    n: Optional[int] = Field(None)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)
    presence_penalty: Optional[float] = presence_penalty
    frequency_penalty: Optional[float] = frequency_penalty

    # llama.cpp specific parameters
    top_k: int = top_k
    repeat_penalty: float = repeat_penalty

    # Unknown parameters but required by GenerationConfig
    repeat_penalty_last_n: int = repeat_penalty_last_n
    seed: int = seed


class Message(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): message in choice
    """

    role: Literal["system", "user", "assistant"]
    content: str


class Choice(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): choice in completion response
    """

    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponseBody(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): response body for /chat/completions
    """

    id: str
    object: str
    created: int
    choices: list[Choice]
    usage: dict[str, int]


session_config = SessionConfig(
    threads=8,
    batch_size=8,
    context_length=2048,
    keys_memory_type=Precision.FP32,
    values_memory_type=Precision.FP32,
    prefer_mmap=True,
)


async def get_llm_model(body: ChatCompletionRequestBody):
    """_summary_

    Args:
        body (ChatCompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """
    return AutoModel.from_pretrained(
        model_path_or_repo_id=f"./{MODELS_FOLDER}/{body.model}",
        session_config=session_config,
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
        log.info(
            "Downloading model... %s/%s", DEFAULT_MODEL_HG_REPO_ID, DEFAULT_MODEL_FILE
        )
        hf_hub_download(
            repo_id=DEFAULT_MODEL_HG_REPO_ID,
            cache_dir=CACHE_FOLDER,
            local_dir=MODELS_FOLDER,
            filename=DEFAULT_MODEL_FILE,
        )
        log.info(
            "Downloading meta... %s/%s", DEFAULT_MODEL_HG_REPO_ID, DEFAULT_MODEL_META
        )
        hf_hub_download(
            repo_id=DEFAULT_MODEL_HG_REPO_ID,
            cache_dir=CACHE_FOLDER,
            local_dir=MODELS_FOLDER,
            filename=DEFAULT_MODEL_META,
        )
        log.info("Default model/meta downloaded to %s", DEFAULT_MODEL_HG_REPO_ID)


@app.get("/ping")
async def ping():
    """_summary_

    Returns:
        _type_: pong!
    """
    return {"ialacol": "pong"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponseBody)
async def chat_completions(
    body: ChatCompletionRequestBody, llm_model: Annotated[Model, Depends(get_llm_model)]
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
        raise HTTPException(
            status_code=422,
            detail="n, logit_bias, user, presence_penalty and frequency_penalty are not supporte.",
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
    log.debug("Prompt:%s", prompt)

    if body.stream is True:
        log.debug("Streaming...")
        return ModelStreamingResponse(
            prompt,
            body.model,
            llm_model,
            GenerationConfig(
                top_k=body.top_k,
                top_p=body.top_p,
                temperature=body.temperature,
                repetition_penalty=body.repeat_penalty,
                repetition_penalty_last_n=body.repeat_penalty_last_n,
                seed=body.seed,
                max_new_tokens=body.max_tokens,
                stop_words=body.stop,
            ),
        )
    generation_result: GenerationResult = llm_model.generate(prompt)

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
        "created": time(),
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
