from typing import (
    Literal,
    List,
    Optional,
    Dict,
)
from pydantic import BaseModel, Field

from fields import (
    max_tokens,
    temperature,
    top_p,
    stop,
    stream,
    model,
    presence_penalty,
    frequency_penalty,
    top_k,
    repeat_penalty,
)


class CompletionRequestBody(BaseModel):
    """_summary_
    from from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    prompt: str = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = max_tokens
    temperature: Optional[float] = temperature
    top_p: Optional[float] = top_p
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[List[str]] = stop
    stream: bool = stream
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty
    frequency_penalty: Optional[float] = frequency_penalty

    model: str = model
    n: Optional[int] = 1
    logprobs: Optional[int] = Field(None)
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)

    # llama.cpp specific parameters
    top_k: int = top_k
    repeat_penalty: float = repeat_penalty


class ChatCompletionRequestMessage(BaseModel):
    """_summary_
    from from  https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="The role of the message."
    )
    content: str = Field(default="", description="The content of the message.")


class ChatCompletionRequestBody(BaseModel):
    """_summary_
    Request body for /chat/completions.
    from from  https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    messages: List[ChatCompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )
    max_tokens: int = max_tokens
    temperature: Optional[float] = temperature
    top_p: Optional[float] = top_p
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
