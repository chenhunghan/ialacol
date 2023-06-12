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
    repeat_penalty_last_n,
    seed,
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
