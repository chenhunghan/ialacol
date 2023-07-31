from typing import (
    Any,
    Literal,
    List,
    Optional,
    Dict,
)
from pydantic import BaseModel, Field


class CompletionRequestBody(BaseModel):
    """_summary_
    from from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    prompt: str = Field(
        default="", description="The prompt to generate completions for."
    )
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    stop: None | str | List[str]
    stream: bool
    model: str
    # llama.cpp specific parameters
    top_k = Optional[int]
    repetition_penalty = Optional[float]
    last_n_tokens = Optional[int]
    seed = Optional[int]
    batch_size = Optional[int]
    threads = Optional[int]

    # ignored or currently unsupported
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    logprobs: Optional[int] = Field()
    presence_penalty: Optional[float] = Any
    frequency_penalty: Optional[float] = Any
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    n: Optional[int] = 1
    logprobs: Optional[int] = Field(None)
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)


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
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    stop: None | str | List[str]
    stream: bool
    model: str
    # llama.cpp specific parameters
    top_k = Optional[int]
    repetition_penalty = Optional[float]
    last_n_tokens = Optional[int]
    seed = Optional[int]
    batch_size = Optional[int]
    threads = Optional[int]

    # ignored or currently unsupported
    n: Optional[int] = Field(None)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)
    presence_penalty: Optional[float] = Any
    frequency_penalty: Optional[float] = Any
