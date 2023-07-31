from typing import (
    Any,
    Literal,
    List,
    Optional,
)
from pydantic import BaseModel, Field


class CompletionRequestBody(BaseModel):
    """_summary_
    from from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    prompt: str = Field(
        default="", description="The prompt to generate completions for."
    )
    max_tokens = Optional[int]
    temperature = Optional[float]
    top_p = Optional[float]
    stop = Optional[List[str] | str]
    stream: bool = Field()
    model: str = Field()
    # llama.cpp specific parameters
    top_k = Optional[int]
    repetition_penalty = Optional[float]
    last_n_tokens = Optional[int]
    seed = Optional[int]
    batch_size = Optional[int]
    threads = Optional[int]

    # ignored or currently unsupported
    suffix: Any = Any
    presence_penalty: Any = Any
    frequency_penalty: Any = Any
    echo: Any = Any
    n: Any = Any
    logprobs: Any = Any
    best_of: Any = Any
    logit_bias: Any = Any
    user: Any = Any

    class Config:
        arbitrary_types_allowed = True


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
    max_tokens = Optional[int]
    temperature = Optional[float]
    top_p = Optional[float]
    stop = Optional[List[str] | str]
    stream: bool = Field()
    model: str = Field()
    # llama.cpp specific parameters
    top_k = Optional[int]
    repetition_penalty = Optional[float]
    last_n_tokens = Optional[int]
    seed = Optional[int]
    batch_size = Optional[int]
    threads = Optional[int]

    # ignored or currently unsupported
    n: Any = Any
    logit_bias: Any = Any
    user: Any = Any
    presence_penalty: Any = Any
    frequency_penalty: Any = Any

    class Config:
        arbitrary_types_allowed = True
