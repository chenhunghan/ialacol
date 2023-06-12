from typing import Literal
from pydantic import BaseModel

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
