from typing import List
from pydantic import Field

model: str = Field()

max_tokens: None | int = Field(ge=1, le=99999999)

temperature: None | float = Field(ge=0.0, le=2.0)

top_p: None | float = Field(
    ge=0.0,
    le=1.0,
)

stop: None | str | List[str] = Field()

stream: bool = Field()

# ggml only
top_k: None | int = Field(ge=0)

# ggml only
repetition_penalty: None | float = Field(ge=0.0)

# ignore
presence_penalty: None | float = Field()
# ignore
frequency_penalty: None | float = Field()
