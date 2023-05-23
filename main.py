from typing import Optional, Awaitable, Callable, Iterator, Union
from llm_rs import Mpt
from starlette.types import Send
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


app = FastAPI()

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]

model = Mpt("cache/mpt-7b-storywriter-q4_0.bin")

class EmptyIterator(Iterator[Union[str, bytes]]):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class ModelStreamingResponse(StreamingResponse):
    """Streaming response for model, inheritance from StreamingResponse."""

    sender: Sender

    def __init__(
        self,
        generate: Generate,
        status_code: int = 200,
        media_type: str = "text/event-stream",
    ) -> None:
        super().__init__(
            content=EmptyIterator(), status_code=status_code, media_type=media_type
        )
        self.generate = generate

    async def stream_response(self, send: Send) -> None:
        """Rewrite stream_response to send response to client."""
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        async def sender(chunk: Union[str, bytes]):
            if not isinstance(chunk, bytes):
                chunk = chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        self.sender = sender
        self.generate(self.sender)

        # send empty body to client to close connection
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def get_generate(query: str) -> Generate:
    """_summary_

    Args:
        query (str): query to generate llm response

    Returns:
        Generate: _description_
    """

    async def generate(sender: Sender):
        def callback(token: str) -> Optional[bool]:
            sender(token)

        model.generate(query, callback=callback)

    return generate


class CompletionsRequestBody(BaseModel):
    """Request body for streaming."""

    query: str


@app.get("/ping")
async def ping():
    """_summary_

    Returns:
        _type_: pong!
    """
    return {"ialacol": "pong"}


@app.post("/chat/completions")
async def completions(body: CompletionsRequestBody) -> StreamingResponse:
    """_summary_

    Args:
        body (CompletionsRequestBody): parsed request body

    Returns:
        StreamingResponse: streaming response
    """
    return ModelStreamingResponse(get_generate(body.query))
