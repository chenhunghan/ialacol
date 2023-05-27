from typing import Awaitable, Callable, Iterator, Union
from queue import Queue, Empty
from threading import Thread
from llm_rs.auto import AutoModel
from llm_rs.base_model import Model
from llm_rs.config import GenerationConfig, SessionConfig
from starlette.types import Send
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.responses import StreamingResponse

app = FastAPI()

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
Generate = Callable[[Sender], Awaitable[None]]

session_config = SessionConfig(threads=8)
model = AutoModel.from_pretrained(
    "Rustformers/pythia-ggml",
    model_file="pythia-70m-q4_0-ggjt.bin",
    session_config=session_config,
)


class EmptyIterator(Iterator[Union[str, bytes]]):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class ModelStreamingResponse(StreamingResponse):
    """Streaming response for model, inheritance from StreamingResponse."""

    sender: Sender
    DONE = object()

    def __init__(
        self,
        query: str,
        llmModel: Model,
        status_code: int = 200,
        media_type: str = "text/event-stream",
    ) -> None:
        super().__init__(
            content=EmptyIterator(), status_code=status_code, media_type=media_type
        )
        self.query = query
        self.llmModel = llmModel
        self.queue = Queue()

    def get_model_response(self):
        """Get model response and put it into queue."""
        generation_config = GenerationConfig(temperature=0.01)

        def callback(token: str):
            self.queue.put(token)

        self.llmModel.generate(
            self.query, callback=callback, generation_config=generation_config
        )
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
        thread = Thread(target=self.get_model_response)
        thread.start()
        while True:
            try:
                token = self.queue.get()
                if token is self.DONE:
                    break
                await send(
                    {
                        "type": "http.response.body",
                        "body": bytes(token, "utf-8"),
                        "more_body": True,
                    }
                )
            # from get_nowait()
            except Empty:
                break
        thread.join()
        # send empty body to client to close connection
        await send({"type": "http.response.body", "body": b"", "more_body": False})


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
    return ModelStreamingResponse(body.query, model)
