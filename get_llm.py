import os

from ctransformers import LLM, AutoModelForCausalLM
from request_body import ChatCompletionRequestBody, CompletionRequestBody
from get_auto_config import get_auto_config
from log import log

async def get_llm(
    body: ChatCompletionRequestBody | CompletionRequestBody,
) -> LLM:
    """_summary_

    Args:
        body (ChatCompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """

    auto_config = get_auto_config(body)

    log.debug("Creating LLM instance...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=f"{os.getcwd()}/models/{body.model}",
        local_files_only=True,
        config=auto_config,
    )

    return llm
