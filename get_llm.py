from ctransformers import LLM, AutoModelForCausalLM
from request_body import ChatCompletionRequestBody, CompletionRequestBody
from get_env import get_env


async def get_llm(
    body: ChatCompletionRequestBody | CompletionRequestBody,
) -> LLM:
    """_summary_

    Args:
        body (ChatCompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """

    # These are also in "starcoder" format
    # https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML
    # https://huggingface.co/TheBloke/minotaur-15B-GGML
    if (
        "star" in body.model
        or "starchat" in body.model
        or "WizardCoder" in body.model
        or "minotaur-15" in body.model
    ):
        ctransformer_model_type = "starcoder"

    ctransformer_model_type = "llama"
    if "llama" in body.model:
        ctransformer_model_type = "llama"
    if "mpt" in body.model:
        ctransformer_model_type = "mpt"
    if "replit" in body.model:
        ctransformer_model_type = "replit"
    if "falcon" in body.model:
        ctransformer_model_type = "falcon"
    if "dolly" in body.model:
        ctransformer_model_type = "dolly-v2"
    if "stablelm" in body.model:
        ctransformer_model_type = "gpt_neox"

    MODELS_FOLDER = get_env("MODELS_FOLDER", "models")

    return AutoModelForCausalLM.from_pretrained(
        f"./{MODELS_FOLDER}/{body.model}", model_type=ctransformer_model_type
    )
