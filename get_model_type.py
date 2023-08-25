from request_body import ChatCompletionRequestBody, CompletionRequestBody
from get_env import get_env


def get_model_type(
    body: ChatCompletionRequestBody | CompletionRequestBody,
) -> str:
    ctransformer_model_type = "llama"
    # These are also in "starcoder" format
    # https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML
    # https://huggingface.co/TheBloke/minotaur-15B-GGML
    if (
        "star" in body.model
        or "starchat" in body.model
        or "WizardCoder" in body.model
        or "minotaur-15" in body.model
    ):
        ctransformer_model_type = "gpt_bigcode"
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
    # matching https://huggingface.co/stabilityai/stablecode-completion-alpha-3b
    if "stablecode" in body.model:
        ctransformer_model_type = "gpt_neox"
    # matching https://huggingface.co/EleutherAI/pythia-70m
    if "pythia" in body.model:
        ctransformer_model_type = "gpt_neox"

    MODE_TYPE = get_env("MODE_TYPE", "")
    if len(MODE_TYPE) > 0:
        ctransformer_model_type = MODE_TYPE
    return ctransformer_model_type