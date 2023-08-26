from get_env import get_env


def get_model_type(
    filename: str,
) -> str:
    ctransformer_model_type = "llama"
    # These are also in "starcoder" format
    # https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML
    # https://huggingface.co/TheBloke/minotaur-15B-GGML
    if (
        "star" in filename
        or "starchat" in filename
        or "WizardCoder" in filename
        or "minotaur-15" in filename
    ):
        ctransformer_model_type = "gpt_bigcode"
    if "llama" in filename:
        ctransformer_model_type = "llama"
    if "mpt" in filename:
        ctransformer_model_type = "mpt"
    if "replit" in filename:
        ctransformer_model_type = "replit"
    if "falcon" in filename:
        ctransformer_model_type = "falcon"
    if "dolly" in filename:
        ctransformer_model_type = "dolly-v2"
    if "stablelm" in filename:
        ctransformer_model_type = "gpt_neox"
    # matching https://huggingface.co/stabilityai/stablecode-completion-alpha-3b
    if "stablecode" in filename:
        ctransformer_model_type = "gpt_neox"
    # matching https://huggingface.co/EleutherAI/pythia-70m
    if "pythia" in filename:
        ctransformer_model_type = "gpt_neox"
    # codegen family are in gptj, codegen2 isn't but not supported by ggml/ctransformer yet
    # https://huggingface.co/Salesforce/codegen-2B-multi
    # https://huggingface.co/ravenscroftj/CodeGen-2B-multi-ggml-quant
    if "codegen" in filename:
        ctransformer_model_type = "gptj"

    MODE_TYPE = get_env("MODE_TYPE", "")
    if len(MODE_TYPE) > 0:
        ctransformer_model_type = MODE_TYPE
    return ctransformer_model_type
