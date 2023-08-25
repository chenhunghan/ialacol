from ctransformers import Config, AutoConfig

from request_body import ChatCompletionRequestBody, CompletionRequestBody
from get_env import get_env, get_env_or_none
from get_default_thread import get_default_thread
from get_model_type import get_model_type
from log import log

THREADS = int(get_env("THREADS", str(get_default_thread())))


def get_auto_config(
    body: CompletionRequestBody | ChatCompletionRequestBody,
) -> AutoConfig:
    # ggml only, follow ctransformers defaults
    TOP_K = int(get_env("TOP_K", "40"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
    TOP_P = float(get_env("TOP_P", "1.0"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
    TEMPERATURE = float(get_env("TEMPERATURE", "1"))
    # ggml only, follow ctransformers defaults
    REPETITION_PENALTY = float(get_env("REPETITION_PENALTY", "1.1"))
    # ggml only, follow ctransformers defaults
    LAST_N_TOKENS = int(get_env("LAST_N_TOKENS", "64"))
    # ggml only, follow ctransformers defaults
    SEED = int(get_env("SEED", "-1"))
    # ggml only, follow ctransformers defaults
    BATCH_SIZE = int(get_env("BATCH_SIZE", "8"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
    MAX_TOKENS = int(get_env("MAX_TOKENS", "9999999"))
    # OpenAI API defaults https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
    STOP = get_env_or_none("STOP")
    # ggml only, follow ctransformers defaults
    CONTEXT_LENGTH = int(get_env("CONTEXT_LENGTH", "-1"))
    # the layers to offloading to the GPU
    GPU_LAYERS = int(get_env("GPU_LAYERS", "0"))

    log.debug("TOP_K: %s", TOP_K)
    log.debug("TOP_P: %s", TOP_P)
    log.debug("TEMPERATURE: %s", TEMPERATURE)
    log.debug("REPETITION_PENALTY: %s", REPETITION_PENALTY)
    log.debug("LAST_N_TOKENS: %s", LAST_N_TOKENS)
    log.debug("SEED: %s", SEED)
    log.debug("BATCH_SIZE: %s", BATCH_SIZE)
    log.debug("THREADS: %s", THREADS)
    log.debug("MAX_TOKENS: %s", MAX_TOKENS)
    log.debug("STOP: %s", STOP)
    log.debug("CONTEXT_LENGTH: %s", CONTEXT_LENGTH)
    log.debug("GPU_LAYERS: %s", GPU_LAYERS)

    top_k = body.top_k if body.top_k else TOP_K
    top_p = body.top_p if body.top_p else TOP_P
    temperature = body.temperature if body.temperature else TEMPERATURE
    repetition_penalty = (
        body.repetition_penalty if body.repetition_penalty else REPETITION_PENALTY
    )
    last_n_tokens = body.last_n_tokens if body.last_n_tokens else LAST_N_TOKENS
    seed = body.seed if body.seed else SEED
    batch_size = body.batch_size if body.batch_size else BATCH_SIZE
    threads = body.threads if body.threads else THREADS
    max_new_tokens = body.max_tokens if body.max_tokens else MAX_TOKENS
    stop = body.stop if body.stop else STOP

    log.info("top_k: %s", top_k)
    log.info("top_p: %s", top_p)
    log.info("temperature: %s", temperature)
    log.info("repetition_penalty: %s", repetition_penalty)
    log.info("last_n_tokens: %s", last_n_tokens)
    log.info("seed: %s", seed)
    log.info("batch_size: %s", batch_size)
    log.info("threads: %s", threads)
    log.info("max_new_tokens: %s", max_new_tokens)
    log.info("stop: %s", stop)

    log.info("CONTEXT_LENGTH: %s", CONTEXT_LENGTH)
    log.info("GPU_LAYERS: %s", GPU_LAYERS)

    config = Config(
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        max_new_tokens=max_new_tokens,
        stop=stop,
        context_length=CONTEXT_LENGTH,
        gpu_layers=GPU_LAYERS,
    )

    model_type = get_model_type(body)

    log.info("model_type: %s", model_type)

    auto_config = AutoConfig(
        config=config,
        model_type=model_type,
    )
    return auto_config
