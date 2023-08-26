# ialacol (l-o-c-a-l-a-i)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

## Introduction

ialacol (pronounced "localai") is a lightweight drop-in replacement for OpenAI API.

It is an OpenAI API-compatible wrapper [ctransformers](https://github.com/marella/ctransformers) supporting [GGML](https://github.com/ggerganov/ggml)/[GPTQ](https://github.com/PanQiWei/AutoGPTQ) with optional CUDA/Metal acceleration.

ialacol is inspired by other similar projects like [LocalAI](https://github.com/go-skynet/LocalAI), [privateGPT](https://github.com/imartinez/privateGPT), [local.ai](https://github.com/louisgv/local.ai), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [closedai](https://github.com/closedai-project/closedai), and [mlc-llm](https://github.com/mlc-ai/mlc-llm), with a specific focus on Kubernetes deployment.

## Features

- Compatibility with OpenAI APIs, compatible with [langchain](https://github.com/hwchase17/langchain).
- Lightweight, easy deployment on Kubernetes clusters with a 1-click Helm installation.
- Streaming first! For better UX.
- Optional CUDA acceleration.
- Compatible with [Github Copilot VSCode Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot), see [Copilot](#copilot)

## Supported Models

See [Receipts](#receipts) below for instructions of deployments.

- [LLaMa 2 variants](https://huggingface.co/meta-llama)
- [OpenLLaMA variants](https://github.com/openlm-research/open_llama)
- [StarCoder variants](https://huggingface.co/bigcode/starcoder)
- [WizardCoder](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)
- [StarChat variants](https://huggingface.co/HuggingFaceH4/starchat-beta)
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b)
- [MPT-30B](https://huggingface.co/mosaicml/mpt-30b)
- [Falcon](https://falconllm.tii.ae/)

And all LLMs supported by [ctransformers](https://github.com/marella/ctransformers/tree/main/models/llms).

## Blogs

- [Containerized AI before Apocalypse 🐳🤖](https://dev.to/chenhunghan/containerized-ai-before-apocalypse-1569)
- [Deploy Llama 2 AI on Kubernetes, Now](https://dev.to/chenhunghan/deploy-llama-2-ai-on-kubernetes-now-2jc5)
- [Cloud Native Workflow for Private MPT-30B AI Apps](https://dev.to/chenhunghan/cloud-native-workflow-for-private-ai-apps-2omb)
- [Offline AI 🤖 on Github Actions 🙅‍♂️💰](https://dev.to/chenhunghan/offline-ai-on-github-actions-38a1)

## Quick Start

### Kubernetes

`ialacol` offer first class citizen support for Kubernetes, which means you can automate/configure everything compare to runing without.

To quickly get started with ialacol on Kubernetes, follow the steps below:

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install llama-2-7b-chat ialacol/ialacol
```

By defaults, it will deploy [Meta's Llama 2 Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) model quantized by [TheBloke](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML).

Port-forward

```sh
kubectl port-forward svc/llama-2-7b-chat 8000:8000
```

Chat with the default model `llama-2-7b-chat.ggmlv3.q4_0.bin` using `curl`

```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{ "messages": [{"role": "user", "content": "How are you?"}], "model": "llama-2-7b-chat.ggmlv3.q4_0.bin", "stream": false}' \
     http://localhost:8000/v1/chat/completions
```

Alternatively, using OpenAI's client library (see more examples in the `examples/openai` folder).

```sh
openai -k "sk-fake" \
     -b http://localhost:8000/v1 -vvvvv \
     api chat_completions.create -m llama-2-7b-chat.ggmlv3.q4_0.bin \
     -g user "Hello world!"
```

### Run in Container

#### Image from Github Registry

There is a [image](https://github.com/chenhunghan/ialacol/pkgs/container/ialacol) hosted on ghcr.io (alternatively [CUDA11](https://github.com/chenhunghan/ialacol/pkgs/container/ialacol-cuda11),[CUDA12](https://github.com/chenhunghan/ialacol/pkgs/container/ialacol-cuda12),[METAL](https://github.com/chenhunghan/ialacol/pkgs/container/ialacol-metal),[GPTQ](https://github.com/chenhunghan/ialacol/pkgs/container/ialacol-gptq) variants).

```sh
docker run --rm -it -p 8000:8000 \
     -e DEFAULT_MODEL_HG_REPO_ID="TheBloke/Llama-2-7B-Chat-GGML" \
     -e DEFAULT_MODEL_FILE="llama-2-7b-chat.ggmlv3.q4_0.bin" \
     ghcr.io/chenhunghan/ialacol:latest
```

#### From Source

For developers/contributors

##### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
DEFAULT_MODEL_HG_REPO_ID="TheBloke/stablecode-completion-alpha-3b-4k-GGML" DEFAULT_MODEL_FILE="stablecode-completion-alpha-3b-4k.ggmlv1.q4_0.bin" LOGGING_LEVEL="DEBUG" THREAD=4 uvicorn main:app --reload --host 0.0.0.0 --port 9999
```

##### Docker

Build image

```sh
docker build --file ./Dockerfile -t ialacol .
```

Run container

```sh
export DEFAULT_MODEL_HG_REPO_ID="TheBloke/orca_mini_3B-GGML"
export DEFAULT_MODEL_FILE="orca-mini-3b.ggmlv3.q4_0.bin"
docker run --rm -it -p 8000:8000 \
     -e DEFAULT_MODEL_HG_REPO_ID=$DEFAULT_MODEL_HG_REPO_ID \
     -e DEFAULT_MODEL_FILE=$DEFAULT_MODEL_FILE ialacol
```

## GPU Acceleration

To enable GPU/CUDA acceleration, you need to use the container image built for GPU and add `GPU_LAYERS` environment variable. `GPU_LAYERS` is determine by the size of your GPU memory. See the PR/discussion in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1412) to find the best value.

### CUDA 11

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-cuda11:latest`
- `deployment.env.GPU_LAYERS` is the layer to off loading to GPU.

### CUDA 12

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-cuda12:latest`
- `deployment.env.GPU_LAYERS` is the layer to off loading to GPU.

Only `llama`, `falcon`, `mpt` and `gpt_bigcode`(StarCoder/StarChat) support CUDA.

#### Llama with CUDA12

```sh
helm install llama2-7b-chat-cuda12 ialacol/ialacol -f examples/values/llama2-7b-chat-cuda12.yaml
```

Deploys llama2 7b model with 40 layers offloadind to GPU. The inference is accelerated by CUDA 12.

#### StarCoderPlus with CUDA12

```sh
helm install starcoderplus-guanaco-cuda12 ialacol/ialacol -f examples/values/starcoderplus-guanaco-cuda12.yaml
```

Deploys [Starcoderplus-Guanaco-GPT4-15B-V1.0 model](https://huggingface.co/LoupGarou/Starcoderplus-Guanaco-GPT4-15B-V1.0) with 40 layers offloadind to GPU. The inference is accelerated by CUDA 12.

### CUDA Driver Issues

If you see `CUDA driver version is insufficient for CUDA runtime version` when making the request, you are likely using a Nvidia Driver that is not [compatible with the CUDA version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

Upgrade the driver manually on the node (See [here](https://github.com/awslabs/amazon-eks-ami/issues/1060) if you are using CUDA11 + AMI). Or try different version of CUDA.

### Metal

To enable Metal support, use the image `ialacol-metal` built for metal.

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-metal:latest`

For example

```sh
helm install llama2-7b-chat-metal ialacol/ialacol -f examples/values/llama2-7b-chat-metal.yaml.yaml
```

### GPTQ

To use GPTQ, you must

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-gptq:latest`
- `deployment.env.MODEL_TYPE` = `gptq`

For example

```sh
helm install llama2-7b-chat-gptq ialacol/ialacol -f examples/values/llama2-7b-chat-gptq.yaml.yaml
```

```sh
kubectl port-forward svc/llama2-7b-chat-gptq 8000:8000
openai -k "sk-fake" -b http://localhost:8000/v1 -vvvvv api chat_completions.create -m gptq_model-4bit-128g.safetensors -g user "Hello world!"
```

## Tips

### Copilot

`ialacol` can be use as a copilot client as GitHub's Copilot is almost identical API as OpenAI completion API.

However, few things need to keep in mind:

1. Copilot client sends a lenthy prompt, to include all the related context for code completion, see [copilot-explorer](https://github.com/thakkarparth007/copilot-explorer), which give heavy load on the server, if you are trying to run `ialacol` locally, opt-in `TRUNCATE_PROMPT_LENGTH` environmental variable to truncate the prompt from the beginning to reduce the workload.

2. Copilot sends request in parallel, to increase the throughput, you probably need a queue like [text-inference-batcher]([text-inference-batcher](https://github.com/ialacol/text-inference-batcher).

Start two instances of ialacol:

```bash
gh repo clone chenhunghan/ialacol && cd ialacol && python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r requirements.txt
LOGGING_LEVEL="DEBUG"
THREAD=2
DEFAULT_MODEL_HG_REPO_ID="TheBloke/stablecode-completion-alpha-3b-4k-GGML"
DEFAULT_MODEL_FILE="stablecode-completion-alpha-3b-4k.ggmlv1.q4_0.bin"
TRUNCATE_PROMPT_LENGTH=100 # optional
uvicorn main:app --host 0.0.0.0 --port 9998
uvicorn main:app --host 0.0.0.0 --port 9999
```

Start [tib](https://github.com/ialacol/text-inference-batcher), pointing to upstream ialacol instances.

```bash
gh repo clone ialacol/text-inference-batcher && cd text-inference-batcher && npm install
UPSTREAMS="http://localhost:9999,http://localhost:9999" npm start
```

Configure VSCode Github Copilot to use [tib](https://github.com/ialacol/text-inference-batcher).

```json
"github.copilot.advanced": {
     "debug.overrideEngine": "stablecode-completion-alpha-3b-4k.ggmlv1.q4_0.bin",
     "debug.testOverrideProxyUrl": "http://localhost:8000",
     "debug.overrideProxyUrl": "http://localhost:8000"
}
```

### Creative v.s. Conservative

LLMs are known to be sensitive to parameters, the higher `temperature` leads to more "randomness" hence LLM becomes more "creative", `top_p` and `top_k` also contribute to the "randomness"

If you want to make LLM be creative.

```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{ "messages": [{"role": "user", "content": "Tell me a story."}], "model": "llama-2-7b-chat.ggmlv3.q4_0.bin", "stream": false, "temperature": "2", "top_p": "1.0", "top_k": "0" }' \
     http://localhost:8000/v1/chat/completions
```

If you want to make LLM be more consistent and genereate the same result with the same input.

```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{ "messages": [{"role": "user", "content": "Tell me a story."}], "model": "llama-2-7b-chat.ggmlv3.q4_0.bin", "stream": false, "temperature": "0.1", "top_p": "0.1", "top_k": "40" }' \
     http://localhost:8000/v1/chat/completions
```

## Roadmap

- [x] Support `starcoder` model type via [ctransformers](https://github.com/marella/ctransformers), including:
  - StarChat <https://huggingface.co/TheBloke/starchat-beta-GGML>
  - StarCoder <https://huggingface.co/TheBloke/starcoder-GGML>
  - StarCoderPlus <https://huggingface.co/TheBloke/starcoderplus-GGML>
- [x] Mimic restof OpenAI API, including `GET /models` and `POST /completions`
- [ ] GPU acceleration (CUDA/METAL)
- [ ] Support `POST /embeddings` backed by huggingface Apache-2.0 embedding models such as [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [hkunlp/instructor](https://huggingface.co/hkunlp/instructor-large)
- [ ] Suuport Apache-2.0 [fastchat-t5-3b](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0)
- [ ] Support more Apache-2.0 models such as [codet5p](https://huggingface.co/Salesforce/codet5p-16b) and others listed [here](https://github.com/eugeneyan/open-llms)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chenhunghan/ialacol&type=Date)](https://star-history.com/#chenhunghan/ialacol&Date)

## Receipts

### Llama-2

Deploy [Meta's Llama 2 Chat](https://huggingface.co/meta-llama) model quantized by [TheBloke](https://huggingface.co/TheBloke).

7B Chat

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install llama2-7b-chat ialacol/ialacol -f examples/values/llama2-7b-chat.yaml
```

13B Chat

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install llama2-13b-chat ialacol/ialacol -f examples/values/llama2-13b-chat.yaml
```

70B Chat

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install llama2-70b-chat ialacol/ialacol -f examples/values/llama2-70b-chat.yaml
```

### OpenLM Research's OpenLLaMA Models

Deploy [OpenLLaMA 7B](https://github.com/openlm-research/open_llama) model quantized by [rustformers](https://huggingface.co/rustformers/open-llama-ggml).

ℹ️ This is a base model, likely only useful for text completion.

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install openllama-7b ialacol/ialacol -f examples/values/openllama-7b.yaml
```

### VMWare's OpenLlama 13B Open Instruct

Deploy [OpenLLaMA 13B Open Instruct](https://huggingface.co/VMware/open-llama-13b-open-instruct) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install openllama-13b-instruct ialacol/ialacol -f examples/values/openllama-13b-instruct.yaml
```

### Mosaic's MPT Models

Deploy [MosaicML's MPT-7B](https://www.mosaicml.com/blog/mpt-7b) model quantized by [rustformers](https://huggingface.co/rustformers). ℹ️ This is a base model, likely only useful for text completion.

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install mpt-7b ialacol/ialacol -f examples/values/mpt-7b.yaml
```

Deploy [MosaicML's MPT-30B Chat](https://www.mosaicml.com/blog/mpt-30b) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install mpt-30b-chat ialacol/ialacol -f examples/values/mpt-30b-chat.yaml
```

### Falcon Models

Deploy [Uncensored Falcon 7B](https://huggingface.co/ehartford/WizardLM-Uncensored-Falcon-7b) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install falcon-7b ialacol/ialacol -f examples/values/falcon-7b.yaml
```

Deploy [Uncensored Falcon 40B](https://huggingface.co/ehartford/WizardLM-Uncensored-Falcon-40b) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install falcon-40b ialacol/ialacol -f examples/values/falcon-40b.yaml
```

### StarCoder Models (startcoder, startchat, starcoderplus, WizardCoder)

Deploy [`starchat-beta`](https://huggingface.co/TheBloke/starchat-beta-GGML) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add starchat https://chenhunghan.github.io/ialacol
helm repo update
helm install starchat-beta ialacol/ialacol -f examples/values/starchat-beta.yaml
```

Deploy [`WizardCoder`](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0) model quantized by [TheBloke](https://huggingface.co/TheBloke).

```sh
helm repo add starchat https://chenhunghan.github.io/ialacol
helm repo update
helm install wizard-coder-15b ialacol/ialacol -f examples/values/wizard-coder-15b.yaml
```

### Pythia Models

Deploy light-weight [`pythia-70m`](https://huggingface.co/rustformers/pythia-ggml) model with only 70 millions paramters (~40MB) quantized by [rustformers](https://huggingface.co/rustformers).

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install pythia70m ialacol/ialacol -f examples/values/pythia-70m.yaml
```

### RedPajama Models

Deploy [`RedPajama` 3B](https://huggingface.co/rustformers/redpajama-3b-ggml) model

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install redpajama-3b ialacol/ialacol -f examples/values/redpajama-3b.yaml
```

### StableLM Models

Deploy [`stableLM`](https://huggingface.co/rustformers/stablelm-ggml) 7B model

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install stablelm-7b ialacol/ialacol -f examples/values/stablelm-7b.yaml
```

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pip freeze > requirements.txt
```
