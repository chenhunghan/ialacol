# ialacol (l-o-c-a-l-a-i)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

## Introduction

ialacol (pronounced "localai") is an open-source project that provides a boring, lightweight, self-hosted, private, and commercially usable LLM streaming service. It is built on top of  [ctransformers](https://github.com/marella/ctransformers).

This project is inspired by other similar projects like [LocalAI](https://github.com/go-skynet/LocalAI), [privateGPT](https://github.com/imartinez/privateGPT), [local.ai](https://github.com/louisgv/local.ai), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [closedai](https://github.com/closedai-project/closedai), and [mlc-llm](https://github.com/mlc-ai/mlc-llm), with a specific focus on Kubernetes deployment.

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

## Features

- Compatibility with OpenAI APIs, allowing you to use any frameworks that are built on top of OpenAI APIs such as [langchain](https://github.com/hwchase17/langchain).
- Lightweight, easy deployment on Kubernetes clusters with a 1-click Helm installation.
- Streaming first! For better UX.
- Optional CUDA acceleration.

## Quick Start

To quickly get started with ialacol, follow the steps below:

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
openai -k "sk-fake" -b http://localhost:8000/v1 -vvvvv api chat_completions.create -m llama-2-7b-chat.ggmlv3.q4_0.bin -g user "Hello world!"
```

## GPU Acceleration

To enable GPU/CUDA acceleration, you need to use the container image built for GPU and add `GPU_LAYERS` environment variable. `GPU_LAYERS` is determine by the size of your GPU memory. See the PR/discussion in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1412) to find the best value.

### CUDA 11

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-cuda11:latest`
- `deployment.env.GPU_LAYERS` is the layer to off loading to GPU.

### CUDA 12

- `deployment.image` = `ghcr.io/chenhunghan/ialacol-cuda11:latest`
- `deployment.env.GPU_LAYERS` is the layer to off loading to GPU.

For example

```sh
helm install llama2-7b-chat-cuda11 ialacol/ialacol -f examples/values/llama2-7b-chat-cuda11.yaml
```

Deploys llama2 7b model with 40 layers offloadind to GPU. The inference is accelerated by CUDA 11.

## Tips

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
