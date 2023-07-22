# ialacol (l-o-c-a-l-a-i)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

## Introduction

ialacol (pronounced "localai") is an open-source project that provides a boring, lightweight, self-hosted, private, and commercially usable LLM streaming service.

It is built on top of  [ctransformers](https://github.com/marella/ctransformers/tree/main/models/llms).

This project is inspired by other similar projects like [LocalAI](https://github.com/go-skynet/LocalAI), [privateGPT](https://github.com/imartinez/privateGPT), [local.ai](https://github.com/louisgv/local.ai), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [closedai](https://github.com/closedai-project/closedai), and [mlc-llm](https://github.com/mlc-ai/mlc-llm), with a specific focus on Kubernetes deployment, streaming, and commercially usable LLMs.

## Supported Models

See "Receipts" below for instructions of deployments.

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

- Compatibility with OpenAI APIs, allowing you to use OpenAI's Python client or any frameworks that are built on top of OpenAI APIs such as [langchain](https://github.com/hwchase17/langchain).
- Lightweight, easy deployment on Kubernetes clusters with a 1-click Helm installation.
- Support for various commercially usable models.

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
     -d '{ "messages": [{"role": "user", "content": "How are you?"}], "model": "llama-2-7b-chat.ggmlv3.q4_0.bin"}' \
     http://localhost:8000/v1/chat/completions
```

Alternatively, using OpenAI's client library (see more examples in the `examples/openai` folder).

```python
import openai

openai.api_key = "placeholder_to_avoid_exception" # needed to avoid an exception
openai.api_base = "http://localhost:8000/v1"

chat_completion = openai.ChatCompletion.create(
  model="llama-2-7b-chat.ggmlv3.q4_0.bin",
  messages=[{"role": "user", "content": "Hello world!"}]
)

print(chat_completion.choices[0].message.content)
```

## Roadmap

- [x] Support `starcoder` model type via [ctransformers](https://github.com/marella/ctransformers), including:
  - StarChat <https://huggingface.co/TheBloke/starchat-beta-GGML>
  - StarCoder <https://huggingface.co/TheBloke/starcoder-GGML>
  - StarCoderPlus <https://huggingface.co/TheBloke/starcoderplus-GGML>
- [x] Mimic restof OpenAI API, including `GET /models` and `POST /completions`
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

Deploy [OpenLLaMA 7B](https://github.com/openlm-research/open_llama) model quantized by [rustformers](https://huggingface.co/rustformers/open-llama-ggml). ℹ️ This is a base model, likely only useful for text completion.

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

Deploy [MosaicML's MPT-7B](https://www.mosaicml.com/blog/mpt-7b) model quantized by [rustformers](https://huggingface.co/rustformers/mpt-7b-ggml). ℹ️ This is a base model, likely only useful for text completion.

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

Deploy `starchat-beta` model <https://huggingface.co/TheBloke/starchat-beta-GGML> quantized by TheBloke.

```sh
helm repo add starchat https://chenhunghan.github.io/ialacol
helm repo update
helm install starchat-beta ialacol/ialacol -f examples/values/starchat-beta.yaml
```

Deploy `WizardCoder` model <https://huggingface.co/WizardLM/WizardCoder-15B-V1.0> quantized by TheBloke.

```sh
helm repo add starchat https://chenhunghan.github.io/ialacol
helm repo update
helm install wizard-coder-15b ialacol/ialacol -f examples/values/wizard-coder-15b.yaml
```

### Pythia Models

Deploy light-weight model `pythia-70m` wih only 70 millions paramters (~40MB)

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install pythia70m ialacol/ialacol -f examples/values/pythia-70m.yaml
```

### RedPajama Models

Deploy `RedPajama` 3B model

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install redpajama-3b ialacol/ialacol -f examples/values/redpajama-3b.yaml
```

### StableLM Models

Deploy `stableLM` 7B model

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
