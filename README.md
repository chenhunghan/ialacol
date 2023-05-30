# ialacol (l-o-c-a-l-a-i)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

🦄 Self-hosted, 🔒 private, 🐟 scalable, 🤑 commercially usable, 💬 LLM chat streaming service with 1-click Kubernetes cluster installation on premises or public clouds.

## Introduction

ialacol (pronounced "localai") is an open-source project that provides a self-hosted, private, and commercially usable chat streaming service. It is built on top of the great projects [llm-rs-python](https://github.com/LLukas22/llm-rs-python) and [llm](https://github.com/rustformers/llm) and aims to support all [known-good-models](https://github.com/rustformers/llm/blob/main/doc/known-good-models.md) supported by `llm-rs`. This project is inspired by other similar projects like [LocalAI](https://github.com/go-skynet/LocalAI), [privateGPT](https://github.com/imartinez/privateGPT), [local.ai](https://github.com/louisgv/local.ai), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [closedai](https://github.com/closedai-project/closedai), and [mlc-llm](https://github.com/mlc-ai/mlc-llm), with a specific focus on Kubernetes deployment, streaming, and commercially usable models.

## Features

- Compatibility with OpenAI APIs, allowing you to use OpenAI's Python client or any frameworks that are built on top of OpenAI APIs such as [langchain](https://github.com/hwchase17/langchain).
- Easy deployment on Kubernetes clusters with a 1-click Helm installation.
- Support for various commercially usable models.

## Quick Start

To quickly get started with ialacol, follow the steps below:

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install mpt7b ialacol/ialacol
```

By defaults, it will deploy [MosaicML's MPT-7B](https://www.mosaicml.com/blog/mpt-7b) model quantized by [rustformers](https://huggingface.co/rustformers/mpt-7b-ggml).

Port-forward
```sh
kubectl port-forward svc/mpt7b 8000:8000
```

Chat with the default model `mpt-7b-q4_0.bin` using `curl`
```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{ "messages": [{"role": "user", "content": "How are you?"}], "model": "mpt-7b-q4_0.bin"}' \
     http://localhost:8000/v1/chat/completions
```

Alternatively, using OpenAI's client library (see more examples in the `examples/openai` folder).

```python
import openai

openai.api_key = "placeholder_to_avoid_exception" # needed to avoid an exception
openai.api_base = "http://localhost:8000/v1"

chat_completion = openai.ChatCompletion.create(
  model="mpt-7b-q4_0.bin",
  messages=[{"role": "user", "content": "Hello world!"}]
)

print(chat_completion.choices[0].message.content)
```

## Roadmap

- [ ] Mimic restof OpenAI API, including `GET /models` and `POST /completions`
- [ ] Support `POST /embeddings` backed by huggingface Apache-2.0 embedding models such as [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [hkunlp/instructor](https://huggingface.co/hkunlp/instructor-large)
- [ ] Suuport Apache-2.0 [fastchat-t5-3b](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0)
- [ ] Support more Apache-2.0 models such as [codet5p](https://huggingface.co/Salesforce/codet5p-16b) and others listed [here](https://github.com/eugeneyan/open-llms)

## Receipts

Deploy light-weight model `pythia-70m` wih only 70 millions paramters (~40MB)

```sh
cat > values.yaml <<EOF
replicas: 1
deployment:
  image: quay.io/chenhunghan/ialacol:latest
  env:
    DEFAULT_MODEL_HG_REPO_ID: rustformers/pythia-ggml
    DEFAULT_MODEL_FILE: pythia-70m-q4_0.bin
    DEFAULT_MODEL_META: pythia-70m-q4_0.meta
resources:
  {}
cache:
  persistence:
    size: 1Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
cacheMountPath: /app/cache
model:
  persistence:
    size: 1Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
modelMountPath: /app/models
service:
  type: ClusterIP
  port: 80
  annotations: {}
nodeSelector: {}
tolerations: []
affinity: {}
EOF
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install pythia70m ialacol/ialacol -f values.yaml
```

Deploy `RedPajama` 3B model

```sh
cat > values.yaml <<EOF
replicas: 1
deployment:
  image: quay.io/chenhunghan/ialacol:latest
  env:
    DEFAULT_MODEL_HG_REPO_ID: rustformers/redpajama-ggml
    DEFAULT_MODEL_FILE: RedPajama-INCITE-Base-3B-v1-q5_1-ggjt.bin
    DEFAULT_MODEL_META: RedPajama-INCITE-Base-3B-v1-q5_1-ggjt.meta
resources:
  {}
cache:
  persistence:
    size: 5Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
cacheMountPath: /app/cache
model:
  persistence:
    size: 5Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
modelMountPath: /app/models
service:
  type: ClusterIP
  port: 80
  annotations: {}
nodeSelector: {}
tolerations: []
affinity: {}
EOF
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install redpajama3B ialacol/ialacol -f values.yaml
```

Deploy `stableLM` 7B model

```sh
cat > values.yaml <<EOF
replicas: 1
deployment:
  image: quay.io/chenhunghan/ialacol:latest
  env:
    DEFAULT_MODEL_HG_REPO_ID: rustformers/stablelm-ggml
    DEFAULT_MODEL_FILE: stablelm-tuned-alpha-7b-q4_0.bin
    DEFAULT_MODEL_META: stablelm-tuned-alpha-7b-q4_0.meta
resources:
  {}
cache:
  persistence:
    size: 10Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
cacheMountPath: /app/cache
model:
  persistence:
    size: 10Gi
    accessModes:
      - ReadWriteOnce
    storageClass: ~
modelMountPath: /app/models
service:
  type: ClusterIP
  port: 80
  annotations: {}
nodeSelector: {}
tolerations: []
affinity: {}
EOF
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install stablelm7B ialacol/ialacol -f values.yaml
```

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pip freeze > requirements.txt
```
