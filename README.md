# ialacol (reads: localai)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

🦄 Self hosted, 🔒 private, 🐟 scalable, 🤑 commercially usable, 💬 LLM chat streaming service with 1-click Kubernetes cluster installation on any cloud

```sh
helm repo add ialacol https://chenhunghan.github.io/ialacol
helm repo update
helm install mpt7b ialacol/ialacol # named the release as `mpt7b` because by default the helm chart will install mpt7b model.
```

This project is inspired by [LocalAI](https://github.com/go-skynet/LocalAI), [privateGPT](https://github.com/imartinez/privateGPT), [local.ai](https://github.com/louisgv/local.ai), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [closedai](https://github.com/closedai-project/closedai), [closedai](https://github.com/closedai-project/closedai), [mlc-llm](https://github.com/mlc-ai/mlc-llm), but with a focus on Kubernetes deployment, streaming, and commercially usable models only.

Esssentially `ialacol` is a OpenAI RESTful API-compatible HTTP interface built on top of the great projects [llm-rs-python](https://github.com/LLukas22/llm-rs-python) and [llm](https://github.com/rustformers/llm), we aims to support all [known-good-models](https://github.com/rustformers/llm/blob/main/doc/known-good-models.md) supported by `llm-rs`.

## Compatible with OpenAI APIs

We aim at covering all public [OpenAI APIs](https://platform.openai.com/docs/api-reference), therefore you can use `curl`

```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{ "messages": [{"role": "user", "content": "How are you?"}], "stream": true, "model": "pythia-70m-q4_0.bin"}' \
     http://localhost:8000/v1/chat/completio
```

or OpenAI python client (see more examples in the `examples/openai` folder)

```python
import openai

openai.api_key = "placeholder_to_avoid_exception" # needed to avoid an exception
openai.api_base = "http://localhost:8000/v1" # this is the public address of the ialacol server

chat_completion = openai.ChatCompletion.create(
  model="pythia-70m-q4_0.bin", # the model filename in the env.MODELS_FOLDER directory
  messages=[{"role": "user", "content": "Hello world! I am using OpenAI's python client library!"}]
)

print(chat_completion.choices[0].message.content)
```

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
