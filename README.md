# ialacol (reads: localai)

[![Docker Repository on Quay](https://quay.io/repository/chenhunghan/ialacol/status "Docker Repository on Quay")](https://quay.io/repository/chenhunghan/ialacol)

🦄 Self hosted, 🔒 private, 🐟 scalable, 🤑 commercially usable, 💬 LLM chat streaming service with 1-click Kubernetes cluster installation on any cloud

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

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pip freeze > requirements.txt
```
