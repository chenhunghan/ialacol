import openai

openai.api_key = "placeholder_to_avoid_exception" # needed to avoid an exception
openai.api_base = "http://localhost:8000/v1" # this is the public address of the ialacol server

# create a chat completion
chat_completion = openai.ChatCompletion.create(
  model="pythia-70m-q4_0.bin",
  messages=[{"role": "user", "content": "Hello world! I am using OpenAI's python client library!"}]
)

# print the chat completion
print(chat_completion.choices[0].message.content)
