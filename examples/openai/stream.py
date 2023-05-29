import openai
import time

openai.api_key = "placeholder_to_avoid_exception" # needed to avoid an exception
openai.api_base = "http://localhost:8000/v1" # this is the public address of the ialacol server

# Rest are from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
#
# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat

# record the time before the request is sent
start_time = time.time()

response = openai.ChatCompletion.create(
    model="pythia-70m-q4_0.bin", # the model filename in the env.MODELS_FOLDER directory
    messages=[
        {'role': 'user', 'content': 'Hello, I am a human.'},
    ],
    stream=True  # we set stream=True
)

# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk['choices'][0]['delta']  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")
full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
print(f"Full conversation received: {full_reply_content}")
