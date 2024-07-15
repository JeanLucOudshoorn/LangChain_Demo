import asyncio
from ollama import AsyncClient
import ollama

# Regular
# response = ollama.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue? Explain in maximum five words',
#   },
# ])
# print(response['message']['content'])


# Streaming
async def chat():
    message = {'role': 'user', 'content': 'Write a Haiku about the beauty of sunflowers'}
    async for part in await AsyncClient().chat(model='llama3', messages=[message], stream=True):
        print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
