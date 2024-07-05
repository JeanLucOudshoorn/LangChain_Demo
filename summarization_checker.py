import os
import openai
from dotenv import load_dotenv
from langchain.chains import LLMSummarizationCheckerChain
from langchain_openai import OpenAI

# Load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Summarization checker chain
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
text = """
Your 9-year old might like these recent discoveries made by The James Webb Space Telescope (JWST):
• In 2023, The JWST spotted a number of galaxies nicknamed "green peas." They were given this name because they are small, round, and green, like peas.
• The telescope captured images of galaxies that are over 13 billion years old. This means that the light from these galaxies has been traveling for over 13 billion years to reach us.
• JWST took the very first pictures of a planet outside of our own solar system. These distant worlds are called "exoplanets." Exo means "from outside."
These discoveries can spark a child's imagination about the infinite wonders of the universe."""

result = checker_chain.invoke(text)

print(result)
