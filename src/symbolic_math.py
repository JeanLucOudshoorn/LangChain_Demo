import os
import openai
from dotenv import load_dotenv
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain_openai import OpenAI

# Load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create LLMSymbolicMathChain base on SymPy
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)

# Integrals and derivatives
result = llm_symbolic_math.invoke("What is the derivative of sin(x)*exp(x) with respect to x?")
print(result['answer'])

# Solving equations
# llm_symbolic_math.invoke("What are the solutions to this equation y^3 + 1/3y?")
