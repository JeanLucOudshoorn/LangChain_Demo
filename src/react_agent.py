import os
import openai
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

# Load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the tool
tools = [PythonREPLTool()]

# Provide the instruction prompt
instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

# Define the agent
agent = create_react_agent(ChatOpenAI(model='gpt-3.5-turbo', temperature=0), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Build and train a neural network
agent_executor.invoke(
    {
        "input": """Write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 100 epochs and print every 10 epochs.
Return prediction for x = 5"""
    }
)
