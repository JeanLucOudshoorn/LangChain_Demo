import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI, ChatOpenAI
import openai
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

# Load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load a dataset
titanic = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)

agent = create_pandas_dataframe_agent(OpenAI(model='gpt-3.5-turbo-instruct', temperature=0),
                                      titanic,
                                      verbose=True,
                                      allow_dangerous_code=True)

agent.invoke("What is the correlation between the fare and the rate of survival?")
