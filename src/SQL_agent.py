import os
import openai
from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase

# Load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize LLM and agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Query the database
agent_executor.invoke(
    "Which tables are there in the data?"
)
