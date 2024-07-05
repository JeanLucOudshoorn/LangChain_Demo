from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Instantiate LLM and prompt
llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_template("Tell me a short joke about a {job} going to {place}")

# LangChain Expressive Language chain syntax
chain = prompt | llm | StrOutputParser()

# Print output
print(chain.invoke({'job': 'chef', 'place': 'France'}))
