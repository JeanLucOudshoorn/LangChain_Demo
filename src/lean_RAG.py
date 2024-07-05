import os
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredPowerPointLoader


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Load PDF
loader = UnstructuredPDFLoader("../examle_data/Notulen_Vergadering_van_eigenaars_11-03-2024.pdf")
pages = loader.load_and_split()

# Load another PDF
loader = UnstructuredPDFLoader("../examle_data/5611XD_97-20240629-071044.pdf")
pages += loader.load_and_split()

# Load a PowerPoint document
loader = UnstructuredPowerPointLoader("../examle_data/process_mining_example.pptx")
pages += loader.load_and_split()

# Create the model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create the vectorstore
vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

# Create retriever
retriever = vectorstore.as_retriever(search_type='mmr',
                                     search_kwargs={'k': 5})

# Your query
query = "What is the impact of process mining?"

# Use the retriever
relevant_documents = retriever.invoke(query)

# Build a prompt template
template = """
Answer the question below using the context:
{context}

Vraag: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

response = chain.invoke("Wa?")

print(response)
