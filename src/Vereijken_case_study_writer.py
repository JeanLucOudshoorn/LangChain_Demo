import os
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Load the environment
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load Vereijken vector store
vector_store = FAISS.load_local('vector_stores/vereijken.faiss', OpenAIEmbeddings(),
                                allow_dangerous_deserialization=True)

# Create retriever
retriever = vector_store.as_retriever(search_type='mmr',
                                      search_kwargs={'k': 4})

# Load Example vector store
example_vector_store = FAISS.load_local('vector_stores/vereijken.faiss', OpenAIEmbeddings(),
                                        allow_dangerous_deserialization=True)

# Create retriever
example_retriever = example_vector_store.as_retriever(search_type='mmr',
                                                      search_kwargs={'k': 1})

# Create the model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Build a prompt template
template = """
You are an expert copy writer that writes sections of a case study for a professional website of a data
consultancy firm. The case studies highlight specific data science / engineering / visualization applications that have
been developed for a client. You will receive an exact instruction, an example and information about the client company 
and application to guide you. Do as the instruction states and make extensive use of the example and additional information.

Instruction:
{instruction}

Keep the tone professional but approachable and not too dry.
Give enough detail to show that we know what we are doing, but not so much that a PhD is needed to understand.
Stay very close to the format shown in the example.

Here is an example to guide you:
{example}

Use the following information about the company and application:
{context}
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "example": example_retriever, "instruction": RunnablePassthrough()}
)

# Define company
company_name = "Vereijken Kwekerijen"

# Instructions
instructions = [
    f"Write a client profile for {company_name}",
    f"Write a section explaining the unique challenges and problems that {company_name} was facing",
    f"Write a section explaining Bright Cape's approach to solve the problems {company_name} was facing",
    f"Write a section explaining the results and the impact that the solution has had for {company_name}"
 ]

# Set up section writing chain with LCEL
section_writing_chain = setup_and_retrieval | prompt | model | output_parser

# Create an empty list for written sections
written_sections = []

# Iteratively call the LLM to the sections
for instruction in instructions:
    response = section_writing_chain.invoke(instruction)
    written_sections.append(response)

print(written_sections)


# Build a prompt template to rewrite all the sections to fit better together
template = """
You are an expert copy writer that writes sections of a case study for a professional website of a data
consultancy firm. The case studies highlight specific data science / engineering / visualization applications that have
been developed for a client. You will receive an exact instruction, an example and information about the client company 
and application to guide you. Do as the instruction states and make extensive use of the example and additional information.

Instruction:
{instruction}

Keep the tone professional but approachable and not too dry.
Give enough detail to show that we know what we are doing, but not so much that a PhD is needed to understand.
Stay very close to the format shown in the example.

Here is an example to guide you:
{example}

Use the following information about the company and application:
{context}
"""

# Build prompt from template
prompt = ChatPromptTemplate.from_template(template)

# Define the final chain
final_chain = prompt | model | output_parser
