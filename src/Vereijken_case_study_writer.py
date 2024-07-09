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
vector_store = FAISS.load_local('../vector_stores/vereijken.faiss', OpenAIEmbeddings(),
                                allow_dangerous_deserialization=True)

# Create retriever
retriever = vector_store.as_retriever(search_type='mmr',
                                      search_kwargs={'k': 4})

# Load Example vector store
example_vector_store = FAISS.load_local('../vector_stores/vereijken.faiss', OpenAIEmbeddings(),
                                        allow_dangerous_deserialization=True)

# Create retriever
example_retriever = example_vector_store.as_retriever(search_type='mmr',
                                                      search_kwargs={'k': 1})

# Create the model
model = ChatOpenAI(model="gpt-4o")

# Build a prompt template
template = """
You are an expert copy writer that writes sections of a case study for a professional website of a data
consultancy firm (Bright Cape). The case studies highlight specific data science / engineering / visualization 
applications that have been developed for a client. You will receive an exact instruction, an example and information 
about the client company and application to guide you. Do as the instruction states and make extensive use of the 
example and additional information. Do not write more than indicated in the instruction. Avoid repetition and keep the 
sections relatively concise and short.

Instruction:
{instruction}

Keep the tone professional but approachable and not too dry. Give enough detail to show that Bright Cape employs 
competent specialists, but not so much that a PhD is needed to understand. Stay very close to the format shown in the 
example. Do not explain what type of company Bright Cape is, only focus on the client company and the application. 
Only write things relevant to the current section. For example, when asked to write a client profile, only write a 
client profile; do not go into the challenges or model.

Let's think step by step.

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
    f"Write a client profile for {company_name} of no more than a few sentences, "
    f"explaining what industry they are in, what they do, and what their vision is",
    f"Write a section explaining the unique challenges and problems that {company_name} was facing of 1-2 paragraphs",
    f"Write a section explaining Bright Cape's approach to solve the problems {company_name} was facing of 2-3 paragraphs, "
    f"explain what type of model was used and why",
    f"Write a section explaining the results and the impact that the solution has had for {company_name} of 1-2 paragraphs. "
    f"Use numbers to quantify the impact, such as a percentage reduction in cost"
 ]

# Set up section writing chain with LCEL
section_writing_chain = setup_and_retrieval | prompt | model | output_parser

# Create an empty list for written sections
written_sections = []

# Iteratively call the LLM to the sections
for instruction in instructions:
    response = section_writing_chain.invoke(instruction)
    written_sections.append(response)

# View the written sections
print(written_sections)

# Define the final template
final_template = """
You are an expert copy writer that writes a full case study from separate sections for a data consultancy firm.
There are four sections in total, which are provided below in the correct order. Do not alter the content of 
the sections too much, as they are already correct. You are only allowed to delete or rewrite a sentence here and there.
 Your aim is to create a cohesive whole out of the separate sections, with a focus in particular on a natural,
 smooth transition from one section to the next. However, the sections should remain separated: keep the headers in 
 between the sections. Keep the tone professional but approachable and not too dry. Avoid repetition and keep the 
 sections relatively concise.

Please find the four sections below:

Section 1: Client Profile
{section1}

Section 2: Unique Challenges and Problems
{section2}

Section 3: Bright Cape's Approach
{section3}

Section 4: Results and Impact
{section4}

Also write a good consulting title for the piece. Here are some examples of good titles:
- An AI power play: Fueling the next wave of innovation in the energy sector
- From farm to tablet: Building a new business to solve an old challenge
- Building a next-generation carbon platform to accelerate the path to net zero
- Banking on innovation: How ING uses generative AI to put people first
"""

final_prompt = ChatPromptTemplate.from_template(final_template)

# LangChain Expressive Language chain syntax
chain = final_prompt | model | output_parser

# Print output
full_case_study = chain.invoke({'section1': written_sections[0],
                                'section2': written_sections[1],
                                'section3': written_sections[2],
                                'section4': written_sections[3]})

print(full_case_study)

# Save the string 'full_case_study' to a .txt file
with open('../results/full_case_study_4o_1.txt', 'w') as f:
    f.write(full_case_study)

print("\nThe full case study was successfully saved")
