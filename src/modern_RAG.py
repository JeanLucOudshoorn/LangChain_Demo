# Step 1.) Data preparation
# import os
# import openai
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader

# From website
urls = [
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Snow_leopard",
]

# collect data using selenium url loader
loader = SeleniumURLLoader(urls=urls)
documents = loader.load()

# Step 3.) Embedding and vector store
document_list = []
for doc in documents:
    d = str(doc.page_content).replace("\\n", " ").replace("\\t", " ").replace("\n", " ").replace("\t", " ")
    document_list.append(d)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = SemanticChunker(embedding_function)
docs = text_splitter.create_documents(document_list)

# From PDF
loader = UnstructuredPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

# Create the model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create the vectorstore
vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())

# docs = vectorstore.similarity_search("How will the community be engaged?", k=2)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

retriever = vectorstore.as_retriever()

# Build a prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

chain.invoke("where did harrison work?")


# # storing embeddings in a folder
# vector_store = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
#
# # use this to load vector database
# vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
#
# # Step 4.) Setting up the retrieval Q&A chain
# PROMPT_TEMPLATE = """
# Go through the context and answer given question strictly based on context.
# Context: {context}
# Question: {question}
# Answer:
# """
#
# # Set up the chain
# qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
#     )
#
# # Get the answer
# result = qa_chain({"query": query})

