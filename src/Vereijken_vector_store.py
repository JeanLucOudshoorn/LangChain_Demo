import os
import glob
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredPowerPointLoader

# Load the environment
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize pages
pages = []

# Define the directory
directory = 'Vereijken_input'

# Loop over all files in the directory
for filename in glob.glob(os.path.join(directory, '*')):
    # Check the file extension
    if filename.endswith('.pdf'):
        loader = UnstructuredPDFLoader(filename)
        print(f"Processing: {filename}")
    elif filename.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(filename)
        print(f"Processing: {filename}")
    else:
        continue  # Skip files with other extensions

    # Load and split the file, then add to pages
    pages += loader.load_and_split()

print(f"Length of pages: {len(pages)}")

# Add the website
urls = [
    "https://vereijkenkwekerijen.nl/?lang=en",
]

# collect data using selenium url loader
loader = SeleniumURLLoader(urls=urls)
pages += loader.load_and_split()

print(f"Length of pages after URL loader: {len(pages)}")

# Light preprocessing
for page in pages:
    page.page_content = str(page.page_content).replace("\\n", " ")\
                                              .replace("\\t", " ")\
                                              .replace("\n", " ")\
                                              .replace("\t", " ")

print(f"Length of pages after preprocessing: {len(pages)}")

# Create the vectorstore
vectorstore = FAISS.from_documents(pages, OpenAIEmbeddings())

# Save the vectorstore
vectorstore.save_local('vector_stores/vereijken.faiss')

# Sections of example case study
sections = [
"""
Client profile
Client: Covolt
Industry: Solar energy
Process: Asset management

Covolt B.V. is a company in the renewable energy sector, where they employ their intelligent energy management system.
This system optimizes the production of solar parks and automatically offers the energy to the market. This results in
a substantial improvement in both efficiency and reliability of these parks.
""",

"""
The problem
As part of their energy management solution, Covolt aims to predict future energy production for one day-ahead and
intraday periods. This predictive capability would enable them to refine their supply bids on the market,
as the effectiveness of energy trading heavily relies on accurate predictions of the production. Within this framework,
the primary challenges lie in training forecasting models with limited historical data (± 2 years) and generating
predictions for entirely new locations that lack any historical data.
""",

"""
Approach
Bright Cape has trained multiple machine learning models to forecast energy production, utilizing both internal and
external data sources. Since weather data was indicated as the primary forecasting driver, a comparison study of
various external weather sources was conducted to select the most accurate weather APIs. Subsequently,
Bright Cape assisted in creating the foundational data infrastructure to retrieve and integrate these diverse
data sources.

Next, this infrastructure was extended by setting up dedicated training, validation, and forecasting pipelines to
ensure continuous operability on the Microsoft Azure cloud platform. These pipelines were designed with modularity
in mind, facilitating the simultaneous development, evaluation, and deployment of both one day-ahead and intraday
forecasting models. Furthermore, a roll-out strategy was implemented to enable the scalability of the models across
locations throughout the Netherlands, including entirely new sites without sufficient available data.

Feature importance techniques were used to assess the significance of each variable and to determine the most
significant predictors, including factors like irradiation and time of the year. These techniques provide valuable
insights for data scientists and business users into what the model has learned from the data and improves the
model’s transparency.
""",

"""
Solutions and added value
Bright Cape assisted in the design, development, and deployment of the forecasting pipelines and the underlying data
infrastructure. The trained models significantly outperformed both the baseline market prediction strategies for
the pilot locations. Forecasting errors were cut in halve, where hourly day-ahead forecasting errors were reduced from
3.3% to 1.5%, and 15-minute intraday forecasting errors improving from 4.7% to 2.3%. Consequently, this led to more
effective energy trading on the market and increased profit margins. Additionally, the project yielded a return on
investment in under a year as it enabled the acquisition of new clientele.

Results
ROI < 1 year
as the project enabled the acquisition of new clientele

From 3.3 to 1.5%
reduced hourly day-ahead baseline forecasting error

From 4.7 to 2.3%
reduced 15-min intra-day baseline forecasting error
"""
]

# Light preprocessing
for i in range(len(sections)):
    sections[i] = sections[i].replace("\\n", " ") \
                             .replace("\\t", " ") \
                             .replace("\n", " ") \
                             .replace("\t", " ") \

# Example vector store
example_vector_store = FAISS.from_texts(sections, OpenAIEmbeddings())

# Save the vectorstore
example_vector_store.save_local('../vector_stores/covolt_case_study_example.faiss')
