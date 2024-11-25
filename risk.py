import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import matplotlib.pyplot as plt

#Loading the CSV

file_path = ("/Users/vichu/Downloads/comprehensive_mutual_funds_data.csv")
loader = CSVLoader(file_path=file_path)
data = loader.load()


