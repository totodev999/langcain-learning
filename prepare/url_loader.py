from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.globals import set_debug, set_verbose
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os
import numpy as np
import datetime
import time

load_dotenv()
set_verbose(True)
set_debug(True)

openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    collection_name="japanese_companies",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

with open("./wikipedia_japan_prime_market.txt", "r") as f:
    urls = f.readlines()

target_links = []
for url in urls:
    edited_url = url.replace("\n", "")
    target_links.append(edited_url)


print(f"{datetime.datetime.now()}start loading data")
data = UnstructuredURLLoader(urls=target_links).load()
print(f"{datetime.datetime.now()}finish loading data")


splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","。",""], chunk_size=4000 ,chunk_overlap=800)
texts = splitter.split_documents(data)
print(f"splitted {len(texts)} texts")

def split_list_by_max_size(lst, max_size):
    return [lst[i:i + max_size] for i in range(0, len(lst), max_size)]

max_size = 100
splitted_texts = split_list_by_max_size(texts, max_size)

for i, splitted_text in enumerate(splitted_texts):
    print(f"Processing {i+1}/{len(splitted_texts)}")
    if len(splitted_text) > 0:
        db.add_documents(splitted_text)
        time.sleep(60)
    
