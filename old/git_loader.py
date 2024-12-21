from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.globals import set_debug, set_verbose
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
set_verbose(True)
set_debug(True)

openai_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

def file_filter(file_path:str)->bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

vector_store.add_documents(documents)