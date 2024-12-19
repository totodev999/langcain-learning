from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.globals import set_debug, set_verbose
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma


from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
set_verbose(True)
set_debug(True)

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")


os.environ["OPENAI_API_KEY"] = openai_key
os.environ["TAVILY_API_KEY"] = tavily_key


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

chain = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | prompt | model | StrOutputParser()

chain.invoke("LangChainの概要を教えて")