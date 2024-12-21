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
os.environ["OPENAI_API_KEY"] = openai_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    collection_name="japanese_companies",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

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

chain.invoke("極洋株式会社に関連する企業は？")