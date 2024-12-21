from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.globals import set_debug, set_verbose
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

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


def log_context(context):
    # print("------contextAll------")
    # print(context["context"])
    for c in context["context"]:
        print("------contextMin------")
        print(c)


log_context_passthrough = RunnablePassthrough(func=log_context)


def log_prompt(prompt: ChatPromptTemplate):
    print("------propt------")
    print(prompt.messages)
    return prompt


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | log_context_passthrough
    | prompt
    | log_prompt
    | model
    | StrOutputParser()
)

ai_message = chain.invoke("極洋株式会社に関連する企業は？")

print(ai_message)
