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

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


hypothetical_prompt = ChatPromptTemplate.from_template("""\
 次の質問に回答する一文を書いてください。

 質問: {question}
 """)

hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

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
        # print(c)


log_context_passthrough = RunnablePassthrough(func=log_context)


def log_prompt(prompt: ChatPromptTemplate):
    print("------propt------")
    # print(prompt.messages)
    return prompt


retriever = db.as_retriever()

chain = (
    {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever,
    }
    | log_context_passthrough
    | prompt
    | log_prompt
    | model
    | StrOutputParser()
)

ai_message = chain.invoke("日本で有名なIT企業を10社挙げてください")

print(ai_message)
