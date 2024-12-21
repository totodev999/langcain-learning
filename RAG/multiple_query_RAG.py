from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.globals import set_debug, set_verbose
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


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

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., title="検索クエリのリスト")


query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。
                                                           
質問: {question}
""")


def format_queries(queries: QueryGenerationOutput):
    return queries.queries


query_generation__chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | format_queries
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
        # print(c)


log_context_passthrough = RunnablePassthrough(func=log_context)


def log_prompt(prompt: ChatPromptTemplate):
    print("------propt------")
    # print(prompt.messages)
    return prompt


class MyRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        super().__init__()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"[MyRetriever] Starting retrieval for query: {query}")
        documents = db.similarity_search(query=query)

        return documents


retriever = MyRetriever()


chain = (
    {
        "question": RunnablePassthrough(),
        "context": query_generation__chain | retriever.map(),
    }
    | log_context_passthrough
    | prompt
    | log_prompt
    | model
    | StrOutputParser()
)

ai_message = chain.invoke("日本で有名なIT企業を10社挙げてください")

print(ai_message)
