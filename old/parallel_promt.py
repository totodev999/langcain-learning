from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.globals import set_debug, set_verbose


from langchain_openai import ChatOpenAI
import pprint
from dotenv import load_dotenv
import os
from typing import Iterator

load_dotenv()
set_verbose(True)
set_debug(True)

key = os.getenv("OPENAI_API_KEY")
print(key)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

optimist_prompt = ChatPromptTemplate.from_messages([
    ("system","あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください"),
    ("human","{topic}")
])

optimistic_chain = optimist_prompt | model | output_parser

pessimistic_prompt = ChatPromptTemplate.from_messages([
    ("system","あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください"),
    ("human","{topic}")
])

pessimistic_chain = pessimistic_prompt | model | output_parser

parallel_chain = RunnableParallel({
    "optimistic_opinion":optimistic_chain,
    "pessimistic_opinion":pessimistic_chain
})
output = parallel_chain.invoke({"topic": "生成AIの進化について"})
pprint.pprint(output)