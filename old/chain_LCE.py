from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug, set_verbose

from dotenv import load_dotenv
import os
from typing import Iterator

load_dotenv()
# set_verbose(True)
# set_debug(True)

key = os.getenv("OPENAI_API_KEY")
print(key)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

# def upper(text:str)->str:
#     return text.upper()

def upper(text:Iterator[str])->Iterator[str]:
    for t in text:
        yield t.upper()

# If you use RunnableLambda, I don't know why, but sream is not working.
chain = prompt | model | output_parser | upper

for chunk in chain.stream({"dish":"カレー"}):
    print(chunk,end="",flush=True)


# ai_message = chain.invoke({"dish":"カレー"})
# print(ai_message)
