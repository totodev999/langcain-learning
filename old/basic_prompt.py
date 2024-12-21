from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_verbose, set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


from pydantic import BaseModel, Field

import logging
import os
from dotenv import load_dotenv

class Recipe(BaseModel):
    ingredients: list[str] =Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
set_verbose(True)
set_debug(True)

key = os.getenv("OPENAI_API_KEY")
print(key)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# messages = [
#     SystemMessage("You are a helpful assistant."),
#     HumanMessage("こんにちは！私はジョンと言います"),
#     AIMessage(content="こんにちは、ジョンさん！どのようにお手伝いできますか？"),
#     HumanMessage(content="私の名前がわかりますか？"),
# ]

# ai_message = model.invoke(messages)
# print(ai_message)
# print(ai_message.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーが入力した料理のレシピを教えてください\n\n" "{format_instructions}",),
    ("human","{dish}")
])
prompt_format_instructed = prompt.partial(format_instructions=format_instructions)

prompt_value = prompt_format_instructed.invoke({"dish","カレー"})
print(prompt_value.messages[0].content)
print("-----------------")
print(prompt_value.messages[1].content)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
result = model.invoke(prompt_value)
print("result.content")
print(result.content)

recipe = output_parser.invoke(result)
print("parsed recipe")
print(type(recipe))
print(recipe)