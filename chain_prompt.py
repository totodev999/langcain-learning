from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug, set_verbose

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

class Recipe(BaseModel):
    ingredients:list[str] = Field(description="ingredients of the dish")
    steps:list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)

load_dotenv()
set_verbose(True)
set_debug(True)

key = os.getenv("OPENAI_API_KEY")
print(key)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system","ユーザーが入力した料理のレシピを教えてください\n\n{format_instructions}"),
    ("human","{dish}")
])
prompt_with_instructions = prompt.partial(format_instructions=output_parser.get_format_instructions())


model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(response_format={"type":"json_object"})

chain = prompt_with_instructions | model | output_parser

output = chain.invoke({"dish":"カレー"})
print(type(output))
print(output.ingredients)
print(output.steps)

