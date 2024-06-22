from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from prompts import easement_prompt
import dotenv

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = prompt | llm | parser

print(chain.invoke({"query": easement_prompt}))
