from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from prompts import easement_prompt
import dotenv

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# json_schema = {
#     "title": "joke",
#     "description": "Joke to tell user.",
#     "type": "object",
#     "properties": {
#         "setup": {
#             "type": "string",
#             "description": "The setup of the joke",
#         },
#         "punchline": {
#             "type": "string",
#             "description": "The punchline to the joke",
#         },
#         "rating": {
#             "type": "integer",
#             "description": "How funny the joke is, from 1 to 10",
#         },
#     },
#     "required": ["setup", "punchline"],
# }
# structured_llm = llm.with_structured_output(json_schema)

# structured_llm.invoke("Tell me a joke about cats")


json_schema = {
    "title": "decision_tree",
    "description": "Decision tree for determining if a case qualifies as an easement in England and Wales case law.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "object",
            "description": "The tree",
        },
    },
    "required": ["setup"],
}


structured_llm = llm.with_structured_output(json_schema)

print(structured_llm.invoke(easement_prompt))
