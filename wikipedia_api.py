from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up on wikipedia"
    )

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=10000, load_all_available_meta=True)

wikipedia_tool = WikipediaQueryRun(
    name="wikipedia",
    description=(
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    ),
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=False,
)


print(f"Name: {wikipedia_tool.name}")
print(f"Description: {wikipedia_tool.description}")
print(f"args schema: {wikipedia_tool.args}")
print(f"returns directly?: {wikipedia_tool.return_direct}")


docs = wikipedia_tool.invoke({"query": "Arithmetic coding"})

