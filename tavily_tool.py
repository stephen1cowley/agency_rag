from langchain_community.tools.tavily_search import TavilySearchResults
import dotenv

dotenv.load_dotenv()

tavily_tool = TavilySearchResults(
    name="tavily_search_results_json",
    description=(
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    ),
    max_results=2
)


if __name__ == "__main__":

    print(f"Name: {tavily_tool.name}")
    print(f"Description: {tavily_tool.description}")
    print(f"args schema: {tavily_tool.args}")
    print(f"returns directly?: {tavily_tool.return_direct}")

    print(tavily_tool.invoke({"query": "What is the current state of the UK election?"}))
