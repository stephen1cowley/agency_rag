#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
import dotenv
from langgraph.checkpoint import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



search = TavilySearchResults(max_results=2)


### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool, search]

system_message = "You are a helpful assistant with access to two tools."
agent_executor = create_react_agent(llm, tools, checkpointer=MemorySaver(), messages_modifier=system_message)


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


add_routes(
    app,
    agent_executor,
    path="/my_runnable",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
