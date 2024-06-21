#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes
from langgraph.prebuilt import create_react_agent
import dotenv
from langgraph.checkpoint import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from wikipedia_tool import wikipedia_tool
from tavily_tool import tavily_tool


dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [wikipedia_tool, tavily_tool]

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
