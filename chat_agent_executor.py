import json
from typing import Annotated, Callable, Optional, Sequence, TypedDict, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from langgraph._api.deprecation import deprecated
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt.tool_node import ToolNode


# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep


def create_react_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool]],
    messages_modifier: Optional[Union[SystemMessage, str, Callable, Runnable]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools or a ToolExecutor instance.
        messages_modifier: An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.
            Can take a few different forms:
            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
        checkpointer: An optional checkpoint saver object. This is useful for persisting
            the state of the graph (e.g., as chat memory).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "agent", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "agent", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.

    Returns:
        A compiled LangChain runnable that can be used for chat interactions.
    """

    if isinstance(tools, ToolExecutor):
        tool_classes = tools.tools
    else:
        tool_classes = tools
    model = model.bind_tools(tool_classes)

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    messages_modifier = "You are a helpful assistant."
    _system_message: BaseMessage = SystemMessage(content=messages_modifier)
    model_runnable = (lambda messages: [_system_message] + messages) | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        messages = state["messages"]
        response = model_runnable.invoke(messages, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = await model_runnable.ainvoke(messages, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


# Keep for backwards compatibility
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "create_function_calling_executor",
    "AgentState",
]
