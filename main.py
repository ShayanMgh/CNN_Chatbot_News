from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain.agents import Tool
from langchain_community.llms import Ollama
from langgraph.checkpoint.memory import MemorySaver


# -- your push notification tool stays the same --
load_dotenv()
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user  = os.getenv("PUSHOVER_USER")
pushover_url   = "https://api.pushover.net/1/messages.json"

def push(text: str):
    """Send a push notification to the user"""
    requests.post(
        pushover_url,
        data={
            "token": pushover_token,
            "user": pushover_user,
            "message": text
        }
    )

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Useful for when you want to send a push notification"
)

# -- async playwright setup unchanged --
import nest_asyncio
nest_asyncio.apply()

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
tool_dict = {t.name: t for t in tools}

navigate_tool     = tool_dict["navigate_browser"]
extract_text_tool = tool_dict["extract_text"]

# example usage
async def demo():
    await navigate_tool.arun({"url": "https://www.cnn.com"})
    text = await extract_text_tool.arun({})
    print(text[:500])

import asyncio
asyncio.run(demo())

all_tools = tools + [tool_push]

llm = Ollama(model="llama3.2")

# LangChain LlamaCpp supports bind_tools() just like ChatOpenAI
llm_with_tools = llm.bind_tools(all_tools)

# --- rest of your state‐graph/chatbot setup unchanged ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    # invoke the Llama model (with tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)