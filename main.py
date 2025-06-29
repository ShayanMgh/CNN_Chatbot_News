
from typing import Annotated
from typing_extensions import TypedDict
import os
import requests
from dotenv import load_dotenv
import nest_asyncio
import gradio as gr

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# 1. Load environment & configure Pushover
# ---------------------------------------------------------------------------
load_dotenv()
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER  = os.getenv("PUSHOVER_USER")
PUSHOVER_URL   = "https://api.pushover.net/1/messages.json"

def push(text: str):
    """Send a push notification via Pushover."""
    resp = requests.post(
        PUSHOVER_URL,
        data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": text},
        timeout=10
    )
    resp.raise_for_status()

push_tool = Tool(
    name="send_push_notification",
    func=push,
    description="Send a push notification via Pushover"
)

# ---------------------------------------------------------------------------
# 2. Setup Playwright browser tools
# ---------------------------------------------------------------------------
nest_asyncio.apply()
async_browser = create_async_playwright_browser(headless=True)
playwright_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
# Restrict to the single-input browse tools
BROWSER_TOOL_NAMES = {"navigate_browser", "extract_text"}
browser_tools = [t for t in playwright_toolkit.get_tools() if t.name in BROWSER_TOOL_NAMES]

# Combine all tools
TOOLS = browser_tools + [push_tool]

# ---------------------------------------------------------------------------
# 3. Initialize local Llama model via Ollama
# ---------------------------------------------------------------------------
llm = OllamaLLM(model="llama3.2")

# ---------------------------------------------------------------------------
# 4. Initialize a structured-chat agent for multi-input support
# ---------------------------------------------------------------------------
agent = initialize_agent(
    llm=llm,
    tools=TOOLS,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=False,
)

# ---------------------------------------------------------------------------
# 5. LangGraph flow: wrap agent in a graph
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

async def chatbot_node(state: State):
    user_msg = state["messages"][-1]["content"]
    reply = agent.run(user_msg)
    return {"messages": [{"role": "assistant", "content": reply}]}

builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "chatbot")

memory_ckpt = MemorySaver()
CHAT_GRAPH = builder.compile(checkpointer=memory_ckpt)

# ---------------------------------------------------------------------------
# 6. Gradio chat interface
# ---------------------------------------------------------------------------
config = {"configurable": {"thread_id": "10"}}

async def chat_fn(user_input: str, history):
    out = await CHAT_GRAPH.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    return out["messages"][-1].content


def main():
    gr.ChatInterface(chat_fn, type="messages").launch()

if __name__ == "__main__":
    main()

