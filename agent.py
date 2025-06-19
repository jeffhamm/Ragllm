from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API Keys or configuration values

load_dotenv()

class AgentState(TypedDict): 
    messages: List[HumanMessage]

llm = ChatVertexAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None,
   
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


user_input = input("Enter: ")

while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
    