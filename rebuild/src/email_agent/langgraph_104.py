from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt



class OverallState(TypedDict):
    input: str
    user_feedback: str

def node1(state: OverallState) -> OverallState:
    print("node1")
    pass

def node3(state: OverallState) -> OverallState:
    print("node3")
    pass

def human_feedback(state: OverallState) -> OverallState:
    print("human_feedback")
    feedback = interrupt("Please provide feedback:")
    return {"user_feedback": feedback}

if __name__ == '__main__':
    graph = StateGraph(OverallState)
    graph.add_node("node1", node1)
    graph.add_node("human_feedback", human_feedback)
    graph.add_node("node3", node3)
    graph.add_sequence("node1", "human_feedback", "node3")

    config = {"configurable": {"thread_id": "1"}}
    memory = InMemorySaver()
    graph.compile(checkpointer=memory)


