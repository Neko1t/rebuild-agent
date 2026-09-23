from typing import Literal
from langgraph.graph import START, END, StateGraph
from langgraph.graph import MessagesState
from email_agent.langgraph_101 import model_with_tools, write_email
from dotenv import load_dotenv
load_dotenv("../.env", override=True)

def call_llm(state: MessagesState) -> MessagesState:
    """Run LLM"""

    output = model_with_tools.invoke(state["messages"])
    return {"messages": [output]}

def run_tool(state: MessagesState):
    """Run tool"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = write_email.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["run_tool", "__end__"]:
    """Route to tool handler, or end if Done tool called"""

    # Get the last message
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message is a tool call, check if it's a Done tool call
    if last_message.tool_calls:
        print("="*10)
        print(last_message.tool_calls)
        print("=" * 10)
        return "run_tool"
    # Otherwise, we stop (reply to the user)
    return END
if __name__ == '__main__':
    workflow = StateGraph(MessagesState)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("run_tool", run_tool)
    workflow.add_edge(START, "call_llm")
    workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
    workflow.add_edge("run_tool", "call_llm")
    workflow.add_edge("call_llm", END)

    # Run the workflow
    app = workflow.compile()
    result = app.invoke({"messages": [{"role": "user",
                                       "content": "Draft a response to my boss (boss@company.ai) confirming that I want to attend Interrupt!"}]})
    for m in result["messages"]:
        m.pretty_print()

