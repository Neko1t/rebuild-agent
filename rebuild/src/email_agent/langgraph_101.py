from typing import TypedDict

from langchain.tools import tool
from langgraph.graph import START, StateGraph, END

from dotenv import load_dotenv
load_dotenv("../.env", override=True)
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking":{
            "type":"disabled"
        }
    }
)

class OverallState(TypedDict):
    request: str
    email: str

@tool
def write_email(to: str, subject: str, body: str):
    "Write an email to someone with the subject and body"
    return f"Write an email to {to} with the subject {subject} and body: {body}"

model_with_tools = model.bind_tools([write_email],  parallel_tool_calls=False)

def write_email_node(state: OverallState) -> OverallState:
    output = model_with_tools.invoke(state["request"])
    args = output.tool_calls[0]["args"]
    email = write_email.invoke(args)
    return {"email": email}


if __name__ == "__main__":
    workflow = StateGraph(OverallState)
    workflow.add_node("write_email_node", write_email_node)
    workflow.add_edge(START, "write_email_node")
    workflow.add_edge("write_email_node", END)

    graph = workflow.compile()
    graph.invoke({"request": "Draft a response to my boss (boss@company.ai) about tomorrow's meeting"})

