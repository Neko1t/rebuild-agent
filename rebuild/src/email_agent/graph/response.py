"""Build the response agent subgraph."""

from typing import Literal

from email_agent.domain.models import model
from email_agent.graph.state import OverallState
from email_agent.prompts.prompts import (
    agent_system_prompt,
    default_background,
    default_cal_preferences,
    default_response_preferences,
)
from email_agent.tools.base import get_tools, get_tools_by_name
from email_agent.tools.prompt_templates import AGENT_TOOLS_PROMPT
from langgraph.graph import END, START, StateGraph

tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# 定义有工具的模型
model_with_tools = model.bind_tools(tools)

# 开始定义子图. 这个子图包含REACT loop
# llm节点, 用于与llm通信
def llm_node(state: OverallState):
    """Use the model to decide the tool_calling."""
    # 直接将全部内容交给模型
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt.format(
                        tools_prompt=AGENT_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences)
                     },

                ]
                + state["messages"]
            )
        ]
    }

# 工具调用节点, 根据模型返回的tool_calls 内容调用对应工具并记录到OverallState的messages字段中
def tool_call_node(state: OverallState) -> OverallState:
    """Perform the requested tool calls."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}


# 路由函数, 用于判断是否需要进入tool_call_node
def tool_router(state: OverallState) -> Literal["tool_call_node", "__end__"]:
    """Route model output to a tool node or the end of the graph."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    if not tool_calls:
        return END
    if any(tool_call["name"] == "Done" for tool_call in tool_calls):
        if len(tool_calls) != 1:
            raise ValueError("Done cannot be combined with other tool calls")
        return END
    return "tool_call_node"

# 构建子图
builder = StateGraph(OverallState)

#添加节点
builder.add_node("tool_call_node", tool_call_node)
builder.add_node("llm_node", llm_node)

# 添加边
builder.add_edge(START, "llm_node")
# 条件边
builder.add_conditional_edges("llm_node", tool_router, path_map={
    "tool_call_node": "tool_call_node",
    END: END
})

builder.add_edge("tool_call_node", "llm_node")

response_agent = builder.compile()
