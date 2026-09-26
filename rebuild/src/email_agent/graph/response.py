# 这个文件将构造response 子图, 用于llm通信以及工具判断调用
from typing import Literal

from email_agent.domain.models import model
from email_agent.graph.state import RouterSchema, OverallState
from langgraph.graph import START, END, StateGraph
from email_agent.prompts.prompts import agent_system_prompt, default_background, default_response_preferences, default_cal_preferences
from email_agent.tools.prompt_templates import AGENT_TOOLS_PROMPT
# 首先导入所有的tool
from rebuild.src.email_agent.tools.base import get_tools, get_tools_by_name

tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# 定义有工具的模型和输出结构化结果的模型
model_with_tools = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(RouterSchema)

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
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}


# 路由函数, 用于判断是否需要进入tool_call_node
def tool_router(state: OverallState) -> Literal["tool_call_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
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
