# 这个文件将设计 triage 节点, 用于分类邮件, 它依赖于response中的子图
from typing import Literal

from email_agent.graph.response import response_agent
from email_agent.graph.state import RouterSchema, StateInput, OverallState
from langgraph.types import Command
from langgraph.graph import START, END, StateGraph
from email_agent.domain.models import model
from email_agent.prompts.prompts import triage_system_prompt, default_background, default_triage_instructions, \
    triage_user_prompt
from email_agent.utils.utils import parse_email, format_email_markdown

# 整体思路如下:
# 1. 构建输出结构化结果的model
model_with_structured_output = model.with_structured_output(RouterSchema)

# 2. 定义分类节点, 分为三个类别: "ignore", "notify", "respond", 跳转到不同节点去
def triage_node(state: OverallState) -> Command[Literal["response_agent", "__end__"]]:
    """Classify the email into one of three categories: ignore, notify, respond."""
    input = state["email_input"] # 这是一个字典, 需要对其中的属性进行处理
    author, to, subject, email_thread = parse_email(input)

    # 拿到这些属性后, 我们开始拼接系统提示词和用户提示词
    # 2.1 系统提示词
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    # 2.2 用户提示词
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # 2.3 将两个提示词交给模型
    result = model_with_structured_output.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # 2.4 解析模型输出, 共三种结果: "ignore", "notify", "respond", 我们仅处理"respond", 逻辑判断中只更改变量值, 最后统一构建Command对象
    classification = result.classification
    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        # Add the email to the messages
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif classification == "ignore":
        goto = END
        update = {
            "classification_decision": classification,
        }
    elif classification == "notify":
        goto = END
        update = {
            "classification_decision": classification,
        }
    else:
        raise ValueError(f"Invalid classification: {classification}")
    return Command(goto=goto, update=update)

overall_workflow = (
    StateGraph(OverallState, input=StateInput)
    .add_node("triage_node", triage_node)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_node")
)

email_assistant = overall_workflow.compile()


