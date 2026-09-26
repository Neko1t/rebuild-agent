from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_101 import model, write_email

agent = create_agent(
    tools=[write_email],
    model=model,
    system_prompt="You are a helpful email assistant.",
    checkpointer=InMemorySaver(),
)
if __name__ == '__main__':
    config = {"configurable": {"thread_id": "1"}}
    result = agent.invoke({"message": [{"role": "user", "content": "Hello, I need to write an email to my friend"}]}, config=config)
    print(result)
    state = agent.get_state(config=config)
    for message in state.values["messages"]:
        message.pretty_print()


