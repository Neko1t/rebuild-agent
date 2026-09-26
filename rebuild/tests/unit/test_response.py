"""Test response subgraph routing without a live model."""

import importlib
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class StubModel:
    """Return preset model messages in order."""

    def __init__(self, responses):
        """Store responses for later invocations."""
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        """Record input messages and return the next response."""
        self.calls.append(list(messages))
        return self.responses.pop(0)


@pytest.fixture
def response_module(monkeypatch):
    """Import the response subgraph with an offline credential placeholder."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    return importlib.import_module("email_agent.graph.response")


def test_plain_response_ends_without_tools(response_module, monkeypatch):
    """End when the model returns a plain answer."""
    stub_model = StubModel([AIMessage(content="No tool is needed")])
    monkeypatch.setattr(response_module, "model_with_tools", stub_model)

    result = response_module.response_agent.invoke(
        {"messages": [HumanMessage(content="Hello")], "email_input": {}, "classification_decision": "respond"}
    )

    assert result["messages"][-1].content == "No tool is needed"
    assert len(stub_model.calls) == 1


def test_tool_result_returns_to_model(response_module, monkeypatch):
    """Pass tool output back to the model before ending."""
    stub_model = StubModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "check_calendar_availability", "args": {"day": "Monday"}, "id": "call-1"}],
            ),
            AIMessage(content="Monday has open times"),
        ]
    )
    monkeypatch.setattr(response_module, "model_with_tools", stub_model)

    result = response_module.response_agent.invoke(
        {"messages": [HumanMessage(content="Check Monday")], "email_input": {}, "classification_decision": "respond"}
    )

    tool_result = stub_model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "call-1"
    assert "Monday" in tool_result.content
    assert result["messages"][-1].content == "Monday has open times"


def test_done_call_ends(response_module, monkeypatch):
    """End when Done is the only requested tool."""
    stub_model = StubModel(
        [AIMessage(content="", tool_calls=[{"name": "Done", "args": {"done": True}, "id": "call-2"}])]
    )
    monkeypatch.setattr(response_module, "model_with_tools", stub_model)

    result = response_module.response_agent.invoke(
        {"messages": [HumanMessage(content="Finish")], "email_input": {}, "classification_decision": "respond"}
    )

    assert result["messages"][-1].tool_calls[0]["name"] == "Done"
    assert len(stub_model.calls) == 1


def test_done_cannot_skip_another_tool(response_module, monkeypatch):
    """Reject mixed Done calls before executing another tool."""
    stub_model = StubModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "Done", "args": {"done": True}, "id": "call-3"},
                    {
                        "name": "write_email",
                        "args": {"to": "boss@example.com", "subject": "Update", "content": "Hello"},
                        "id": "call-4",
                    },
                ],
            )
        ]
    )
    write_email = Mock()
    monkeypatch.setattr(response_module, "model_with_tools", stub_model)
    monkeypatch.setitem(response_module.tools_by_name, "write_email", write_email)

    with pytest.raises(ValueError, match="Done cannot be combined"):
        response_module.response_agent.invoke(
            {"messages": [HumanMessage(content="Finish and send")], "email_input": {}, "classification_decision": "respond"}
        )

    write_email.assert_not_called()
