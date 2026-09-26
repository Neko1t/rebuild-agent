"""Offline end-to-end tests for the migrated email assistant dataset."""

import importlib
import json
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from email_agent.graph.state import RouterSchema


DATASET_PATH = Path(__file__).parents[2] / "evals" / "dataset.jsonl"


def load_dataset():
    """Load the reference email examples from JSONL."""
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StubStructuredModel:
    """Return the expected triage classification."""

    def __init__(self, classification):
        """Store the classification and calls."""
        self.classification = classification

    def invoke(self, messages):
        """Return a deterministic structured response."""
        return RouterSchema(
            reasoning="offline integration test",
            classification=self.classification,
        )


class StubResponseModel:
    """Return one tool call at a time from the expected sequence."""

    def __init__(self, tool_names):
        """Build deterministic AI messages for the requested tools."""
        self.responses = [
            self._message(tool_name, index)
            for index, tool_name in enumerate(tool_names)
        ]
        self.calls = []

    @staticmethod
    def _message(tool_name, index):
        """Create a valid tool call for a known rebuild tool."""
        normalized_name = "Done" if tool_name.lower() == "done" else tool_name
        arguments = {
            "check_calendar_availability": {"day": "Tuesday"},
            "schedule_meeting": {
                "attendees": ["Lance Martin"],
                "subject": "Offline test meeting",
                "duration_minutes": 45,
                "preferred_day": datetime(2025, 4, 15),
                "start_time": 14,
            },
            "write_email": {
                "to": "lance@company.com",
                "subject": "Offline test",
                "content": "This is an offline test email.",
            },
            "Done": {"done": True},
        }[normalized_name]
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": normalized_name,
                    "args": arguments,
                    "id": f"offline-call-{index}",
                }
            ],
        )

    def invoke(self, messages):
        """Record the prompt and return the next response."""
        self.calls.append(list(messages))
        return self.responses.pop(0)


def extract_tool_names(messages):
    """Extract tool call names from LangChain messages or dictionaries."""
    names = []
    for message in messages:
        tool_calls = (
            message.get("tool_calls", [])
            if isinstance(message, dict)
            else getattr(message, "tool_calls", [])
        )
        names.extend(call["name"].lower() for call in tool_calls)
    return names


@pytest.fixture
def graph_modules(monkeypatch):
    """Import the parent graph and response subgraph for monkeypatching."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    triage_module = importlib.import_module("email_agent.graph.triage")
    response_module = importlib.import_module("email_agent.graph.response")
    return triage_module, response_module


@pytest.mark.parametrize(
    "case",
    load_dataset(),
    ids=lambda case: case["id"],
)
def test_reference_dataset_runs_through_parent_graph(
    graph_modules,
    monkeypatch,
    case,
):
    """Verify triage routing and expected tool-call order end to end."""
    triage_module, response_module = graph_modules
    triage_model = StubStructuredModel(case["expected_classification"])
    response_model = StubResponseModel(case["expected_tool_calls"])
    monkeypatch.setattr(triage_module, "model_with_structured_output", triage_model)
    monkeypatch.setattr(response_module, "model_with_tools", response_model)

    result = triage_module.email_assistant.invoke(
        {"email_input": case["email_input"]}
    )

    assert result["classification_decision"] == case["expected_classification"]
    assert extract_tool_names(result.get("messages", [])) == [
        tool_name.lower() for tool_name in case["expected_tool_calls"]
    ]
    if case["expected_classification"] == "respond":
        assert response_model.calls
    else:
        assert response_model.calls == []
