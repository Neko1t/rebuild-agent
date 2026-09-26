"""Offline triage tests migrated from the reference evaluation dataset."""

import importlib
import json
from pathlib import Path

import pytest
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
    """Return one deterministic structured classification."""

    def __init__(self, classification):
        """Store the classification and calls for later assertions."""
        self.classification = classification
        self.calls = []

    def invoke(self, messages):
        """Record the prompt and return the configured classification."""
        self.calls.append(messages)
        return RouterSchema(
            reasoning="offline test classification",
            classification=self.classification,
        )


@pytest.fixture
def triage_module(monkeypatch):
    """Import the triage graph without making a live model request."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    return importlib.import_module("email_agent.graph.triage")


@pytest.mark.parametrize(
    "case",
    load_dataset(),
    ids=lambda case: case["id"],
)
def test_reference_dataset_routes_each_email(triage_module, monkeypatch, case):
    """Match every reference classification to the expected graph route."""
    stub_model = StubStructuredModel(case["expected_classification"])
    monkeypatch.setattr(triage_module, "model_with_structured_output", stub_model)

    command = triage_module.triage_node({"email_input": case["email_input"]})

    expected_goto = (
        "response_agent"
        if case["expected_classification"] == "respond"
        else "__end__"
    )
    assert command.goto == expected_goto
    assert command.update["classification_decision"] == case["expected_classification"]
    assert len(stub_model.calls) == 1
    assert case["email_input"]["subject"] in stub_model.calls[0][1]["content"]

    if case["expected_classification"] == "respond":
        assert command.update["messages"][0]["role"] == "user"
        assert case["email_input"]["subject"] in command.update["messages"][0]["content"]
    else:
        assert "messages" not in command.update
