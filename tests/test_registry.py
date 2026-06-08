# tests/test_registry.py
from harness.tools.registry import ToolRegistry
from harness.tools.std import calc


def test_unknown_tool_with_suggestion():
    registry = ToolRegistry(tools=[calc])
    result = registry.dispatch("calculator", {"expression": "2+2"}, "call-1")
    assert result.is_error
    assert "Did you mean 'calc'?" in result.content


def test_validation_missing_required():
    registry = ToolRegistry(tools=[calc])
    result = registry.dispatch("calc", {}, "call-1")
    assert result.is_error
    assert "expression" in result.content
    assert "required" in result.content


def test_validation_wrong_type():
    registry = ToolRegistry(tools=[calc])
    result = registry.dispatch("calc", {"expression": 42}, "call-1")
    assert result.is_error
    assert "string" in result.content.lower() or "str" in result.content.lower()


def test_loop_detection():
    registry = ToolRegistry(tools=[calc])
    for i in range(3):
        result = registry.dispatch("calc", {"expression": "1+1"}, f"call-{i}")
    # the third call should be caught by the loop detector
    result = registry.dispatch("calc", {"expression": "1+1"}, "call-3")
    assert result.is_error
    assert "tool-call loop" in result.content


def test_happy_path():
    registry = ToolRegistry(tools=[calc])
    result = registry.dispatch("calc", {"expression": "2+2"}, "call-1")
    assert not result.is_error
    assert result.content == "4"
