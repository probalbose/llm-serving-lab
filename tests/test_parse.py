import json

import pytest

from serving_lab.client import parse_sse_line


def sse(payload: str) -> str:
    """Wrap a payload as one SSE (Server-Sent Events) line, the way the server sends it."""
    return "data: " + payload

def test_content_line_returns_text():
    line = sse('{"choices":[{"delta":{"content":"KV"}}]}')
    assert parse_sse_line(line) == "KV"

def test_malformed_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_sse_line(sse("{not json}"))

def test_role_only_line_returns_none() -> None:
    line = sse('{"choices":[{"delta":{"role":"assistant","content":null}}]}')
    assert parse_sse_line(line) is None

def test_done_returns_none() -> None:
    assert parse_sse_line(sse("[DONE]")) is None

def test_blank_line_returns_none() -> None:
    assert parse_sse_line("") is None