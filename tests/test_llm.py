"""Offline tests for the LLM client: SSE parsing, request shape, error handling.

No network access — httpx.MockTransport stands in for the backend. Live verification
against a real OpenAI-compatible server (e.g. local Ollama) is scripts/check_llm.py.
"""

import json

import httpx
import pytest

from agent.llm import (
    LLMError,
    Message,
    OpenAICompatibleLLMClient,
    Role,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)


def sse(*payloads: dict | str) -> bytes:
    """Build an SSE body from a sequence of JSON-able chunks or raw strings ("[DONE]")."""
    lines = []
    for payload in payloads:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode("utf-8")


def make_client(handler, **kwargs) -> OpenAICompatibleLLMClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test/v1")
    return OpenAICompatibleLLMClient(
        base_url="http://test/v1", api_key="key", model="test-model", _client=http_client, **kwargs
    )


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    choice = {"delta": delta, "finish_reason": finish_reason}
    return {"choices": [choice]}


# --- Message / ToolSpec serialization -------------------------------------------------


def test_user_message_to_api():
    assert Message(Role.USER, "привет").to_api() == {"role": "user", "content": "привет"}


def test_tool_result_message_to_api():
    msg = Message(Role.TOOL, content="ok", tool_call_id="call_1")
    assert msg.to_api() == {"role": "tool", "content": "ok", "tool_call_id": "call_1"}


def test_assistant_tool_call_message_to_api():
    msg = Message(Role.ASSISTANT, tool_calls=(ToolCall("call_1", "submit_booking", '{"a": 1}'),))
    assert msg.to_api() == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_booking", "arguments": '{"a": 1}'},
            }
        ],
    }


def test_tool_spec_to_api():
    spec = ToolSpec("end_call", "Ends the call.", {"type": "object", "properties": {}})
    assert spec.to_api() == {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "Ends the call.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# --- Streaming: text ---------------------------------------------------------------------


async def test_streams_text_deltas_in_order():
    body = sse(
        chunk({"role": "assistant", "content": "При"}),
        chunk({"content": "вет"}),
        chunk({}, finish_reason="stop"),
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    events = [e async for e in client.stream([Message(Role.USER, "hi")])]

    assert events == [TextDelta("При"), TextDelta("вет"), StreamEnd("stop")]


async def test_empty_content_delta_is_not_emitted():
    body = sse(chunk({"content": ""}), chunk({"content": "ok"}, finish_reason="stop"), "[DONE]")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = make_client(handler)
    events = [e async for e in client.stream([Message(Role.USER, "hi")])]

    assert events == [TextDelta("ok"), StreamEnd("stop")]


async def test_reasoning_delta_is_never_surfaced_as_text():
    body = sse(
        chunk({"reasoning": "мысли вслух...", "content": "Да"}),
        chunk({"reasoning_content": "ещё мысли"}, finish_reason="stop"),
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = make_client(handler)
    events = [e async for e in client.stream([Message(Role.USER, "hi")])]

    assert events == [TextDelta("Да"), StreamEnd("stop")]


# --- Streaming: tool calls -----------------------------------------------------------------


async def test_accumulates_tool_call_across_chunks():
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "submit_", "arguments": '{"na'},
                    }
                ]
            }
        ),
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "booking", "arguments": 'me": "Иван"}'}}
                ]
            },
            finish_reason="tool_calls",
        ),
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = make_client(handler)
    events = [e async for e in client.stream([Message(Role.USER, "hi")])]

    assert events == [
        ToolCallEvent(ToolCall("call_1", "submit_booking", '{"name": "Иван"}')),
        StreamEnd("tool_calls"),
    ]


async def test_two_parallel_tool_calls_do_not_interleave():
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "a", "arguments": "1"}},
                    {"index": 1, "id": "call_2", "function": {"name": "b", "arguments": "2"}},
                ]
            },
            finish_reason="tool_calls",
        ),
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = make_client(handler)
    events = [e async for e in client.stream([Message(Role.USER, "hi")])]

    assert set(events[:2]) == {
        ToolCallEvent(ToolCall("call_1", "a", "1")),
        ToolCallEvent(ToolCall("call_2", "b", "2")),
    }
    assert events[2] == StreamEnd("tool_calls")


# --- Request body shape -------------------------------------------------------------------


async def test_reasoning_effort_omitted_by_default():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(chunk({}, finish_reason="stop"), "[DONE]"))

    client = make_client(handler)
    [_ async for _ in client.stream([Message(Role.USER, "hi")])]

    assert "reasoning_effort" not in captured["body"]


async def test_reasoning_effort_passed_through_when_set():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(chunk({}, finish_reason="stop"), "[DONE]"))

    client = make_client(handler, reasoning_effort="none")
    [_ async for _ in client.stream([Message(Role.USER, "hi")])]

    assert captured["body"]["reasoning_effort"] == "none"


async def test_tools_omitted_when_not_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(chunk({}, finish_reason="stop"), "[DONE]"))

    client = make_client(handler)
    [_ async for _ in client.stream([Message(Role.USER, "hi")])]

    assert "tools" not in captured["body"]


async def test_tools_included_when_given():
    captured = {}
    spec = ToolSpec("end_call", "Ends the call.", {"type": "object", "properties": {}})

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(chunk({}, finish_reason="stop"), "[DONE]"))

    client = make_client(handler)
    [_ async for _ in client.stream([Message(Role.USER, "hi")], tools=[spec])]

    assert captured["body"]["tools"] == [spec.to_api()]


async def test_request_is_a_streaming_chat_completion():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(chunk({}, finish_reason="stop"), "[DONE]"))

    client = make_client(handler)
    [_ async for _ in client.stream([Message(Role.SYSTEM, "sys"), Message(Role.USER, "hi")])]

    assert captured["request"].url.path == "/v1/chat/completions"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["stream"] is True
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


# --- Errors ---------------------------------------------------------------------------------


async def test_http_error_status_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    client = make_client(handler)
    with pytest.raises(LLMError, match="500"):
        [_ async for _ in client.stream([Message(Role.USER, "hi")])]


async def test_transport_failure_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler)
    with pytest.raises(LLMError, match="could not reach"):
        [_ async for _ in client.stream([Message(Role.USER, "hi")])]


async def test_malformed_sse_chunk_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: not-json\n\n")

    client = make_client(handler)
    with pytest.raises(LLMError, match="malformed"):
        [_ async for _ in client.stream([Message(Role.USER, "hi")])]
