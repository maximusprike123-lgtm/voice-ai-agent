"""Vendor-agnostic LLM client for a generic OpenAI-compatible Chat Completions API.

Nothing outside this module should know which backend is behind LLM_BASE_URL. The rest of
the app (DialogueEngine, tools) speaks only in terms of Message / ToolSpec / StreamEvent.
"""

import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import httpx

# Request fields the client sets itself; `extra_body` must not replace them.
RESERVED_BODY_FIELDS = frozenset({"model", "messages", "stream", "tools", "reasoning_effort"})


class LLMError(Exception):
    """Raised when the backend cannot be reached or returns something we can't use."""


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A single function call, as requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON text, exactly as the model produced it


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    # Set on an ASSISTANT message that requested tool calls (content is often None then).
    tool_calls: tuple[ToolCall, ...] | None = None
    # Set on a TOOL message: which call this is the result of.
    tool_call_id: str | None = None

    def to_api(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class TextDelta:
    """A chunk of assistant text, in the order it was generated."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """A complete tool call (arguments fully accumulated from the stream)."""

    call: ToolCall


@dataclass(frozen=True)
class StreamEnd:
    finish_reason: str


StreamEvent = TextDelta | ToolCallEvent | StreamEnd


class LLMClient(Protocol):
    def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Stream one assistant turn. Raises LLMError on transport or protocol failure."""
        ...


@dataclass
class _PendingToolCall:
    """Accumulates one streamed tool call, whose id/name/arguments arrive across chunks."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def to_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)


@dataclass
class OpenAICompatibleLLMClient:
    """Talks to any server implementing POST {base_url}/chat/completions with SSE streaming."""

    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    # Optional passthrough of extra top-level request fields, merged into every request body
    # as-is (the OpenAI SDK calls this "extra_body"). For backend-specific routing options,
    # e.g. OpenRouter's {"provider": {"sort": "latency"}}. None: nothing is added. It may not
    # override the fields this client owns.
    extra_body: dict[str, Any] | None = None
    timeout_seconds: float = 60.0
    _client: httpx.AsyncClient | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        clash = sorted(RESERVED_BODY_FIELDS & set(self.extra_body or {}))
        if clash:
            raise ValueError(
                f"extra_body may not override request fields owned by the client: {clash}"
            )

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _build_body(self, messages: list[Message], tools: list[ToolSpec] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "stream": True,
        }
        if tools:
            body["tools"] = [t.to_api() for t in tools]
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            body.update(copy.deepcopy(self.extra_body))
        return body

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[StreamEvent]:
        body = self._build_body(messages, tools)
        client = self._http_client()
        owns_client = self._client is None
        try:
            async with client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code != 200:
                    error_text = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"LLM backend returned HTTP {response.status_code}: {error_text[:500]}"
                    )
                async for event in self._parse_sse(response):
                    yield event
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach LLM backend at {self.base_url}: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def _parse_sse(self, response: httpx.Response) -> AsyncIterator[StreamEvent]:
        pending_calls: dict[int, _PendingToolCall] = {}

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise LLMError(f"malformed SSE chunk from LLM backend: {payload[:200]}") from exc

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            # Some backends stream reasoning under delta["reasoning"] or
            # delta["reasoning_content"] even when the model isn't meant to think out loud
            # (or a caller left reasoning_effort unset). Never surface it as user-facing text.
            content = delta.get("content")
            if content:
                yield TextDelta(content)

            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                pending = pending_calls.setdefault(index, _PendingToolCall())
                if tc.get("id"):
                    pending.id = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    pending.name += function["name"]
                if function.get("arguments"):
                    pending.arguments += function["arguments"]

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                for pending in pending_calls.values():
                    yield ToolCallEvent(pending.to_call())
                yield StreamEnd(finish_reason)
                return
