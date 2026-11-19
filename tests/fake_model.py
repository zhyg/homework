"""Deterministic FakeModel implementing smolagents.Model for reproducible harness testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smolagents import (
    ChatMessage,
    ChatMessageToolCall,
    MessageRole,
    Model,
    TokenUsage,
    Tool,
)
from smolagents.models import ChatMessageToolCallFunction


@dataclass
class ScriptedCall:
    tool_name: str
    arguments: Any
    tool_call_id: str | None = None
    thought: str = ""
    input_tokens: int = 120
    output_tokens: int = 45


class FakeModel(Model):
    """Scripted smolagents Model that emits deterministic ChatMessage tool calls and records prompts."""

    def __init__(
        self,
        script: list[ScriptedCall | list[ScriptedCall]],
        *,
        model_id: str = "fake-coding-model-v1",
    ) -> None:
        super().__init__(model_id=model_id)
        self._script = list(script)
        self._cursor = 0
        self.received_messages: list[list[ChatMessage]] = []

    @property
    def call_count(self) -> int:
        return self._cursor

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        self.received_messages.append(list(messages))
        if self._cursor >= len(self._script):
            raise RuntimeError(
                f"FakeModel script exhausted after {self._cursor} turn(s)."
            )
        entry = self._script[self._cursor]
        self._cursor += 1

        calls = entry if isinstance(entry, list) else [entry]
        tool_calls: list[ChatMessageToolCall] = []
        total_in = 0
        total_out = 0
        thoughts: list[str] = []

        for idx, item in enumerate(calls, start=1):
            tcid = item.tool_call_id or f"call_{self._cursor}_{idx}"
            tool_calls.append(
                ChatMessageToolCall(
                    id=tcid,
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=item.tool_name,
                        arguments=item.arguments,
                    ),
                )
            )
            total_in += item.input_tokens
            total_out += item.output_tokens
            if item.thought:
                thoughts.append(item.thought)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="\n".join(thoughts) if thoughts else None,
            tool_calls=tool_calls,
            token_usage=TokenUsage(input_tokens=total_in, output_tokens=total_out),
        )
