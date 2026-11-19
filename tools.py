"""Thin smolagents Tool declarations and pure handler bindings for Guarded Coding Harness.

All business rules (argument checks, permissions, sandbox boundaries, approvals, duplicate limits,
write-ahead journaling, and audit logging) live centrally in gate.py, sandbox.py, and harness.py—
NOT scattered inside individual tool handlers.
"""

from __future__ import annotations

import threading
from typing import Any

from smolagents import Tool

from gate import ArgSpec, ToolGate, ToolPolicy
from sandbox import Sandbox
from schemas import PermissionMode, PlanItem, RiskLevel, RunState


class _GateOnlyTool(Tool):
    """Base smolagents Tool whose forward() refuses direct bypass outside ToolGate."""

    def _forbid_direct_call(self) -> None:
        raise RuntimeError(
            f"Tool '{self.name}' cannot be executed directly; all tool calls must pass through ToolGate."
        )


class ListFilesTool(_GateOnlyTool):
    name = "list_files"
    description = "List non-sensitive files in a workspace directory."
    inputs = {
        "directory": {
            "type": "string",
            "description": "Relative directory path inside workspace (defaults to '.').",
            "nullable": True,
        }
    }
    output_type = "string"

    def forward(self, directory: str | None = None) -> str:
        self._forbid_direct_call()
        return ""


class ReadFileTool(_GateOnlyTool):
    name = "read_file"
    description = "Read a file inside the workspace, optionally slicing by line range."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative file path inside the workspace.",
        },
        "start_line": {
            "type": "integer",
            "description": "Optional 1-based starting line number.",
            "nullable": True,
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-based ending line number (inclusive).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        self._forbid_direct_call()
        return ""


class SearchCodeTool(_GateOnlyTool):
    name = "search_code"
    description = "Search for a literal substring across workspace files."
    inputs = {
        "query": {
            "type": "string",
            "description": "Non-empty literal string to search for.",
        },
        "directory": {
            "type": "string",
            "description": "Optional relative directory to restrict search.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, directory: str | None = None) -> str:
        self._forbid_direct_call()
        return ""


class WriteFileTool(_GateOnlyTool):
    name = "write_file"
    description = "Create or overwrite a file inside the workspace. Requires human approval."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative destination path inside workspace (e.g., artifacts/report.md).",
        },
        "content": {
            "type": "string",
            "description": "Full UTF-8 text content to write.",
        },
    }
    output_type = "string"

    def forward(self, path: str, content: str) -> str:
        self._forbid_direct_call()
        return ""


class EditFileTool(_GateOnlyTool):
    name = "edit_file"
    description = "Replace one unique text occurrence in an existing workspace file. Requires human approval."
    inputs = {
        "path": {
            "type": "string",
            "description": "Relative file path inside workspace.",
        },
        "old_text": {
            "type": "string",
            "description": "Exact existing substring to replace.",
        },
        "new_text": {
            "type": "string",
            "description": "Replacement text.",
        },
    }
    output_type = "string"

    def forward(self, path: str, old_text: str, new_text: str) -> str:
        self._forbid_direct_call()
        return ""


class RunTestTool(_GateOnlyTool):
    name = "run_test"
    description = "Run a whitelisted test command inside the sandboxed workspace."
    inputs = {
        "command": {
            "type": "string",
            "description": "Whitelisted test command such as 'pytest -q' or 'python -m pytest'.",
        },
        "timeout_sec": {
            "type": "number",
            "description": "Optional command timeout in seconds.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, command: str, timeout_sec: float | None = None) -> str:
        self._forbid_direct_call()
        return ""


class UpdatePlanTool(_GateOnlyTool):
    name = "update_plan"
    description = "Update the structured execution plan stored in run state."
    inputs = {
        "goal": {
            "type": "string",
            "description": "High-level goal description.",
        },
        "items": {
            "type": "array",
            "description": "List of plan items: [{'id': '1', 'title': '...', 'status': 'pending|in_progress|done'}].",
        },
    }
    output_type = "string"

    def forward(self, goal: str, items: list[dict[str, Any]]) -> str:
        self._forbid_direct_call()
        return ""


def build_smolagents_tools() -> list[Tool]:
    """Return the list of smolagents Tool schemas exposed to ToolCallingAgent."""
    return [
        ListFilesTool(),
        ReadFileTool(),
        SearchCodeTool(),
        WriteFileTool(),
        EditFileTool(),
        RunTestTool(),
        UpdatePlanTool(),
    ]


def register_coding_tools(
    *,
    gate: ToolGate,
    sandbox: Sandbox,
    run_state: RunState,
    cancel_event: threading.Event | None = None,
    require_write_approval: bool = True,
) -> None:
    """Register all coding tool handlers and central policies on ToolGate."""

    def handle_update_plan(goal: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        parsed: list[PlanItem] = []
        for idx, raw in enumerate(items, start=1):
            if not isinstance(raw, dict) or "title" not in raw:
                raise ValueError("Each plan item must be a dict containing 'title'.")
            parsed.append(
                PlanItem(
                    id=str(raw.get("id", idx)),
                    title=str(raw["title"]),
                    status=str(raw.get("status", "pending")),
                )
            )
        run_state.plan.goal = goal
        run_state.plan.items = parsed
        run_state.plan.updated_at_step = run_state.step_number
        return run_state.plan.to_dict()

    gate.register(
        ToolPolicy(
            name="list_files",
            handler=lambda directory=".": sandbox.list_files(directory=directory),
            arg_schema={
                "directory": ArgSpec(str, required=False, non_empty=True, allow_none=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            is_mutating=False,
            path_args=("directory",),
            evidence_kind="READ",
        )
    )

    gate.register(
        ToolPolicy(
            name="read_file",
            handler=lambda path, start_line=None, end_line=None: sandbox.read_file(
                path=path, start_line=start_line, end_line=end_line
            ),
            arg_schema={
                "path": ArgSpec(str, required=True, non_empty=True),
                "start_line": ArgSpec(int, required=False, allow_none=True),
                "end_line": ArgSpec(int, required=False, allow_none=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            is_mutating=False,
            path_args=("path",),
            evidence_kind="READ",
        )
    )

    gate.register(
        ToolPolicy(
            name="search_code",
            handler=lambda query, directory=".": sandbox.search_code(
                query=query, directory=directory
            ),
            arg_schema={
                "query": ArgSpec(str, required=True, non_empty=True),
                "directory": ArgSpec(str, required=False, non_empty=True, allow_none=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            is_mutating=False,
            path_args=("directory",),
            evidence_kind="READ",
        )
    )

    gate.register(
        ToolPolicy(
            name="write_file",
            handler=lambda path, content: sandbox.write_file(path=path, content=content),
            arg_schema={
                "path": ArgSpec(str, required=True, non_empty=True),
                "content": ArgSpec(str, required=True, non_empty=False),
            },
            permission=PermissionMode.WORKSPACE_WRITE,
            risk_level=RiskLevel.HIGH,
            requires_approval=require_write_approval,
            is_mutating=True,
            path_args=("path",),
            evidence_kind="WRITE",
        )
    )

    gate.register(
        ToolPolicy(
            name="edit_file",
            handler=lambda path, old_text, new_text: sandbox.edit_file(
                path=path, old_text=old_text, new_text=new_text
            ),
            arg_schema={
                "path": ArgSpec(str, required=True, non_empty=True),
                "old_text": ArgSpec(str, required=True, non_empty=True),
                "new_text": ArgSpec(str, required=True, non_empty=False),
            },
            permission=PermissionMode.WORKSPACE_WRITE,
            risk_level=RiskLevel.HIGH,
            requires_approval=require_write_approval,
            is_mutating=True,
            path_args=("path",),
            evidence_kind="WRITE",
        )
    )

    gate.register(
        ToolPolicy(
            name="run_test",
            handler=lambda command, timeout_sec=None: sandbox.run_command(
                command=command,
                timeout_sec=timeout_sec,
                cancel_event=cancel_event,
            ),
            arg_schema={
                "command": ArgSpec(str, required=True, non_empty=True),
                "timeout_sec": ArgSpec((int, float), required=False, allow_none=True),
            },
            permission=PermissionMode.TEST_EXEC,
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            is_mutating=False,
            command_args=("command",),
            evidence_kind="TEST_RUN",
        )
    )

    gate.register(
        ToolPolicy(
            name="update_plan",
            handler=handle_update_plan,
            arg_schema={
                "goal": ArgSpec(str, required=True, non_empty=True),
                "items": ArgSpec(list, required=True, non_empty=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            is_mutating=False,
            evidence_kind=None,
        )
    )

    # Register final_answer in Gate as well so literally 100% of tool calls pass through Gate.
    gate.register(
        ToolPolicy(
            name="final_answer",
            handler=lambda answer: answer,
            arg_schema={
                "answer": ArgSpec(str, required=True, non_empty=True),
            },
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            is_mutating=False,
            evidence_kind=None,
        )
    )
