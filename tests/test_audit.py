"""Tests for append-only AuditLogger and full decision-chain correlation (A1, A2)."""

from __future__ import annotations

from pathlib import Path

from audit import AuditLogger
from harness import GuardedHarness
from schemas import RunStatus, TaskType
from tests.fake_model import FakeModel, ScriptedCall


def test_a1_rejected_call_audit_contains_required_fields_and_integrity(tmp_path: Path) -> None:
    """A1: Rejected tool call audit record includes rule, decision, actor, and toolCallId; hash chain stays valid."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello\n", encoding="utf-8")

    model = FakeModel(
        [
            # Denied by Gate (G1 missing required 'path')
            ScriptedCall("read_file", {}, tool_call_id="tc_a1_bad_arg"),
            # Denied by Sandbox (S1 path escape)
            ScriptedCall("read_file", {"path": "artifacts/../../.env"}, tool_call_id="tc_a1_escape"),
            # Denied by Sandbox (S3 disallowed command)
            ScriptedCall("run_test", {"command": "rm -rf /"}, tool_call_id="tc_a1_bad_cmd"),
            # Valid read + write request
            ScriptedCall("read_file", {"path": "notes.txt"}, tool_call_id="tc_a1_ok_read"),
            ScriptedCall(
                "write_file",
                {"path": "artifacts/summary.md", "content": "# Summary\nRead notes.txt.\n"},
                tool_call_id="tc_a1_write",
            ),
        ]
    )

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    res = harness.run(
        "Summarize notes.txt into artifacts/summary.md",
        task_type=TaskType.CODE_UNDERSTANDING,
        run_id="run_a1",
        required_deliverables=["artifacts/summary.md"],
    )
    assert res.status == RunStatus.WAITING_APPROVAL

    for tcid, expected_rule in [
        ("tc_a1_bad_arg", "ARG_VALIDATION"),
        ("tc_a1_escape", "SANDBOX_PATH_ESCAPE"),
        ("tc_a1_bad_cmd", "SANDBOX_COMMAND_WHITELIST"),
    ]:
        records = harness.audit.query(run_id="run_a1", tool_call_id=tcid)
        assert len(records) == 1
        rec = records[0]
        assert rec["rule"] == expected_rule
        assert rec["decision"] == "DENY"
        assert rec["actor"] in ("gate", "sandbox")
        assert rec["toolCallId"] == tcid

    # Verify append-only cryptographic hash chain integrity
    assert harness.audit.verify_integrity() is True

    # Tampering with audit log must break integrity verification
    import json
    raw_lines = harness.audit.log_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(raw_lines[0])
    first_event["rule"] = "TAMPERED_RULE"
    tampered = json.dumps(first_event, ensure_ascii=False)
    harness.audit.log_path.write_text("\n".join([tampered] + raw_lines[1:]) + "\n", encoding="utf-8")
    assert harness.audit.verify_integrity() is False


def test_a2_full_task_reconstructs_end_to_end_decision_chain(tmp_path: Path) -> None:
    """A2: A complete task audit log correlates Gate, Sandbox, State, Approval, Checkpoint, Execution, and Completion."""
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "math_ops.py").write_text(
        "def multiply(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_math_ops.py").write_text(
        "from src.math_ops import multiply\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )

    # Phase 1: fails test -> reads code -> triggers sandbox block -> requests edit approval
    model1 = FakeModel(
        [
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="a2_test_fail"),
            ScriptedCall("read_file", {"path": "artifacts/../../.env"}, tool_call_id="a2_sandbox_deny"),
            ScriptedCall("read_file", {"path": "src/math_ops.py"}, tool_call_id="a2_read"),
            ScriptedCall(
                "edit_file",
                {
                    "path": "src/math_ops.py",
                    "old_text": "return a + b",
                    "new_text": "return a * b",
                },
                tool_call_id="a2_edit",
            ),
        ]
    )
    h1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model1)
    r1 = h1.run("Fix multiply bug", task_type=TaskType.TEST_FIX, run_id="run_a2")
    assert r1.status == RunStatus.WAITING_APPROVAL
    approval_id = r1.state.pending_approvals()[0].approval_id

    # Human approves
    h1.approve("run_a2", approval_id, reviewer="reviewer_z", note="approved fix")

    # Phase 2: resume executes edit -> runs passing pytest -> completes
    model2 = FakeModel(
        [
            ScriptedCall("run_test", {"command": "pytest -q"}, tool_call_id="a2_test_pass"),
            ScriptedCall(
                "final_answer",
                {"answer": "Fixed multiply() to return a * b; pytest now passes."},
                tool_call_id="a2_final",
            ),
        ]
    )
    h2 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model2)
    r2 = h2.resume("run_a2")
    assert r2.completed is True

    audit = AuditLogger(state_dir / "audit.jsonl")
    summary = audit.summarize_run("run_a2")
    assert summary["integrity_ok"] is True

    # Verify all required stages are linked in the same run_id chain
    stages = set(summary["stages"])
    assert {"GATE", "SANDBOX", "STATE", "APPROVAL", "EXECUTION", "CHECKPOINT", "COMPLETION", "HARNESS"} <= stages

    # Verify tool_call_id 'a2_edit' traces full lifecycle: REQUIRE_APPROVAL -> APPROVED -> ALLOW -> STARTED -> SUCCEEDED
    edit_chain = audit.reconstruct_chain("run_a2", tool_call_id="a2_edit")
    decisions = [(e["stage"], e["decision"]) for e in edit_chain]
    assert ("GATE", "REQUIRE_APPROVAL") in decisions
    assert ("APPROVAL", "APPROVED") in decisions
    assert ("GATE", "ALLOW") in decisions
    assert ("EXECUTION", "STARTED") in decisions
    assert ("EXECUTION", "SUCCEEDED") in decisions
    # Sequence order must be strictly chronological
    seqs = [e["seq"] for e in edit_chain]
    assert seqs == sorted(seqs)
