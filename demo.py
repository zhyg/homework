"""Executable demo & evidence generator for the Guarded Coding Harness.

Modes:
  1. Default deterministic cross-process demo (FakeModel):
       uv run python demo.py
     Generates submission artifacts in records/:
       - records/run_record.json
       - records/audit_record.jsonl
       - records/checkpoint_resume_record.json

  2. Real LLM mode (T2, OpenAI-compatible API via smolagents.OpenAIServerModel):
       OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... \\
       uv run python demo.py --real-model --task-type test_fix [--auto-approve]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from audit import AuditLogger
from harness import GuardedHarness
from schemas import BudgetConfig, RunStatus, TaskType
from tests.fake_model import FakeModel, ScriptedCall


def _create_sample_repo(workspace: Path) -> dict[str, str]:
    """Create a realistic mini Python repository with a failing unit test."""
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")

    buggy_source = (
        '"""Discount calculator module."""\n\n'
        "def apply_discount(price: float, percent: float) -> float:\n"
        '    """Apply a percentage discount (0-100) to price."""\n'
        "    if percent < 0 or percent > 100:\n"
        '        raise ValueError("percent must be between 0 and 100")\n'
        "    # BUG: adds discount instead of subtracting it\n"
        "    return round(price * (1 + percent / 100.0), 2)\n"
    )
    test_source = (
        "import pytest\n"
        "from src.discount import apply_discount\n\n"
        "def test_apply_discount_normal():\n"
        "    assert apply_discount(100.0, 20.0) == 80.0\n\n"
        "def test_apply_discount_zero():\n"
        "    assert apply_discount(50.0, 0.0) == 50.0\n\n"
        "def test_apply_discount_invalid():\n"
        "    with pytest.raises(ValueError):\n"
        "        apply_discount(100.0, 150.0)\n"
    )
    (workspace / "src" / "discount.py").write_text(buggy_source, encoding="utf-8")
    (workspace / "tests" / "test_discount.py").write_text(test_source, encoding="utf-8")
    return {"src/discount.py": buggy_source}


def run_deterministic_demo(records_dir: Path) -> None:
    """Run an end-to-end cross-process test_fix demo and write submission records."""
    records_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="guarded_harness_demo_"))
    workspace = tmp_root / "workspace"
    state_dir = tmp_root / "state"
    run_id = "run_demo_test_fix_001"
    repo_root = Path(__file__).resolve().parent

    try:
        original_files = _create_sample_repo(workspace)
        process1_pid = os.getpid()

        # Process 1: plan -> run failing pytest -> trigger sandbox escape denial -> read source -> request edit approval
        model_p1 = FakeModel(
            [
                ScriptedCall(
                    tool_name="update_plan",
                    arguments={
                        "goal": "Diagnose and fix failing discount calculation tests",
                        "items": [
                            {"id": "1", "title": "Run pytest to reproduce failure", "status": "in_progress"},
                            {"id": "2", "title": "Inspect src/discount.py", "status": "pending"},
                            {"id": "3", "title": "Patch bug and re-run pytest", "status": "pending"},
                        ],
                    },
                    tool_call_id="demo_call_1_plan",
                    thought="First set an explicit plan for fixing the failing tests.",
                ),
                ScriptedCall(
                    tool_name="run_test",
                    arguments={"command": "pytest -q"},
                    tool_call_id="demo_call_2_test_fail",
                    thought="Run pytest to observe the current failure.",
                ),
                ScriptedCall(
                    tool_name="read_file",
                    arguments={"path": "artifacts/../../.env"},
                    tool_call_id="demo_call_3_blocked_escape",
                    thought="Attempt a forbidden path escape to demonstrate Sandbox interception.",
                ),
                ScriptedCall(
                    tool_name="read_file",
                    arguments={"path": "src/discount.py"},
                    tool_call_id="demo_call_4_read_src",
                    thought="Read src/discount.py to locate the discount bug.",
                ),
                ScriptedCall(
                    tool_name="edit_file",
                    arguments={
                        "path": "src/discount.py",
                        "old_text": (
                            "    # BUG: adds discount instead of subtracting it\n"
                            "    return round(price * (1 + percent / 100.0), 2)"
                        ),
                        "new_text": "    return round(price * (1 - percent / 100.0), 2)",
                    },
                    tool_call_id="demo_call_5_edit_src",
                    thought="Request approval to fix the sign error in apply_discount().",
                ),
            ]
        )

        harness_p1 = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model_p1)
        pause_result = harness_p1.run(
            "Fix the failing test in tests/test_discount.py and verify pytest passes.",
            task_type=TaskType.TEST_FIX,
            run_id=run_id,
            budget=BudgetConfig(max_steps=12, max_tool_calls=25, max_tokens=50_000, max_writes=5),
        )
        assert pause_result.status == RunStatus.WAITING_APPROVAL

        pause_checkpoint = harness_p1.checkpoint_manager.load_latest(run_id)
        pending = pause_result.state.pending_approvals()[0]
        approval_id = pending.approval_id

        # Human reviews and approves the pending edit
        approval_record = harness_p1.approve(
            run_id,
            approval_id,
            reviewer="alice_reviewer",
            note="Verified sign fix in apply_discount(); approved.",
        )

        # Process 2: Launch a real separate Python subprocess to resume from checkpoint!
        # Model attempts a duplicate replay of demo_call_5_edit_src (proving idempotency),
        # updates plan, runs pytest (passes!), and calls final_answer.
        worker_code = f"""
import json, os, sys
sys.path.insert(0, {str(repo_root)!r})
from harness import GuardedHarness
from tests.fake_model import FakeModel, ScriptedCall

model_p2 = FakeModel([
    ScriptedCall(
        "edit_file",
        {{
            "path": "src/discount.py",
            "old_text": "    # BUG: adds discount instead of subtracting it\\n    return round(price * (1 + percent / 100.0), 2)",
            "new_text": "    return round(price * (1 - percent / 100.0), 2)",
        }},
        tool_call_id="demo_call_5_edit_src",
        thought="Simulate replay of already-committed write; Gate must block duplicate write.",
    ),
    ScriptedCall(
        "update_plan",
        {{
            "goal": "Diagnose and fix failing discount calculation tests",
            "items": [
                {{"id": "1", "title": "Run pytest to reproduce failure", "status": "done"}},
                {{"id": "2", "title": "Inspect src/discount.py", "status": "done"}},
                {{"id": "3", "title": "Patch bug and re-run pytest", "status": "done"}},
            ],
        }},
        tool_call_id="demo_call_6_plan_done",
        thought="Mark all plan items complete.",
    ),
    ScriptedCall(
        "run_test",
        {{"command": "pytest -q"}},
        tool_call_id="demo_call_7_test_pass",
        thought="Re-run pytest against the patched code.",
    ),
    ScriptedCall(
        "final_answer",
        {{"answer": "Fixed sign error in src/discount.py::apply_discount (changed 1 + percent/100 to 1 - percent/100). All 3 tests in tests/test_discount.py now pass."}},
        tool_call_id="demo_call_8_final",
        thought="All tests pass on current source hash; return final answer.",
    ),
])
h2 = GuardedHarness(workspace_root={str(workspace)!r}, state_dir={str(state_dir)!r}, model=model_p2)
res = h2.resume({run_id!r})
print(json.dumps({{"process2_pid": os.getpid(), "result": res.to_dict()}}, ensure_ascii=False))
"""
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
            check=True,
        )
        p2_payload = json.loads(proc.stdout.strip().splitlines()[-1])
        process2_pid = p2_payload["process2_pid"]
        assert p2_payload["result"]["completed"] is True
        assert process2_pid != process1_pid

        # Load final state, audit summary, and diff
        final_state = harness_p1.get_run_state(run_id)
        final_checkpoint = harness_p1.checkpoint_manager.load_latest(run_id)
        audit = AuditLogger(state_dir / "audit.jsonl")
        assert audit.verify_integrity() is True

        patched_source = (workspace / "src" / "discount.py").read_text(encoding="utf-8")
        diff_lines = list(
            difflib.unified_diff(
                original_files["src/discount.py"].splitlines(keepends=True),
                patched_source.splitlines(keepends=True),
                fromfile="a/src/discount.py",
                tofile="b/src/discount.py",
            )
        )

        # 1. Write complete run record
        run_record = {
            "summary": {
                "run_id": run_id,
                "task_type": final_state.task_type.value,
                "status": final_state.status.value,
                "stop_reason": final_state.stop_reason.value,
                "final_answer": final_state.final_answer,
                "step_number": final_state.step_number,
                "budget": final_state.budget.to_dict(),
                "audit_integrity_ok": audit.verify_integrity(),
                "audit_summary": audit.summarize_run(run_id),
            },
            "diff": "".join(diff_lines),
            "full_state": final_state.to_dict(),
        }
        (records_dir / "run_record.json").write_text(
            json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # 2. Copy append-only audit record
        shutil.copyfile(state_dir / "audit.jsonl", records_dir / "audit_record.jsonl")

        # 3. Write cross-process checkpoint & resume record
        replay_denials = audit.query(
            run_id=run_id, tool_call_id="demo_call_5_edit_src", decision="DENY"
        )
        checkpoint_resume_record = {
            "run_id": run_id,
            "cross_process_verified": process1_pid != process2_pid,
            "process_1": {
                "pid": process1_pid,
                "status_at_pause": pause_result.status.value,
                "stop_reason": pause_result.stop_reason.value,
                "checkpoint_id": pause_checkpoint.checkpoint_id,
                "step_at_pause": pause_checkpoint.step,
                "budget_at_pause": pause_checkpoint.budget,
                "pending_approvals": pause_checkpoint.pending_approvals,
                "checkpoint_payload": pause_checkpoint.to_dict(),
            },
            "human_approval": approval_record.to_dict(),
            "process_2_resume": {
                "pid": process2_pid,
                "final_status": final_state.status.value,
                "final_stop_reason": final_state.stop_reason.value,
                "final_checkpoint_id": final_checkpoint.checkpoint_id,
                "step_after_resume": final_state.step_number,
                "budget_after_resume": final_state.budget.to_dict(),
                "counters_monotonic_verified": (
                    final_state.budget.steps_used > pause_checkpoint.budget["steps_used"]
                    and final_state.budget.writes_used == 1
                ),
                "duplicate_write_replay_blocked": len(replay_denials) == 1,
                "duplicate_write_denial_audit": replay_denials,
            },
            "checkpoints_sequence": [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "step": c.step,
                    "status": c.run["status"],
                    "steps_used": c.budget["steps_used"],
                    "writes_used": c.budget["writes_used"],
                    "pending_approvals": len(c.pending_approvals),
                }
                for c in harness_p1.checkpoint_manager.list_checkpoints(run_id)
            ],
        }
        (records_dir / "checkpoint_resume_record.json").write_text(
            json.dumps(checkpoint_resume_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Generated records in {records_dir}:")
        print(f"  - {records_dir / 'run_record.json'}")
        print(f"  - {records_dir / 'audit_record.jsonl'}")
        print(f"  - {records_dir / 'checkpoint_resume_record.json'}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_real_model_demo(
    *,
    task_type: TaskType,
    records_dir: Path,
    auto_approve: bool,
) -> None:
    """Run T2 with a real OpenAI-compatible LLM via smolagents.OpenAIServerModel."""
    from smolagents import OpenAIServerModel

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for --real-model mode.")
    model_id = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_base = os.environ.get("OPENAI_BASE_URL")

    model = OpenAIServerModel(model_id=model_id, api_key=api_key, api_base=api_base)
    tmp_root = Path(tempfile.mkdtemp(prefix="guarded_harness_real_"))
    workspace = tmp_root / "workspace"
    state_dir = tmp_root / "state"
    _create_sample_repo(workspace)

    harness = GuardedHarness(workspace_root=workspace, state_dir=state_dir, model=model)
    if task_type == TaskType.CODE_UNDERSTANDING:
        task = "Inspect this repository and write a concise code analysis report to artifacts/analysis.md."
        deliverables = ["artifacts/analysis.md"]
    else:
        task = "Run pytest to diagnose the bug in src/discount.py, fix it, and re-run pytest until all tests pass."
        deliverables = []

    res = harness.run(task, task_type=task_type, required_deliverables=deliverables)
    while res.status == RunStatus.WAITING_APPROVAL:
        for pending in res.state.pending_approvals():
            print(f"\n[APPROVAL REQUIRED] {pending.approval_id}: {pending.tool_name}({pending.arguments})")
            if auto_approve:
                decision = "y"
            else:
                decision = input("Approve this write? [y/N]: ").strip().lower()
            if decision == "y":
                harness.approve(res.run_id, pending.approval_id, reviewer="cli_user")
            else:
                harness.reject(res.run_id, pending.approval_id, reviewer="cli_user")
        res = harness.resume(res.run_id)

    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "real_model_run_record.json").write_text(
        json.dumps(res.state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(state_dir / "audit.jsonl", records_dir / "real_model_audit_record.jsonl")
    print(f"Real-model run finished: status={res.status.value}, stop_reason={res.stop_reason.value}")
    print(f"Workspace preserved at: {workspace}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded Coding Harness demo & evidence generator")
    parser.add_argument("--real-model", action="store_true", help="Run with a real OpenAI-compatible LLM")
    parser.add_argument(
        "--task-type",
        choices=[t.value for t in TaskType],
        default=TaskType.TEST_FIX.value,
        help="Task type for --real-model mode",
    )
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve write requests in --real-model mode")
    parser.add_argument("--records-dir", default="records", help="Output directory for run/audit/checkpoint records")
    args = parser.parse_args()

    if args.real_model:
        run_real_model_demo(
            task_type=TaskType(args.task_type),
            records_dir=Path(args.records_dir),
            auto_approve=args.auto_approve,
        )
    else:
        run_deterministic_demo(Path(args.records_dir))


if __name__ == "__main__":
    main()
