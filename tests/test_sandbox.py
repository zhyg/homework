"""Tests for Sandbox execution boundaries (S1, S2, S3, S4, env scrubbing, and output truncation)."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import pytest

from audit import AuditLogger
from checkpoint import CheckpointManager
from gate import ArgSpec, ToolGate, ToolPolicy
from sandbox import Sandbox, SandboxConfig, SandboxViolationError
from schemas import PermissionMode, RiskLevel, RunState, RunStatus, TaskType
from state_store import StateStore


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _make_gate_with_sandbox(
    tmp_path: Path,
    sandbox_config: SandboxConfig | None = None,
) -> tuple[ToolGate, Sandbox, AuditLogger, dict[str, int]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)
    (workspace / "app.py").write_text("print('ok')\n", encoding="utf-8")

    cfg = sandbox_config or SandboxConfig(workspace_root=workspace)
    sandbox = Sandbox(cfg)
    state_dir = tmp_path / "state"
    audit = AuditLogger(state_dir / "audit.jsonl")
    store = StateStore(state_dir, audit=audit)
    ckpt = CheckpointManager(state_dir, audit=audit)
    snap, ws_hash, code_hash = sandbox.compute_workspace_snapshot()
    run_state = RunState(
        run_id="run_sandbox_test",
        task="sandbox unit test",
        task_type=TaskType.TEST_FIX,
        status=RunStatus.RUNNING,
        workspace_snapshot=snap,
        workspace_hash=ws_hash,
        code_hash=code_hash,
    )
    store.save_run(run_state)

    counts = {"read_file": 0, "run_test": 0}

    gate = ToolGate(
        run_state=run_state,
        sandbox=sandbox,
        state_store=store,
        checkpoint_manager=ckpt,
        audit=audit,
    )
    gate.register(
        ToolPolicy(
            name="read_file",
            handler=lambda path: (counts.__setitem__("read_file", counts["read_file"] + 1) or sandbox.read_file(path)),
            arg_schema={"path": ArgSpec(str, required=True, non_empty=True)},
            permission=PermissionMode.READ_ONLY,
            risk_level=RiskLevel.LOW,
            path_args=("path",),
            evidence_kind="READ",
        )
    )
    gate.register(
        ToolPolicy(
            name="run_test",
            handler=lambda command, timeout_sec=None: (
                counts.__setitem__("run_test", counts["run_test"] + 1)
                or sandbox.run_command(command=command, timeout_sec=timeout_sec)
            ),
            arg_schema={
                "command": ArgSpec(str, required=True, non_empty=True),
                "timeout_sec": ArgSpec((int, float), required=False, allow_none=True),
            },
            permission=PermissionMode.TEST_EXEC,
            risk_level=RiskLevel.MEDIUM,
            command_args=("command",),
            evidence_kind="TEST_RUN",
        )
    )
    return gate, sandbox, audit, counts


def test_s1_path_traversal_escape_rejected_without_handler(tmp_path: Path) -> None:
    """S1: artifacts/../../.env path escape is rejected by Sandbox before handler executes."""
    # Place a real .env outside workspace to ensure it's never read
    (tmp_path / ".env").write_text("SECRET=topsecret\n", encoding="utf-8")
    gate, _sandbox, audit, counts = _make_gate_with_sandbox(tmp_path)

    resp = gate.call("tc_s1_escape", "read_file", {"path": "artifacts/../../.env"})

    assert not resp.ok
    assert resp.status == "DENIED"
    assert resp.tool_call_id == "tc_s1_escape"
    assert resp.rule == "SANDBOX_PATH_ESCAPE"
    assert counts["read_file"] == 0

    events = audit.query(run_id="run_sandbox_test", tool_call_id="tc_s1_escape")
    assert len(events) == 1
    assert events[0]["stage"] == "SANDBOX"
    assert events[0]["rule"] == "SANDBOX_PATH_ESCAPE"
    assert events[0]["decision"] == "DENY"


def test_s2_symlink_escape_and_sensitive_directory_rejected_with_audit(tmp_path: Path) -> None:
    """S2: Symlink pointing outside workspace and sensitive dirs/files are rejected after realpath check and audited."""
    outside_secret = tmp_path / "outside_secret.txt"
    outside_secret.write_text("do-not-leak\n", encoding="utf-8")

    gate, sandbox, audit, counts = _make_gate_with_sandbox(tmp_path)
    # Create symlink inside workspace pointing outside workspace
    escape_link = sandbox.workspace_root / "artifacts" / "leak_link.txt"
    escape_link.symlink_to(outside_secret)

    # Create sensitive files inside workspace
    (sandbox.workspace_root / ".env").write_text("API_KEY=123\n", encoding="utf-8")
    (sandbox.workspace_root / ".git").mkdir()
    (sandbox.workspace_root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    # Create internal symlink pointing at .env
    internal_link = sandbox.workspace_root / "normal_alias.txt"
    internal_link.symlink_to(sandbox.workspace_root / ".env")

    # 1. Symlink escape outside workspace
    r_sym = gate.call("tc_s2_symlink", "read_file", {"path": "artifacts/leak_link.txt"})
    assert not r_sym.ok
    assert r_sym.rule == "SANDBOX_SYMLINK_ESCAPE"
    assert counts["read_file"] == 0

    # 2. Direct sensitive file (.env)
    r_env = gate.call("tc_s2_env", "read_file", {"path": ".env"})
    assert not r_env.ok
    assert r_env.rule == "SANDBOX_SENSITIVE_PATH"
    assert counts["read_file"] == 0

    # 3. Sensitive directory (.git/config)
    r_git = gate.call("tc_s2_git", "read_file", {"path": ".git/config"})
    assert not r_git.ok
    assert r_git.rule == "SANDBOX_SENSITIVE_PATH"
    assert counts["read_file"] == 0

    # 4. Internal symlink to sensitive .env (caught by realpath validation!)
    r_alias = gate.call("tc_s2_alias", "read_file", {"path": "normal_alias.txt"})
    assert not r_alias.ok
    assert r_alias.rule == "SANDBOX_SENSITIVE_PATH"
    assert counts["read_file"] == 0

    # Verify audit trail recorded every denial
    for tcid in ("tc_s2_symlink", "tc_s2_env", "tc_s2_git", "tc_s2_alias"):
        ev = audit.query(run_id="run_sandbox_test", tool_call_id=tcid)
        assert len(ev) == 1
        assert ev[0]["stage"] == "SANDBOX"
        assert ev[0]["decision"] == "DENY"


def test_s3_disallowed_commands_and_shell_injection_blocked(tmp_path: Path) -> None:
    """S3: run_test commands outside whitelist or containing shell metacharacters are rejected without executing."""
    gate, sandbox, audit, counts = _make_gate_with_sandbox(tmp_path)
    (sandbox.workspace_root / "test_ok.py").write_text("def test_pass():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    # Disallowed commands
    bad_commands = [
        ("tc_s3_rm", "rm -rf /tmp/nonexistent", "SANDBOX_COMMAND_WHITELIST"),
        ("tc_s3_cat", "cat /etc/passwd", "SANDBOX_COMMAND_WHITELIST"),
        ("tc_s3_py_c", "python -c 'print(1)'", "SANDBOX_COMMAND_WHITELIST"),
        ("tc_s3_chain", "pytest && echo pwned", "SANDBOX_SHELL_INJECTION"),
        ("tc_s3_pipe", "pytest | cat", "SANDBOX_SHELL_INJECTION"),
        ("tc_s3_subst", "pytest $(whoami)", "SANDBOX_SHELL_INJECTION"),
    ]
    for tcid, cmd, expected_rule in bad_commands:
        resp = gate.call(tcid, "run_test", {"command": cmd})
        assert not resp.ok
        assert resp.tool_call_id == tcid
        assert resp.rule == expected_rule
        assert counts["run_test"] == 0

    # Allowed command succeeds
    ok_resp = gate.call("tc_s3_ok", "run_test", {"command": "pytest -q test_ok.py"})
    assert ok_resp.ok
    assert ok_resp.output["passed"] is True
    assert counts["run_test"] == 1
    assert len(audit.query(run_id="run_sandbox_test", stage="SANDBOX", decision="DENY")) == len(bad_commands)


def _write_spawning_test_file(workspace: Path, pid_file: Path) -> None:
    """Create a pytest test that spawns a grandchild sleeping process and records grandchild PID to pid_file."""
    (workspace / "test_hang.py").write_text(
        f"""
import subprocess, sys, time
from pathlib import Path

def test_spawns_grandchild():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path({str(pid_file)!r}).write_text(str(child.pid), encoding="utf-8")
    time.sleep(60)
""",
        encoding="utf-8",
    )


def test_s4_timeout_kills_entire_process_group(tmp_path: Path) -> None:
    """S4 (timeout): Command timeout terminates the entire POSIX process group including grandchild processes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid_file = tmp_path / "grandchild_timeout.pid"
    _write_spawning_test_file(workspace, pid_file)

    sandbox = Sandbox(SandboxConfig(workspace_root=workspace, default_timeout_sec=1.2, kill_grace_sec=0.15))
    result = sandbox.run_command("pytest -q test_hang.py", timeout_sec=1.0)

    assert result.timed_out is True
    assert result.passed is False
    assert result.pid is not None and result.pgid is not None

    # Wait briefly for kernel process table cleanup
    time.sleep(0.15)
    assert not _pid_is_alive(result.pid), "Parent process should be dead after timeout"
    assert pid_file.exists(), "Grandchild process should have been spawned before timeout"
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    assert not _pid_is_alive(grandchild_pid), "Grandchild process in process group must also be dead!"


def test_s4_user_cancel_kills_entire_process_group(tmp_path: Path) -> None:
    """S4 (cancel): User cancellation signal terminates the entire POSIX process group including grandchild."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid_file = tmp_path / "grandchild_cancel.pid"
    _write_spawning_test_file(workspace, pid_file)

    sandbox = Sandbox(SandboxConfig(workspace_root=workspace, default_timeout_sec=20.0, kill_grace_sec=0.15))
    cancel_event = threading.Event()

    def trigger_cancel_when_spawned() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                cancel_event.set()
                return
            time.sleep(0.05)
        cancel_event.set()

    canceller = threading.Thread(target=trigger_cancel_when_spawned, daemon=True)
    canceller.start()
    result = sandbox.run_command("pytest -q test_hang.py", timeout_sec=15.0, cancel_event=cancel_event)
    canceller.join(timeout=2.0)

    assert result.cancelled is True
    assert result.timed_out is False
    assert result.passed is False

    time.sleep(0.15)
    assert result.pid is not None and not _pid_is_alive(result.pid)
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    assert not _pid_is_alive(grandchild_pid), "Grandchild process must be dead after user cancellation!"


def test_env_scrubbing_and_output_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sandbox scrubs secret environment variables and truncates oversized stdout."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    monkeypatch.setenv("MY_CUSTOM_SECRET_TOKEN", "tok-999999")

    (workspace / "test_env_and_out.py").write_text(
        """
import os

def test_env_and_large_output():
    assert "OPENAI_API_KEY" not in os.environ
    assert "MY_CUSTOM_SECRET_TOKEN" not in os.environ
    print("X" * 5000)
""",
        encoding="utf-8",
    )

    sandbox = Sandbox(SandboxConfig(workspace_root=workspace, max_output_bytes=512))
    res = sandbox.run_command("pytest -s test_env_and_out.py")
    assert res.passed is True
    assert res.stdout_truncated is True
    assert "[OUTPUT TRUNCATED:" in res.stdout
    assert "sk-super-secret-value" not in res.stdout


def test_dangling_symlink_write_escape_rejected(tmp_path: Path) -> None:
    """Writing through a dangling symlink pointing outside workspace is rejected before creating the outside file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_target = tmp_path / "escaped_new_file.txt"
    dangling = workspace / "evil_link.txt"
    dangling.symlink_to(outside_target)

    sandbox = Sandbox(SandboxConfig(workspace_root=workspace))
    with pytest.raises(SandboxViolationError) as exc_info:
        sandbox.write_file("evil_link.txt", "pwned")

    assert exc_info.value.rule == "SANDBOX_SYMLINK_ESCAPE"
    assert not outside_target.exists()
