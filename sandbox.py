"""Execution sandbox enforcing path, symlink, sensitive-dir, command whitelist, timeout/cancel, env, and output boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from schemas import canonical_json, sha256_text

SHELL_META_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r")

DEFAULT_SENSITIVE_DIRS = (
    ".git",
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".venv",
    ".harness",
)

DEFAULT_SENSITIVE_FILE_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "credentials.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
)

DEFAULT_IGNORED_SNAPSHOT_DIRS = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".harness",
    "node_modules",
)

DEFAULT_ALLOWED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
)

DEFAULT_ENV_WHITELIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "HOME",
    "TMPDIR",
    "TERM",
    "USER",
    "VIRTUAL_ENV",
)

CODE_FILE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".sql",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
}


class SandboxViolationError(Exception):
    """Raised when a path, symlink, sensitive resource, or command violates sandbox policy."""

    def __init__(self, rule: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.rule = rule
        self.message = message
        self.details = details or {}


@dataclass
class SandboxConfig:
    workspace_root: Path
    sensitive_dirs: tuple[str, ...] = DEFAULT_SENSITIVE_DIRS
    sensitive_file_globs: tuple[str, ...] = DEFAULT_SENSITIVE_FILE_GLOBS
    allowed_command_prefixes: tuple[tuple[str, ...], ...] = DEFAULT_ALLOWED_COMMAND_PREFIXES
    env_whitelist: tuple[str, ...] = DEFAULT_ENV_WHITELIST
    default_timeout_sec: float = 15.0
    max_timeout_sec: float = 60.0
    kill_grace_sec: float = 0.25
    max_output_bytes: int = 32_768
    max_file_read_bytes: int = 256_000
    max_file_write_bytes: int = 512_000
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace_root = Path(os.path.realpath(str(self.workspace_root)))


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration_sec: float = 0.0
    pid: int | None = None
    pgid: int | None = None

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "duration_sec": round(self.duration_sec, 4),
            "pid": self.pid,
            "pgid": self.pgid,
        }


class Sandbox:
    """Enforces filesystem, command execution, process-group lifetime, env, and output boundaries."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.workspace_root = config.workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path, symlink, and sensitive directory/file boundary validation
    # ------------------------------------------------------------------

    def _is_sensitive_name(self, name: str, *, is_dir_component: bool) -> bool:
        if is_dir_component and name in self.config.sensitive_dirs:
            return True
        if name in self.config.sensitive_dirs:
            return True
        for pattern in self.config.sensitive_file_globs:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _check_sensitive_parts(self, rel_path: Path, raw_input: str) -> None:
        parts = rel_path.parts
        for idx, part in enumerate(parts):
            is_dir_component = idx < len(parts) - 1
            if self._is_sensitive_name(part, is_dir_component=is_dir_component):
                raise SandboxViolationError(
                    rule="SANDBOX_SENSITIVE_PATH",
                    message=f"Access to sensitive path component '{part}' in '{raw_input}' is forbidden.",
                    details={"requested_path": raw_input, "sensitive_component": part},
                )

    def _contains_symlink_in_chain(self, candidate: Path) -> Path | None:
        """Walk from workspace_root down to candidate and return the first symlink encountered."""
        try:
            rel = candidate.relative_to(self.workspace_root)
        except ValueError:
            return None
        current = self.workspace_root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                return current
        return None

    def validate_path(self, path_str: str, *, for_write: bool = False) -> Path:
        """Validate that path_str resolves inside workspace_root and avoids symlinks/sensitive paths."""
        if not isinstance(path_str, str) or not path_str.strip():
            raise SandboxViolationError(
                rule="SANDBOX_INVALID_PATH",
                message="Path must be a non-empty string.",
                details={"requested_path": path_str},
            )
        if "\x00" in path_str:
            raise SandboxViolationError(
                rule="SANDBOX_INVALID_PATH",
                message="Path must not contain null bytes.",
                details={"requested_path": path_str},
            )

        raw_path = Path(path_str)
        candidate = raw_path if raw_path.is_absolute() else (self.workspace_root / raw_path)

        # Normalize lexically first to distinguish direct traversal (e.g., artifacts/../../.env)
        # from symlink escapes.
        lexical = Path(os.path.normpath(str(candidate)))
        try:
            lexical_rel = lexical.relative_to(self.workspace_root)
        except ValueError:
            raise SandboxViolationError(
                rule="SANDBOX_PATH_ESCAPE",
                message=f"Path '{path_str}' escapes workspace root '{self.workspace_root}'.",
                details={
                    "requested_path": path_str,
                    "lexical_target": str(lexical),
                    "workspace_root": str(self.workspace_root),
                },
            )

        # Check logical path components against sensitive directories/files first.
        self._check_sensitive_parts(lexical_rel, path_str)

        # Detect any symlink along the path (including dangling symlinks for writes).
        symlink_node = self._contains_symlink_in_chain(lexical)

        # Resolve real filesystem landing path.
        real_target = Path(os.path.realpath(str(lexical)))
        try:
            real_rel = real_target.relative_to(self.workspace_root)
        except ValueError:
            if symlink_node is not None:
                raise SandboxViolationError(
                    rule="SANDBOX_SYMLINK_ESCAPE",
                    message=(
                        f"Symlink '{symlink_node}' in path '{path_str}' resolves outside workspace "
                        f"to '{real_target}'."
                    ),
                    details={
                        "requested_path": path_str,
                        "symlink": str(symlink_node),
                        "real_target": str(real_target),
                        "workspace_root": str(self.workspace_root),
                    },
                )
            raise SandboxViolationError(
                rule="SANDBOX_PATH_ESCAPE",
                message=f"Resolved path '{real_target}' escapes workspace root '{self.workspace_root}'.",
                details={
                    "requested_path": path_str,
                    "real_target": str(real_target),
                    "workspace_root": str(self.workspace_root),
                },
            )

        # Also validate the real resolved components against sensitive directories/files
        # (catches internal symlinks pointing at .env or .git/config!).
        self._check_sensitive_parts(real_rel, path_str)

        if for_write and lexical.is_symlink():
            raise SandboxViolationError(
                rule="SANDBOX_SYMLINK_WRITE",
                message=f"Writing through symlink '{path_str}' is forbidden.",
                details={"requested_path": path_str, "real_target": str(real_target)},
            )

        return real_target

    # ------------------------------------------------------------------
    # Safe filesystem primitives
    # ------------------------------------------------------------------

    def list_files(self, directory: str = ".", max_entries: int = 200) -> list[str]:
        target_dir = self.validate_path(directory, for_write=False)
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results: list[str] = []
        for root, dirs, files in os.walk(target_dir, followlinks=False):
            # Prune sensitive and ignored directories
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in DEFAULT_IGNORED_SNAPSHOT_DIRS
                and not self._is_sensitive_name(d, is_dir_component=True)
                and not (Path(root) / d).is_symlink()
            )
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.is_symlink():
                    continue
                if self._is_sensitive_name(fname, is_dir_component=False):
                    continue
                rel = fpath.relative_to(self.workspace_root).as_posix()
                results.append(rel)
                if len(results) >= max_entries:
                    return results
        return results

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        real_path = self.validate_path(path, for_write=False)
        if not real_path.exists() or not real_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        raw_bytes = real_path.read_bytes()
        truncated = False
        if len(raw_bytes) > self.config.max_file_read_bytes:
            raw_bytes = raw_bytes[: self.config.max_file_read_bytes]
            truncated = True

        text = raw_bytes.decode("utf-8", errors="replace")
        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            s_idx = max((start_line or 1) - 1, 0)
            e_idx = end_line if end_line is not None else len(lines)
            selected = lines[s_idx:e_idx]
            text = "\n".join(f"{i + s_idx + 1}: {line}" for i, line in enumerate(selected))
        if truncated:
            text += "\n[FILE TRUNCATED BY SANDBOX]"
        return text

    def search_code(self, query: str, directory: str = ".", max_matches: int = 50) -> list[dict[str, Any]]:
        files = self.list_files(directory=directory, max_entries=500)
        matches: list[dict[str, Any]] = []
        for rel in files:
            real_path = self.workspace_root / rel
            try:
                content = real_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if query in line:
                    matches.append({"path": rel, "line": lineno, "text": line[:200]})
                    if len(matches) >= max_matches:
                        return matches
        return matches

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        real_path = self.validate_path(path, for_write=True)
        encoded = content.encode("utf-8")
        if len(encoded) > self.config.max_file_write_bytes:
            raise SandboxViolationError(
                rule="SANDBOX_WRITE_LIMIT",
                message=(
                    f"Write size ({len(encoded)} bytes) exceeds max_file_write_bytes "
                    f"({self.config.max_file_write_bytes})."
                ),
                details={"path": path, "size": len(encoded)},
            )
        real_path.parent.mkdir(parents=True, exist_ok=True)
        real_path.write_bytes(encoded)
        rel = real_path.relative_to(self.workspace_root).as_posix()
        return {
            "path": rel,
            "bytes_written": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def edit_file(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        real_path = self.validate_path(path, for_write=True)
        if not real_path.exists() or not real_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        original = real_path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ValueError(f"Target text not found in {path}.")
        if occurrences > 1:
            raise ValueError(f"Target text is ambiguous ({occurrences} occurrences) in {path}.")
        updated = original.replace(old_text, new_text, 1)
        encoded = updated.encode("utf-8")
        if len(encoded) > self.config.max_file_write_bytes:
            raise SandboxViolationError(
                rule="SANDBOX_WRITE_LIMIT",
                message="Edited file exceeds max_file_write_bytes.",
                details={"path": path, "size": len(encoded)},
            )
        real_path.write_bytes(encoded)
        rel = real_path.relative_to(self.workspace_root).as_posix()
        return {
            "path": rel,
            "bytes_written": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "replacements": 1,
        }

    # ------------------------------------------------------------------
    # Command whitelist, env sanitization, process-group timeout & cancel
    # ------------------------------------------------------------------

    def validate_command(self, command: str | list[str]) -> list[str]:
        """Validate command against shell-metacharacter rules and strict prefix whitelist."""
        if isinstance(command, str):
            stripped = command.strip()
            if not stripped:
                raise SandboxViolationError(
                    rule="SANDBOX_COMMAND_WHITELIST",
                    message="Command must be a non-empty string.",
                    details={"command": command},
                )
            for token in SHELL_META_TOKENS:
                if token in stripped:
                    raise SandboxViolationError(
                        rule="SANDBOX_SHELL_INJECTION",
                        message=f"Shell metacharacter '{token}' is forbidden in command.",
                        details={"command": command, "forbidden_token": token},
                    )
            try:
                argv = shlex.split(stripped)
            except ValueError as exc:
                raise SandboxViolationError(
                    rule="SANDBOX_COMMAND_PARSE",
                    message=f"Unable to parse command: {exc}",
                    details={"command": command},
                ) from exc
        elif isinstance(command, list) and command and all(isinstance(x, str) for x in command):
            argv = list(command)
            joined = " ".join(argv)
            for token in SHELL_META_TOKENS:
                if token in joined:
                    raise SandboxViolationError(
                        rule="SANDBOX_SHELL_INJECTION",
                        message=f"Shell metacharacter '{token}' is forbidden in command.",
                        details={"command": argv, "forbidden_token": token},
                    )
        else:
            raise SandboxViolationError(
                rule="SANDBOX_COMMAND_WHITELIST",
                message="Command must be a non-empty string or list of strings.",
                details={"command": command},
            )

        matched_prefix: tuple[str, ...] | None = None
        for prefix in self.config.allowed_command_prefixes:
            if len(argv) >= len(prefix) and tuple(argv[: len(prefix)]) == prefix:
                matched_prefix = prefix
                break

        if matched_prefix is None:
            raise SandboxViolationError(
                rule="SANDBOX_COMMAND_WHITELIST",
                message=(
                    f"Command '{' '.join(argv)}' is not allowed by sandbox whitelist. "
                    f"Allowed prefixes: {[' '.join(p) for p in self.config.allowed_command_prefixes]}"
                ),
                details={"command": argv},
            )

        # Reject path traversal arguments that attempt to reference files outside workspace
        for arg in argv[len(matched_prefix) :]:
            if arg.startswith("-"):
                continue
            if ".." in Path(arg).parts or arg.startswith("/"):
                # Validate that any explicit path argument stays inside workspace
                self.validate_path(arg, for_write=False)

        return argv

    def _resolve_executable(self, argv: list[str]) -> list[str]:
        """Resolve python/python3/pytest to active venv binaries when needed."""
        resolved = list(argv)
        exe = resolved[0]
        venv_bin = Path(sys.executable).parent
        if exe in ("python", "python3"):
            resolved[0] = sys.executable
        elif exe == "pytest":
            candidate = venv_bin / "pytest"
            if candidate.exists():
                resolved[0] = str(candidate)
            else:
                found = shutil.which("pytest")
                if found:
                    resolved[0] = found
        return resolved

    def build_sanitized_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in self.config.env_whitelist:
            if key in os.environ:
                env[key] = os.environ[key]
        # Ensure active python venv bin is on PATH so sub-invocations work cleanly
        venv_bin = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "/usr/bin:/bin")
        if venv_bin not in current_path.split(os.pathsep):
            env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(self.workspace_root)
        env.update(self.config.extra_env)
        return env

    @staticmethod
    def _drain_stream(stream: Any, limit: int, collector: dict[str, Any]) -> None:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        omitted = 0
        try:
            while True:
                data = stream.read(4096)
                if not data:
                    break
                if total < limit:
                    remaining = limit - total
                    if len(data) <= remaining:
                        chunks.append(data)
                        total += len(data)
                    else:
                        chunks.append(data[:remaining])
                        total += remaining
                        omitted += len(data) - remaining
                        truncated = True
                else:
                    omitted += len(data)
                    truncated = True
        finally:
            try:
                stream.close()
            except OSError:
                pass
        text = b"".join(chunks).decode("utf-8", errors="replace")
        if truncated:
            text += f"\n[OUTPUT TRUNCATED: {omitted} bytes omitted]"
        collector["text"] = text
        collector["truncated"] = truncated

    def _terminate_process_group(self, pgid: int, proc: subprocess.Popen[bytes]) -> None:
        """Send SIGTERM then SIGKILL to the entire POSIX process group so child/grandchild processes die."""
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.config.kill_grace_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    def run_command(
        self,
        command: str | list[str],
        *,
        timeout_sec: float | None = None,
        cancel_event: threading.Event | None = None,
        on_start: Callable[[int, int], None] | None = None,
    ) -> CommandResult:
        """Run a whitelisted command inside a dedicated process group with strict timeout/cancel and output limits."""
        argv = self.validate_command(command)
        exec_argv = self._resolve_executable(argv)
        effective_timeout = (
            self.config.default_timeout_sec
            if timeout_sec is None
            else min(max(float(timeout_sec), 0.05), self.config.max_timeout_sec)
        )
        env = self.build_sanitized_env()

        start_time = time.monotonic()
        proc = subprocess.Popen(
            exec_argv,
            cwd=str(self.workspace_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=False,
        )
        pgid = os.getpgid(proc.pid)
        if on_start is not None:
            on_start(proc.pid, pgid)

        stdout_box: dict[str, Any] = {"text": "", "truncated": False}
        stderr_box: dict[str, Any] = {"text": "", "truncated": False}
        t_out = threading.Thread(
            target=self._drain_stream,
            args=(proc.stdout, self.config.max_output_bytes, stdout_box),
            daemon=True,
        )
        t_err = threading.Thread(
            target=self._drain_stream,
            args=(proc.stderr, self.config.max_output_bytes, stderr_box),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        timed_out = False
        cancelled = False
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                self._terminate_process_group(pgid, proc)
                break
            if time.monotonic() - start_time >= effective_timeout:
                timed_out = True
                self._terminate_process_group(pgid, proc)
                break
            time.sleep(0.02)

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        duration = time.monotonic() - start_time
        exit_code = proc.returncode if proc.returncode is not None else -9

        return CommandResult(
            command=argv,
            exit_code=exit_code,
            stdout=stdout_box["text"],
            stderr=stderr_box["text"],
            timed_out=timed_out,
            cancelled=cancelled,
            stdout_truncated=stdout_box["truncated"],
            stderr_truncated=stderr_box["truncated"],
            duration_sec=duration,
            pid=proc.pid,
            pgid=pgid,
        )

    # ------------------------------------------------------------------
    # Workspace & code fingerprinting (for C4 drift & H2 stale test detection)
    # ------------------------------------------------------------------

    def is_code_file(self, rel_path: str) -> bool:
        p = Path(rel_path)
        if p.parts and p.parts[0] == "artifacts":
            return False
        return p.suffix.lower() in CODE_FILE_EXTENSIONS or p.name in ("Makefile", "Dockerfile")

    def compute_workspace_snapshot(self) -> tuple[dict[str, str], str, str]:
        """Return (per_file_sha256_map, overall_workspace_hash, code_only_hash)."""
        file_hashes: dict[str, str] = {}
        code_hashes: dict[str, str] = {}
        for root, dirs, files in os.walk(self.workspace_root, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in DEFAULT_IGNORED_SNAPSHOT_DIRS
                and not self._is_sensitive_name(d, is_dir_component=True)
                and not (Path(root) / d).is_symlink()
            )
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.is_symlink():
                    continue
                if self._is_sensitive_name(fname, is_dir_component=False):
                    continue
                rel = fpath.relative_to(self.workspace_root).as_posix()
                digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
                file_hashes[rel] = digest
                if self.is_code_file(rel):
                    code_hashes[rel] = digest

        workspace_hash = sha256_text(canonical_json(file_hashes))
        code_hash = sha256_text(canonical_json(code_hashes))
        return file_hashes, workspace_hash, code_hash

    def diff_snapshots(
        self,
        expected: dict[str, str],
        actual: dict[str, str],
    ) -> list[dict[str, str]]:
        """Describe added, deleted, or modified files between two workspace snapshots."""
        changes: list[dict[str, str]] = []
        all_paths = sorted(set(expected) | set(actual))
        for path in all_paths:
            if path not in expected:
                changes.append({"path": path, "change": "added"})
            elif path not in actual:
                changes.append({"path": path, "change": "deleted"})
            elif expected[path] != actual[path]:
                changes.append({"path": path, "change": "modified"})
        return changes
