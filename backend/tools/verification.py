"""
backend/tools/verification.py
==============================
Static code verification tools the LLM agent can call.

Tools exposed:
  lint_python(code, filename)  — Ruff linting
  typecheck_python(code)       — Pyright type checking
  format_python(code)          — Ruff auto-format (returns fixed code)

These run as subprocesses, writing code to a temp file and capturing output.
All tools are synchronous and wrapped in asyncio.to_thread for non-blocking use.

Usage in the agent:
  The LLM can request verification as a tool call. The agent routes it
  to the appropriate function and injects results back as a context passage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    tool: str
    passed: bool
    output: str          # human-readable summary
    issues: list[dict]   # structured issues [{line, col, code, message}]
    fixed_code: str = "" # populated by format tools


class CodeVerifier:
    """
    Runs static analysis tools on code snippets.
    All methods are async-safe (use asyncio.to_thread internally).
    """

    # ------------------------------------------------------------------
    # Python — Ruff
    # ------------------------------------------------------------------

    async def lint_python(
        self, code: str, filename: str = "snippet.py"
    ) -> VerificationResult:
        """Run Ruff linting on a Python code string."""
        if not shutil.which("ruff"):
            return VerificationResult(
                tool="ruff",
                passed=False,
                output="Ruff not found. Install with: uv add ruff",
                issues=[],
            )
        return await asyncio.to_thread(self._run_ruff_check, code, filename)

    async def format_python(self, code: str) -> VerificationResult:
        """Run Ruff formatter on a Python code string. Returns fixed code."""
        if not shutil.which("ruff"):
            return VerificationResult(
                tool="ruff-format",
                passed=False,
                output="Ruff not found.",
                issues=[],
                fixed_code=code,
            )
        return await asyncio.to_thread(self._run_ruff_format, code)

    async def typecheck_python(self, code: str) -> VerificationResult:
        """Run Pyright type checking on a Python code string."""
        if not shutil.which("pyright"):
            return VerificationResult(
                tool="pyright",
                passed=False,
                output="Pyright not found. Install with: uv add pyright",
                issues=[],
            )
        return await asyncio.to_thread(self._run_pyright, code)

    # ------------------------------------------------------------------
    # Private sync helpers (run in thread)
    # ------------------------------------------------------------------

    def _run_ruff_check(self, code: str, filename: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="paeka_lint_", mode="w",
            encoding="utf-8", delete=False
        ) as tf:
            tf.write(code)
            tmp_path = Path(tf.name)

        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", str(tmp_path)],
                capture_output=True, text=True, timeout=15,
            )
            issues = []
            try:
                raw_issues = json.loads(result.stdout or "[]")
                for issue in raw_issues:
                    issues.append({
                        "line":    issue.get("location", {}).get("row", 0),
                        "col":     issue.get("location", {}).get("column", 0),
                        "code":    issue.get("code", ""),
                        "message": issue.get("message", ""),
                        "fix":     issue.get("fix", {}).get("message", "") if issue.get("fix") else "",
                    })
            except (json.JSONDecodeError, TypeError):
                pass

            passed = result.returncode == 0
            summary = f"Ruff: {'✓ No issues' if passed else f'{len(issues)} issue(s) found'}"
            if not passed and issues:
                summary += "\n" + "\n".join(
                    f"  L{i['line']}:{i['col']} [{i['code']}] {i['message']}"
                    for i in issues[:10]
                )
            return VerificationResult(
                tool="ruff", passed=passed, output=summary, issues=issues
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                tool="ruff", passed=False,
                output="Ruff timed out (15s limit)", issues=[]
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_ruff_format(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="paeka_fmt_", mode="w",
            encoding="utf-8", delete=False
        ) as tf:
            tf.write(code)
            tmp_path = Path(tf.name)

        try:
            subprocess.run(
                ["ruff", "format", str(tmp_path)],
                capture_output=True, text=True, timeout=15, check=False,
            )
            fixed = tmp_path.read_text(encoding="utf-8")
            changed = fixed != code
            return VerificationResult(
                tool="ruff-format",
                passed=True,
                output=f"Ruff format: {'changes applied' if changed else 'already formatted'}",
                issues=[],
                fixed_code=fixed,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                tool="ruff-format", passed=False,
                output="Ruff format timed out", issues=[], fixed_code=code,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_pyright(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="paeka_pyright_", mode="w",
            encoding="utf-8", delete=False
        ) as tf:
            tf.write(code)
            tmp_path = Path(tf.name)

        try:
            result = subprocess.run(
                ["pyright", "--outputjson", str(tmp_path)],
                capture_output=True, text=True, timeout=30,
            )
            issues = []
            try:
                data = json.loads(result.stdout or "{}")
                for diag in data.get("generalDiagnostics", []):
                    rng = diag.get("range", {})
                    start = rng.get("start", {})
                    issues.append({
                        "line":     start.get("line", 0) + 1,
                        "col":      start.get("character", 0),
                        "code":     diag.get("rule", ""),
                        "message":  diag.get("message", ""),
                        "severity": diag.get("severity", "error"),
                    })
            except (json.JSONDecodeError, TypeError):
                pass

            errors   = [i for i in issues if i.get("severity") == "error"]
            warnings = [i for i in issues if i.get("severity") == "warning"]
            passed   = len(errors) == 0
            summary  = (
                f"Pyright: {len(errors)} error(s), {len(warnings)} warning(s)"
                if issues else "Pyright: ✓ No issues"
            )
            if errors:
                summary += "\n" + "\n".join(
                    f"  L{i['line']} [{i['code']}] {i['message']}"
                    for i in errors[:10]
                )
            return VerificationResult(
                tool="pyright", passed=passed, output=summary, issues=issues
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                tool="pyright", passed=False,
                output="Pyright timed out (30s limit)", issues=[]
            )
        finally:
            tmp_path.unlink(missing_ok=True)


# Singleton
_verifier = CodeVerifier()


async def lint_python(code: str, filename: str = "snippet.py") -> VerificationResult:
    return await _verifier.lint_python(code, filename)


async def format_python(code: str) -> VerificationResult:
    return await _verifier.format_python(code)


async def typecheck_python(code: str) -> VerificationResult:
    return await _verifier.typecheck_python(code)
