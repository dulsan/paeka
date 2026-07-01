"""
backend/security/content.py
============================
Content security layer — applied at three points:

  1. INGEST TIME   : before any document chunk reaches Weaviate or the LLM
  2. CHAT INPUT    : before the user message enters the agentic pipeline
  3. CHAT OUTPUT   : before the LLM reply is persisted or streamed to the client

Threat model:
  - Prompt injection via crafted document content
    ("Ignore previous instructions and…")
  - Indirect prompt injection via web search results
  - Script/shell injection in LLM output (code blocks containing rm -rf etc.)
  - Data exfiltration attempts embedded in document text

Design:
  - Pattern matching covers the vast majority of real-world naive attacks.
  - Severity levels: BLOCK (reject entirely), WARN (log + sanitise), PASS.
  - All detections are logged for audit.
  - No LLM call required — fast, synchronous, zero latency overhead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    PASS  = "pass"
    WARN  = "warn"
    BLOCK = "block"


@dataclass
class ScanResult:
    severity: Severity
    findings: list[str] = field(default_factory=list)
    sanitised_text: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.severity == Severity.BLOCK

    @property
    def is_clean(self) -> bool:
        return self.severity == Severity.PASS


# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

# BLOCK-level: clear attempts to hijack the system prompt or identity
_INJECTION_BLOCK: list[re.Pattern] = [p for p in [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(your\s+)?(system\s+)?(prompt|instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!PAEKA)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you('ve)?\s+)?(been\s+)?(told|learned|trained)", re.I),
    re.compile(r"(new|updated|revised)\s+system\s+prompt\s*:", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[INST\]|\[SYS\]|<<SYS>>", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(?!PAEKA)", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?!PAEKA)", re.I),
    re.compile(r"jailbreak|DAN\s+mode|developer\s+mode", re.I),
    re.compile(r"reveal\s+(your\s+)?(full\s+|complete\s+|entire\s+)?(system\s+prompt|instructions?|training)", re.I),
    re.compile(r"print\s+(your\s+)?(full\s+|complete\s+|entire\s+)?(system\s+prompt|full\s+prompt)", re.I),
]]

# WARN-level: suspicious but may be legitimate (log + tag, don't block)
_INJECTION_WARN: list[re.Pattern] = [p for p in [
    re.compile(r"do\s+not\s+(follow|obey|comply)", re.I),
    re.compile(r"override\s+(your\s+)?(safety|guidelines?|rules?)", re.I),
    re.compile(r"without\s+(any\s+)?(restrictions?|limitations?|filters?)", re.I),
    re.compile(r"bypass\s+(the\s+)?(filter|safety|restriction)", re.I),
]]


# ---------------------------------------------------------------------------
# Dangerous output patterns (shell/script injection in LLM responses)
# ---------------------------------------------------------------------------

# Shell commands that could cause damage if blindly executed
_DANGEROUS_SHELL: list[re.Pattern] = [p for p in [
    re.compile(r"\brm\s+-rf?\s+/", re.I),
    re.compile(r"\bdd\s+if=/dev/(zero|random|urandom)\s+of=", re.I),
    re.compile(r":(){ :|:& };:", ),               # fork bomb
    re.compile(r"\bmkfs\.", re.I),
    re.compile(r"\bformat\s+c:", re.I),
    re.compile(r">\s*/dev/sd[a-z]", re.I),
    re.compile(r"\bchmod\s+-R\s+777\s+/", re.I),
    re.compile(r"curl\s+.+\|\s*(bash|sh|python)", re.I),
    re.compile(r"wget\s+.+\|\s*(bash|sh|python)", re.I),
    re.compile(r"base64\s+-d\s+.+\|\s*(bash|sh)", re.I),
]]

# Exfiltration patterns in output
_EXFILTRATION: list[re.Pattern] = [p for p in [
    re.compile(r"(send|POST|GET|fetch|curl|wget).{0,60}(password|api_key|token|secret)", re.I),
    re.compile(r"http[s]?://[^\s]{5,}.{0,30}(passwd|shadow|\.env|secrets?)", re.I),
]]


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class ContentScanner:
    """
    Fast, synchronous content scanner.

    Usage:
        scanner = ContentScanner(settings)

        # At ingestion time (per chunk)
        result = scanner.scan_input(chunk_text, source="document.pdf")
        if result.is_blocked:
            raise ValueError(f"Blocked: {result.findings}")

        # At chat input time
        result = scanner.scan_input(user_message, source="user")

        # At chat output time
        result = scanner.scan_output(llm_reply)
        safe_reply = result.sanitised_text
    """

    def __init__(self, enabled: bool = True, strict_mode: bool = False) -> None:
        """
        Parameters
        ----------
        enabled:
            Set False to bypass all scanning (development mode).
        strict_mode:
            True = WARN-level findings are promoted to BLOCK.
            Recommended for internet-facing / Mode 3 deployments.
        """
        self._enabled   = enabled
        self._strict    = strict_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_input(self, text: str, source: str = "unknown") -> ScanResult:
        """
        Scan user input or document content for injection attempts.

        Parameters
        ----------
        text:
            The text to scan.
        source:
            Label for logging (e.g. "user", "document.pdf", "web:example.com").
        """
        if not self._enabled or not text:
            return ScanResult(severity=Severity.PASS, sanitised_text=text)

        findings: list[str] = []
        severity = Severity.PASS

        for pattern in _INJECTION_BLOCK:
            if pattern.search(text):
                findings.append(f"BLOCK pattern matched: {pattern.pattern[:60]}")
                severity = Severity.BLOCK

        if severity != Severity.BLOCK:
            for pattern in _INJECTION_WARN:
                if pattern.search(text):
                    findings.append(f"WARN pattern matched: {pattern.pattern[:60]}")
                    severity = Severity.WARN

        # In strict mode, warnings become blocks
        if self._strict and severity == Severity.WARN:
            severity = Severity.BLOCK

        if findings:
            logger.warning(
                "Content scan [%s] source='%s' severity=%s findings=%s",
                severity.upper(), source, severity, findings,
            )

        return ScanResult(
            severity=severity,
            findings=findings,
            sanitised_text=text,   # input is not modified, only blocked
        )

    def scan_output(self, text: str) -> ScanResult:
        """
        Scan LLM output for dangerous shell commands or exfiltration attempts.

        Dangerous patterns within code fences are redacted with a warning
        rather than blocking the entire response.
        """
        if not self._enabled or not text:
            return ScanResult(severity=Severity.PASS, sanitised_text=text)

        findings: list[str] = []
        severity = Severity.PASS
        sanitised = text

        for pattern in _DANGEROUS_SHELL:
            if pattern.search(text):
                findings.append(f"Dangerous shell pattern: {pattern.pattern[:60]}")
                severity = Severity.WARN
                sanitised = pattern.sub("[REDACTED: potentially dangerous command]", sanitised)

        for pattern in _EXFILTRATION:
            if pattern.search(text):
                findings.append(f"Potential exfiltration pattern: {pattern.pattern[:60]}")
                severity = Severity.WARN
                sanitised = pattern.sub("[REDACTED: potential data exfiltration]", sanitised)

        if findings:
            logger.warning(
                "Output scan: severity=%s findings=%s", severity, findings
            )

        return ScanResult(
            severity=severity,
            findings=findings,
            sanitised_text=sanitised,
        )

    def scan_web_result(self, text: str, url: str = "") -> ScanResult:
        """
        Scan a web search result — applies the same injection checks
        but with context-appropriate logging (lower trust tier).
        """
        result = self.scan_input(text, source=f"web:{url}")
        if result.severity == Severity.WARN and not self._strict:
            # Web results are untrusted by design — auto-promote warns to blocks
            result = ScanResult(
                severity=Severity.BLOCK,
                findings=result.findings,
                sanitised_text=result.sanitised_text,
            )
        return result
