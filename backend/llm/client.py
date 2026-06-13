"""
backend/llm/client.py
======================
Backward-compatibility shim.

LLMClient is now an alias for LLMProvider so existing code that does:

    from backend.llm.client import LLMClient

continues to work.  New code should import LLMProvider from backend.llm.base
and use create_provider() from backend.llm.factory.
"""

from backend.llm.base import LLMProvider, Message

# Alias kept for backward compatibility
LLMClient = LLMProvider

__all__ = ["LLMClient", "LLMProvider", "Message"]
