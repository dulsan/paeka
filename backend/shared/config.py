"""
backend/shared/config.py
========================
Loads configuration from config/settings.toml.
Environment variable overrides use the PAEKA_ prefix with __ nesting.

  PAEKA_LLM__BASE_URL=http://...
  PAEKA_RETRIEVAL__ENABLED=true
  PAEKA_SECURITY__CONTENT_SCAN_ENABLED=true
  PAEKA_DEPLOY_MODE=production
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    name: str = "PAEKA"
    version: str = "0.10.0"
    description: str = "Personal AI Engineering & Knowledge Assistant"
    debug: bool = False


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class LLMSettings(BaseModel):
    # ── Provider selection ────────────────────────────────────────────
    # "llama_cpp"  (primary — GGUF-native, CPU+GPU offload)
    # "ollama"     (optional — good for Mac / model management GUI)
    # "sglang"     (optional — large multi-GPU deployments only)
    provider: str = "llama_cpp"

    # ── Connection ────────────────────────────────────────────────────
    base_url: str = "http://localhost:8080/v1"   # llama.cpp default port
    api_key: str = "paeka-local"                 # ignored by llama.cpp

    # ── Model ─────────────────────────────────────────────────────────
    # For llama.cpp:  path to the .gguf file on the HOST filesystem
    #                 (mounted into the container at /models)
    # For ollama:     model tag, e.g. "qwen3:35b"
    # For sglang:     HuggingFace repo id, e.g. "Qwen/Qwen3-14B-Instruct"
    model: str = "paeka-model"          # displayed name / tag / repo id
    model_path: str = ""                # absolute path to .gguf (llama.cpp only)

    # ── Generation parameters ─────────────────────────────────────────
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
    request_timeout: int = 180          # generous for large GGUF models

    # ── System prompt ─────────────────────────────────────────────────
    system_prompt: str = "You are PAEKA, a helpful AI assistant."


class ModelSettings(BaseModel):
    """Configuration for the local model management layer."""
    models_dir: str = "models"            # root directory for .gguf files
    auto_scan: bool = True                # scan on startup


class DatabaseSettings(BaseModel):
    sqlite_path: str = "database/sqlite/paeka.db"


class MemorySettings(BaseModel):
    enabled: bool = True
    max_session_messages: int = 50
    summary_threshold: int = 20
    global_memory_limit: int = 10


class RetrievalSettings(BaseModel):
    enabled: bool = False
    weaviate_url: str = "http://localhost:8080"
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cuda"
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cuda"
    top_k: int = 20
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.75
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_hops: int = 2


class IngestionSettings(BaseModel):
    default_parser: str = "auto"
    max_file_bytes: int = 104_857_600
    supported_types: list[str] = Field(default_factory=lambda: [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/markdown",
        "text/html",
    ])
    versioning_enabled: bool = True
    dedup_threshold: float = 0.97


class KnowledgeGraphSettings(BaseModel):
    enabled: bool = False
    db_path: str = "database/sqlite/knowledge_graph.db"
    extraction_batch_size: int = 5
    refinement_passes: list[str] = Field(default_factory=lambda: [
        "merge_duplicates", "prune_weak_edges", "validate_types"
    ])
    min_edge_confidence: float = 0.6
    entity_types: list[str] = Field(default_factory=lambda: [
        "Concept", "Algorithm", "Method", "Framework",
        "Person", "Organisation", "Dataset", "Paper",
        "Tool", "Language", "Variable", "Equation",
    ])
    relation_types: list[str] = Field(default_factory=lambda: [
        "IS_A", "PART_OF", "USES", "IMPLEMENTS", "EXTENDS",
        "DEFINED_IN", "RELATED_TO", "AUTHORED_BY", "CITES",
        "CONTRASTS_WITH", "DEPENDS_ON", "PRODUCES",
    ])


class SandboxSettings(BaseModel):
    enabled: bool = True
    default_timeout: int = 30
    max_timeout: int = 60
    memory_limit: str = "256m"
    cpu_limit: str = "1.0"


class IterationSettings(BaseModel):
    default_score_threshold: float = 0.85
    default_max_iterations: int = 4


class ToolCallingSettings(BaseModel):
    default_max_retries: int = 3
    default_max_iterations: int = 5


class SkillsSettings(BaseModel):
    enabled: bool = True
    skills_dir: str = "backend/skills/definitions"


class ToolsSettings(BaseModel):
    web_search_enabled: bool = False
    searxng_url: str = "http://localhost:8888"
    searxng_categories: str = "general"
    searxng_language: str = "en"
    searxng_max_results: int = 5


class SecuritySettings(BaseModel):
    """
    Content security, authentication, and rate limiting.

    These are primarily set via environment variables (from .env)
    rather than settings.toml, since auth tokens should never be
    committed to version control.
    """
    # Content scanning
    content_scan_enabled: bool = True
    strict_mode: bool = False          # promote WARN→BLOCK; recommended for Mode 3

    # Authentication
    enabled: bool = False              # set true for LAN / internet deployments
    token: str = ""                    # set via PAEKA_AUTH__TOKEN env var

    # Rate limiting
    rate_limit_enabled: bool = False   # set true for LAN / internet deployments
    chat_rpm: int = 20
    upload_rpm: int = 10
    default_rpm: int = 120


class DeploymentSettings(BaseModel):
    """
    Deployment context — controls which security features activate.

    "development"  — localhost, no auth, no rate limiting
    "lan"          — local network, auth + self-signed TLS
    "production"   — internet-facing, auth + Let's Encrypt TLS + rate limiting
    """
    mode: str = "development"   # development | lan | production
    domain: str = "localhost"

    @property
    def is_production(self) -> bool:
        return self.mode == "production"

    @property
    def is_lan_or_production(self) -> bool:
        return self.mode in ("lan", "production")


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "rich"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAEKA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    knowledge_graph: KnowledgeGraphSettings = Field(default_factory=KnowledgeGraphSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    deploy: DeploymentSettings = Field(default_factory=DeploymentSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    iteration: IterationSettings = Field(default_factory=IterationSettings)
    tool_calling: ToolCallingSettings = Field(default_factory=ToolCallingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Top-level deploy mode shortcut (PAEKA_DEPLOY_MODE=production)
    deploy_mode: str = "development"

    def model_post_init(self, __context: Any) -> None:
        # Sync top-level PAEKA_DEPLOY_MODE into deploy.mode
        if self.deploy_mode != "development" and self.deploy.mode == "development":
            object.__setattr__(self, "deploy",
                DeploymentSettings(mode=self.deploy_mode, domain=self.deploy.domain))


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config/settings.toml") -> Settings:
    """Return the cached Settings singleton."""
    raw = _load_toml(Path(config_path))
    return Settings(**raw)
