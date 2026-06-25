"""
backend/shared/config.py
========================
Application configuration.

Change: RetrievalSettings gains qdrant_url field.
        weaviate_url kept for backward compat but no longer used by app.py.
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
    version: str = "0.11.3"
    description: str = "Personal AI Engineering & Knowledge Assistant"
    debug: bool = False


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class LLMSettings(BaseModel):
    # "ollama"    (default - install from https://ollama.com/download/windows)
    # "llama_cpp" (legacy - native binary, more config required)
    # "sglang"    (multi-GPU deployments)
    provider: str = "ollama"

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"

    model: str = "paeka-qwen"
    model_path: str = ""

    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = True
    request_timeout: int = 180

    system_prompt: str = "You are PAEKA, a helpful AI assistant."


class ModelSettings(BaseModel):
    models_dir: str = "models"
    auto_scan: bool = True


class DatabaseSettings(BaseModel):
    sqlite_path: str = "database/sqlite/paeka.db"


class MemorySettings(BaseModel):
    enabled: bool = True
    max_session_messages: int = 50
    summary_threshold: int = 20
    global_memory_limit: int = 10


class RetrievalSettings(BaseModel):
    enabled: bool = False

    # Qdrant (primary vector store)
    qdrant_url: str = "http://localhost:6333"

    # Weaviate kept for backward compat - not used by current app.py
    weaviate_url: str = "http://localhost:8090"

    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cpu"

    top_k: int = 20
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.75

    # 1600 chars ~ 400 tokens for bge-m3 (512-token limit)
    chunk_size: int = 1600
    chunk_overlap: int = 200

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
    content_scan_enabled: bool = True
    strict_mode: bool = False
    enabled: bool = False
    token: str = ""
    rate_limit_enabled: bool = False
    chat_rpm: int = 20
    upload_rpm: int = 10
    default_rpm: int = 120


class DeploymentSettings(BaseModel):
    mode: str = "development"
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

    deploy_mode: str = "development"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Default order is (init, env, dotenv, secrets) -- earlier wins.
        # That makes the TOML dict passed in via Settings(**raw) in
        # get_settings() unconditionally beat PAEKA_* env vars, since it
        # arrives as init kwargs. Env vars and .env are meant to override
        # the static TOML file, so they need to come first instead.
        return env_settings, dotenv_settings, init_settings, file_secret_settings

    def model_post_init(self, __context: Any) -> None:
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
    raw = _load_toml(Path(config_path))
    return Settings(**raw)
