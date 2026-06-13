"""
backend/skills/manager.py
==========================
Skills framework — folder-based, following the Anthropic Claude Code pattern.

A skill is a DIRECTORY (not just a single file) containing:
  SKILL.md        — required: description, trigger keywords, instructions, gotchas
  config.toml     — optional: temperature, max_tokens, tags, setup prompts
  scripts/        — optional: helper scripts the agent can reference
  references/     — optional: API docs, snippets, examples
  assets/         — optional: templates, data files

This mirrors the Claude Code skills architecture described in:
https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills

SKILL.md frontmatter (TOML fenced block at top of file):
  ---
  name = "my_skill"
  description = "Trigger description for auto-selection"
  tags = ["engineering"]
  temperature = 0.3
  max_tokens = 4096
  ---

The description is written FOR the model (trigger/routing text),
not for humans — matching the blog's guidance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n(.+?)\n---\n", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str           # written for model routing, not humans
    system_prompt: str         # full content of SKILL.md after frontmatter
    tags: list[str] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    skill_dir: Path | None = None

    def list_scripts(self) -> list[Path]:
        """Return paths to all scripts in the skill's scripts/ directory."""
        if self.skill_dir is None:
            return []
        scripts_dir = self.skill_dir / "scripts"
        if not scripts_dir.exists():
            return []
        return sorted(scripts_dir.iterdir())

    def list_references(self) -> list[Path]:
        """Return paths to all files in the skill's references/ directory."""
        if self.skill_dir is None:
            return []
        refs_dir = self.skill_dir / "references"
        if not refs_dir.exists():
            return []
        return sorted(refs_dir.iterdir())

    def read_asset(self, name: str) -> str | None:
        """Read a named file from the skill's assets/ directory."""
        if self.skill_dir is None:
            return None
        asset = self.skill_dir / "assets" / name
        if not asset.exists():
            return None
        return asset.read_text(encoding="utf-8")


class SkillsManager:
    """
    Loads skill definitions from a directory of skill folders.

    Each subdirectory is a skill if it contains a SKILL.md file.
    """

    def __init__(self, skills_dir: str = "backend/skills/definitions") -> None:
        self._dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def load(self) -> None:
        """Scan skills_dir and load every skill folder."""
        self._skills.clear()
        if not self._dir.exists():
            logger.warning("Skills directory not found: %s", self._dir)
            self._loaded = True
            return

        for item in sorted(self._dir.iterdir()):
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                skill = _load_skill_dir(item, skill_md)
                self._skills[skill.name] = skill
                logger.debug("Loaded skill: %s from %s", skill.name, item.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load skill from %s: %s", item.name, exc)

        self._loaded = True
        logger.info("Skills loaded: %d from %s", len(self._skills), self._dir)

    def list_skills(self) -> list[Skill]:
        if not self._loaded:
            self.load()
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        if not self._loaded:
            self.load()
        return self._skills.get(name)

    def reload(self) -> None:
        self._loaded = False
        self.load()

    def skill_listing(self) -> str:
        """
        Format the skill list as a compact routing table for injection
        into the system prompt (model-readable, not human-readable).
        """
        if not self._loaded:
            self.load()
        if not self._skills:
            return ""
        lines = ["Available skills (use when task matches description):"]
        for s in self._skills.values():
            lines.append(f"  [{s.name}] {s.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_skill_dir(skill_dir: Path, skill_md: Path) -> Skill:
    """Parse a skill directory and return a Skill object."""
    raw = skill_md.read_text(encoding="utf-8")

    # Extract TOML frontmatter
    meta: dict = {}
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        import tomllib
        try:
            meta = tomllib.loads(m.group(1))
        except Exception:  # noqa: BLE001
            pass
        body = raw[m.end():]

    # Also check for a separate config.toml (overrides frontmatter)
    config_toml = skill_dir / "config.toml"
    if config_toml.exists():
        import tomllib
        with config_toml.open("rb") as fh:
            config = tomllib.load(fh)
        meta.update(config)

    name = meta.get("name") or skill_dir.name
    description = meta.get("description", f"Skill: {name}")

    return Skill(
        name=str(name),
        description=str(description),
        system_prompt=body.strip(),
        tags=list(meta.get("tags", [])),
        temperature=meta.get("temperature"),
        max_tokens=meta.get("max_tokens"),
        skill_dir=skill_dir,
    )
