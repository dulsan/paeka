"""
tests/unit/test_skills.py
==========================
Unit tests for the folder-based skills manager.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile


def _create_skill_dir(base: Path, name: str, skill_md: str) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return skill_dir


def test_load_skill_with_frontmatter():
    from backend.skills.manager import SkillsManager

    md = """---
name = "test_skill"
description = "Use for testing."
tags = ["test"]
temperature = 0.3
---

## Test Skill Mode

Do the thing.
"""
    with tempfile.TemporaryDirectory() as tmp:
        _create_skill_dir(Path(tmp), "test_skill", md)
        mgr = SkillsManager(tmp)
        mgr.load()

        skills = mgr.list_skills()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "test_skill"
        assert s.description == "Use for testing."
        assert s.temperature == pytest.approx(0.3)
        assert "test" in s.tags
        assert "Do the thing." in s.system_prompt


def test_skill_name_falls_back_to_dirname():
    from backend.skills.manager import SkillsManager

    # No frontmatter — name should come from directory name
    md = "## My Skill\n\nInstructions here."
    with tempfile.TemporaryDirectory() as tmp:
        _create_skill_dir(Path(tmp), "fallback_skill", md)
        mgr = SkillsManager(tmp)
        mgr.load()

        skill = mgr.get_skill("fallback_skill")
        assert skill is not None
        assert skill.name == "fallback_skill"


def test_missing_skill_returns_none():
    from backend.skills.manager import SkillsManager
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillsManager(tmp)
        mgr.load()
        assert mgr.get_skill("nonexistent") is None


def test_reload_picks_up_new_skill():
    from backend.skills.manager import SkillsManager

    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillsManager(tmp)
        mgr.load()
        assert len(mgr.list_skills()) == 0

        # Add a new skill while manager is loaded
        md = "---\nname = \"new\"\ndescription = \"New skill.\"\n---\nDo new things."
        _create_skill_dir(Path(tmp), "new", md)
        mgr.reload()

        assert len(mgr.list_skills()) == 1
        assert mgr.get_skill("new") is not None


def test_empty_skills_dir():
    from backend.skills.manager import SkillsManager
    with tempfile.TemporaryDirectory() as tmp:
        mgr = SkillsManager(tmp)
        mgr.load()
        assert mgr.list_skills() == []


def test_skill_listing_for_model():
    from backend.skills.manager import SkillsManager

    md = "---\nname = \"eng\"\ndescription = \"Engineering analysis.\"\n---\nDo engineering."
    with tempfile.TemporaryDirectory() as tmp:
        _create_skill_dir(Path(tmp), "eng", md)
        mgr = SkillsManager(tmp)
        listing = mgr.skill_listing()
        assert "[eng]" in listing
        assert "Engineering analysis." in listing


def test_skill_scripts_and_references():
    from backend.skills.manager import SkillsManager

    md = "---\nname = \"scripted\"\ndescription = \"Has scripts.\"\n---\nContent."
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = _create_skill_dir(Path(tmp), "scripted", md)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "helper.py").write_text("print('hello')")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "api.md").write_text("# API")

        mgr = SkillsManager(tmp)
        mgr.load()
        skill = mgr.get_skill("scripted")
        assert skill is not None
        assert len(skill.list_scripts()) == 1
        assert len(skill.list_references()) == 1
