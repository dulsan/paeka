"""
tests/unit/test_knowledge_graph.py
====================================
Unit tests for the knowledge graph repository.
Uses an in-memory SQLite database — no mocking required.
"""

from __future__ import annotations

import pytest
from backend.shared.database import Database
from backend.knowledge.graph import KnowledgeGraphRepository


@pytest.fixture
async def db():
    d = Database(":memory:")
    await d.connect()
    yield d
    await d.close()


@pytest.fixture
async def repo(db):
    return KnowledgeGraphRepository(db)


@pytest.mark.anyio
async def test_upsert_and_find_node(repo):
    node = await repo.upsert_node(
        label="Transformer",
        entity_type="Algorithm",
        description="Attention-based model",
    )
    assert node.id
    assert node.label == "Transformer"

    found = await repo.find_node_by_label("transformer")  # case-insensitive
    assert found is not None
    assert found.id == node.id


@pytest.mark.anyio
async def test_upsert_node_merges(repo):
    await repo.upsert_node("BERT", "Algorithm", description="First desc", confidence=0.7)
    await repo.upsert_node("BERT", "Algorithm", description="Second desc", confidence=0.9)

    node = await repo.find_node_by_label("BERT")
    assert node is not None
    assert node.confidence == pytest.approx(0.9)
    assert "First desc" in node.description
    assert "Second desc" in node.description


@pytest.mark.anyio
async def test_upsert_edge(repo):
    a = await repo.upsert_node("BERT", "Algorithm")
    b = await repo.upsert_node("Transformer", "Algorithm")
    edge = await repo.upsert_edge(a.id, b.id, "EXTENDS", confidence=0.85)
    assert edge.relation_type == "EXTENDS"
    assert edge.confidence == pytest.approx(0.85)


@pytest.mark.anyio
async def test_self_loop_rejected(repo):
    node = await repo.upsert_node("SelfRef", "Concept")
    with pytest.raises(ValueError, match="Self-loops"):
        await repo.upsert_edge(node.id, node.id, "RELATED_TO")


@pytest.mark.anyio
async def test_get_neighbours(repo):
    src = await repo.upsert_node("PyTorch", "Framework")
    tgt = await repo.upsert_node("Python", "Language")
    await repo.upsert_edge(src.id, tgt.id, "USES")

    outgoing, incoming = await repo.get_neighbours(src.id)
    assert len(outgoing) == 1
    assert outgoing[0].target_id == tgt.id

    _, tgt_incoming = await repo.get_neighbours(tgt.id)
    assert len(tgt_incoming) == 1


@pytest.mark.anyio
async def test_merge_nodes(repo):
    a = await repo.upsert_node("GPT", "Algorithm")
    b = await repo.upsert_node("GPT-2", "Algorithm")
    ctx = await repo.upsert_node("NLP", "Concept")
    await repo.upsert_edge(b.id, ctx.id, "RELATED_TO")

    await repo.merge_nodes(keep_id=a.id, drop_id=b.id)

    # b should be gone
    assert await repo.get_node(b.id) is None

    # edge should now point to a
    outgoing, _ = await repo.get_neighbours(a.id)
    assert any(e.target_id == ctx.id for e in outgoing)


@pytest.mark.anyio
async def test_delete_weak_edges(repo):
    a = await repo.upsert_node("A", "Concept")
    b = await repo.upsert_node("B", "Concept")
    c = await repo.upsert_node("C", "Concept")
    await repo.upsert_edge(a.id, b.id, "RELATED_TO", confidence=0.9)
    await repo.upsert_edge(a.id, c.id, "RELATED_TO", confidence=0.3)

    deleted = await repo.delete_weak_edges(threshold=0.5)
    assert deleted == 1

    edges = await repo.list_edges()
    assert all(e.confidence >= 0.5 for e in edges)


@pytest.mark.anyio
async def test_stats(repo):
    await repo.upsert_node("X", "Concept")
    await repo.upsert_node("Y", "Algorithm")
    stats = await repo.stats()
    assert stats["nodes"] == 2
    assert stats["types"]["Concept"] == 1
    assert stats["types"]["Algorithm"] == 1


@pytest.mark.anyio
async def test_to_networkx(repo):
    a = await repo.upsert_node("A", "Concept")
    b = await repo.upsert_node("B", "Concept")
    await repo.upsert_edge(a.id, b.id, "RELATED_TO")

    g = await repo.to_networkx()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
