"""
backend/api/routes/knowledge.py
================================
Knowledge graph management endpoints.

POST   /api/knowledge/extract/{document_id}  — extract graph from a document
POST   /api/knowledge/refine                 — run refinement passes
GET    /api/knowledge/stats                  — node/edge counts
GET    /api/knowledge/nodes                  — list nodes (filterable by type)
GET    /api/knowledge/nodes/{id}             — get node + neighbours
DELETE /api/knowledge/nodes/{id}             — delete a node
GET    /api/knowledge/query                  — graph context for a query string
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NodeOut(BaseModel):
    id: str
    label: str
    entity_type: str
    description: str
    confidence: float


class EdgeOut(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str
    confidence: float


class NodeDetailOut(NodeOut):
    outgoing: list[EdgeOut]
    incoming: list[EdgeOut]


class GraphStatsOut(BaseModel):
    nodes: int
    edges: int
    types: dict[str, int]


class RefineResponse(BaseModel):
    passes: dict[str, int]


class ExtractResponse(BaseModel):
    document_id: str
    entities_found: int
    relations_found: int


class GraphQueryResponse(BaseModel):
    query: str
    entities: list[dict]
    subgraph: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _require_kg(request: Request):
    kg = request.app.state.kg_repo
    if kg is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph not enabled. Set [knowledge_graph] enabled = true.",
        )
    return kg


@router.post("/knowledge/extract/{document_id}", response_model=ExtractResponse)
async def extract_graph(document_id: str, request: Request) -> ExtractResponse:
    """Run KG extraction over all chunks of a document."""
    _require_kg(request)
    from backend.ingestion.repository import DocumentRepository

    repo = DocumentRepository(request.app.state.db)
    doc = await repo.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_raw = await repo.get_chunks(document_id)
    if not chunks_raw:
        raise HTTPException(status_code=400, detail="Document has no chunks to extract from")

    from backend.retrieval.chunker import TextChunk
    chunks = [
        TextChunk(
            content=c.content,
            chunk_index=c.chunk_index,
            heading=c.heading or "",
            page=c.page or 0,
            element_type=c.element_type,
        )
        for c in chunks_raw
    ]

    extractor = request.app.state.kg_extractor
    result = await extractor.extract_from_chunks(chunks, source_doc=doc.filename)

    return ExtractResponse(
        document_id=document_id,
        entities_found=len(result.entities),
        relations_found=len(result.relations),
    )


@router.post("/knowledge/refine", response_model=RefineResponse)
async def refine_graph(request: Request) -> RefineResponse:
    """Run all configured refinement passes over the graph."""
    _require_kg(request)
    refiner = request.app.state.kg_refiner
    passes = await refiner.run_all_passes()

    # Reload graph cache in retriever after refinement
    kg_retriever = request.app.state.kg_retriever
    if kg_retriever:
        await kg_retriever.load_graph()

    return RefineResponse(passes=passes)


@router.get("/knowledge/stats", response_model=GraphStatsOut)
async def graph_stats(request: Request) -> GraphStatsOut:
    kg = _require_kg(request)
    stats = await kg.stats()
    return GraphStatsOut(**stats)


@router.get("/knowledge/nodes", response_model=list[NodeOut])
async def list_nodes(
    request: Request,
    entity_type: str | None = Query(None, description="Filter by entity type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> list[NodeOut]:
    kg = _require_kg(request)
    nodes = await kg.list_nodes(entity_type=entity_type, min_confidence=min_confidence)
    return [NodeOut(**n.__dict__) for n in nodes]


@router.get("/knowledge/nodes/{node_id}", response_model=NodeDetailOut)
async def get_node(node_id: str, request: Request) -> NodeDetailOut:
    kg = _require_kg(request)
    node = await kg.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    outgoing, incoming = await kg.get_neighbours(node_id)
    return NodeDetailOut(
        **node.__dict__,
        outgoing=[EdgeOut(**e.__dict__) for e in outgoing],
        incoming=[EdgeOut(**e.__dict__) for e in incoming],
    )


@router.delete("/knowledge/nodes/{node_id}", status_code=204)
async def delete_node(node_id: str, request: Request) -> None:
    kg = _require_kg(request)
    node = await kg.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    await kg.delete_node(node_id)


@router.get("/knowledge/query", response_model=GraphQueryResponse)
async def query_graph(
    q: str = Query(..., description="Query text to match against graph entities"),
    request: Request = None,  # type: ignore[assignment]
) -> GraphQueryResponse:
    """Return graph context for a free-text query."""
    _require_kg(request)
    kg_ret = request.app.state.kg_retriever
    if kg_ret is None:
        raise HTTPException(status_code=503, detail="Graph retriever not initialised")

    gc = await kg_ret.query(q)
    return GraphQueryResponse(
        query=q,
        entities=gc.entities,
        subgraph=gc.subgraph,
    )
