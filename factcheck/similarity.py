import os
from typing import Any, Dict, List, Optional, Tuple

# Optional clients
try:
    import pinecone
except Exception:  # pragma: no cover
    pinecone = None  # type: ignore

try:
    from pymilvus import MilvusClient
except Exception:  # pragma: no cover
    MilvusClient = None  # type: ignore


SIM_THRESHOLD = float(os.environ.get("FACTCHECK_SIM_THRESHOLD", "0.85"))


def query_pinecone(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX", "factchecks")
    if not api_key or not pinecone:
        return []
    pinecone.init(api_key=api_key)
    index = pinecone.Index(index_name)
    res = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    matches = []
    for m in res.get("matches", []):
        matches.append({
            "id": m.get("id"),
            "score": float(m.get("score", 0.0)),
            "source": (m.get("metadata") or {}).get("source"),
            "verdict": (m.get("metadata") or {}).get("verdict"),
            "url": (m.get("metadata") or {}).get("url"),
        })
    return matches


def query_milvus(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    uri = os.environ.get("MILVUS_URI")  # e.g., http://localhost:19530
    collection = os.environ.get("MILVUS_COLLECTION", "factchecks")
    if not uri or not MilvusClient:
        return []
    client = MilvusClient(uri)
    results = client.search(
        collection_name=collection,
        data=[embedding],
        limit=top_k,
        output_fields=["source", "verdict", "url"],
    )
    matches = []
    for hit in results[0]:
        matches.append({
            "id": hit.get("id"),
            "score": float(hit.get("distance", 0.0)),
            "source": hit.get("entity", {}).get("source"),
            "verdict": hit.get("entity", {}).get("verdict"),
            "url": hit.get("entity", {}).get("url"),
        })
    return matches


def factcheck_matches(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    # Try Pinecone first, then Milvus
    matches = query_pinecone(embedding, top_k=top_k)
    if not matches:
        matches = query_milvus(embedding, top_k=top_k)
    # normalize similarity (Pinecone: higher is better; Milvus distance may be lower=better)
    normed: List[Dict[str, Any]] = []
    for m in matches:
        sim = float(m.get("score", 0.0))
        # Best effort: if distance range unknown, assume already similarity in [0,1]
        normed.append({
            "source": m.get("source"),
            "verdict": m.get("verdict"),
            "similarity": sim,
            "url": m.get("url"),
        })
    return [m for m in normed if m["similarity"] >= SIM_THRESHOLD]


def apply_factcheck_adjustment(label: str, confidence: float, matches: List[Dict[str, Any]]) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    if not matches:
        return label, confidence, None
    best = max(matches, key=lambda m: m["similarity"])
    verdict = (best.get("verdict") or "").lower()
    # Simple rule: if verdict says false, bias toward Fake; if true, bias toward Real.
    if "false" in verdict or "fake" in verdict:
        return "Fake", max(confidence, 0.9), best
    if "true" in verdict or "real" in verdict:
        return "Real", max(confidence, 0.9), best
    # Otherwise, keep label but include reference
    return label, confidence, best
