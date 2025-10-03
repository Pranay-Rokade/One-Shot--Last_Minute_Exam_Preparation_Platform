# summaria_utils.py
import os
import json
from collections import defaultdict
from itertools import combinations
from sentence_transformers import SentenceTransformer
import numpy as np
import datetime
from pinecone_text.sparse import BM25Encoder

# Load semantic model
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Global BM25 encoder
_bm25 = BM25Encoder().default()
FEEDBACK_FILE = "summaria_feedback.json"


def _sparse_dot(query_vec, doc_vec):
    """
    Compute dot product between sparse dicts (indices/values).
    """
    q_idx, q_val = query_vec["indices"], query_vec["values"]
    d_idx, d_val = doc_vec["indices"], doc_vec["values"]
    score = 0.0
    d_map = dict(zip(d_idx, d_val))
    for i, v in zip(q_idx, q_val):
        if i in d_map:
            score += v * d_map[i]
    return score


def _hybrid_score_dense_sparse(query, docs, alpha=0.6, beta=0.4):
    """
    Compute hybrid score using semantic (MiniLM) + BM25 sparse matching.
    """
    # Dense embeddings
    q_vec = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    d_vecs = _embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    dense_scores = np.dot(d_vecs, q_vec)

    # Sparse (BM25)
    sparse_docs = _bm25.encode_documents(docs)
    q_sparse = _bm25.encode_queries([query])[0]
    bm25_scores = np.array([_sparse_dot(q_sparse, d) for d in sparse_docs])

    # Normalize
    if np.max(dense_scores) > 0:
        dense_scores = dense_scores / np.max(dense_scores)
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)

    return alpha * dense_scores + beta * bm25_scores

def compute_topic_metrics(topics, chunks, alpha=0.6, beta=0.4, threshold=0.3):
    """
    Compute SUMMARIA-style metrics using hybrid (semantic + BM25).
    """
    if not topics:
        return {}, {}

    chunk_texts = [doc.page_content[:1000] for doc in chunks]
    if not chunk_texts:
        return {}, {}

    # Fit BM25 on chunks if not yet fitted
    _bm25.fit(chunk_texts)

    topic_info = {}
    topic_chunk_idxs = defaultdict(list)

    for t in topics:
        sims = _hybrid_score_dense_sparse(t, chunk_texts, alpha, beta)
        for ci, score in enumerate(sims):
            if score >= threshold:
                topic_chunk_idxs[t].append(ci)

    max_chunks = max((len(v) for v in topic_chunk_idxs.values()), default=1)
    n_chunks = len(chunk_texts)

    for t in topics:
        covered = len(topic_chunk_idxs[t])
        topic_info[t] = {
            "truth_degree": round(covered / max_chunks, 4) if max_chunks else 0.0,
            "coverage_degree": round(covered / n_chunks, 4) if n_chunks else 0.0,
            "count": covered,
            "chunks_with": topic_chunk_idxs[t],
        }

    # Co-occurrence
    cooccurrence = {}
    for a, b in combinations(topics, 2):
        set_a, set_b = set(topic_chunk_idxs[a]), set(topic_chunk_idxs[b])
        if not set_a or not set_b:
            rel = 0.0
        else:
            rel = len(set_a & set_b) / max(len(set_a), len(set_b))
        cooccurrence[(a, b)] = round(rel, 4)
        cooccurrence[(b, a)] = round(rel, 4)

    return topic_info, cooccurrence


def build_composite_relations(topic_info, cooccurrence, top_n=5):
    """Same as before — Evidence / Emphasis / Contrast"""
    topics = list(topic_info.keys())
    evidence, emphasis, contrast = [], [], []

    sorted_by_truth = sorted(topics, key=lambda t: topic_info[t]["truth_degree"], reverse=True)
    consider = sorted_by_truth[: min(top_n, len(sorted_by_truth))]

    for a, b in combinations(consider, 2):
        rel = cooccurrence.get((a, b), 0.0)
        truth_a = topic_info[a]["truth_degree"]
        truth_b = topic_info[b]["truth_degree"]
        cov_a = topic_info[a]["coverage_degree"]
        cov_b = topic_info[b]["coverage_degree"]

        if rel >= 0.6:  # Evidence
            nucleus, satellite = (a, b) if truth_a >= truth_b else (b, a)
            evidence.append({
                "nucleus": nucleus,
                "satellite": satellite,
                "relation_strength": rel,
                "verbalization": f"'{satellite}' often appears with '{nucleus}', reinforcing its importance."
            })
            continue

        if rel >= 0.7 and ((cov_a < cov_b and truth_a > truth_b) or (cov_b < cov_a and truth_b > truth_a)):
            nucleus = a if cov_a >= cov_b else b
            satellite = b if nucleus == a else a
            emphasis.append({
                "nucleus": nucleus,
                "satellite": satellite,
                "relation_strength": rel,
                "verbalization": f"Within '{nucleus}', '{satellite}' is especially emphasized."
            })
            continue

        if rel < 0.3 and abs(truth_a - truth_b) >= 0.35:
            contrast.append({
                "topic1": a,
                "topic2": b,
                "truth_diff": round(abs(truth_a - truth_b), 4),
                "relation_strength": rel,
                "verbalization": f"'{a}' and '{b}' contrast in frequency, showing different exam emphasis."
            })

    return {
        "evidence": sorted(evidence, key=lambda x: x["relation_strength"], reverse=True),
        "emphasis": sorted(emphasis, key=lambda x: x["relation_strength"], reverse=True),
        "contrast": sorted(contrast, key=lambda x: x["truth_diff"], reverse=True),
    }


def persist_feedback(feedback):
    """Save feedback to a JSON file"""
    entry = dict(feedback)
    entry.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
    data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True
