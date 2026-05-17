#!/usr/bin/env python3
"""
embedding_index.py
==================
Hardened local index for curriculum embeddings. 
Provides zero-vector-safe nearest-neighbor lookup for competence aims and unit plans.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

class EmbeddingIndex:
    def __init__(self, index_path: Optional[str | Path] = None):
        self.index_path = Path(index_path) if index_path else None
        self.embeddings = []
        self.metadata = []
        
    def add(self, vector: List[float] | np.ndarray, metadata: Dict[str, Any]):
        """Adds a vector and its associated metadata to the index."""
        self.embeddings.append(np.asarray(vector, dtype=np.float32).tolist())
        self.metadata.append(dict(metadata))
        
    def search(self, query_vector: List[float] | np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Performs a zero-vector-safe cosine similarity search."""
        if not self.embeddings:
            return []
            
        idx_matrix = np.asarray(self.embeddings, dtype=np.float32)
        q_vec = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        
        # Numeric stability guards
        def _safe_norm(v, axis=None, keepdims=False):
            n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
            return np.where(n > 1e-12, n, 1.0)

        idx_norm = idx_matrix / _safe_norm(idx_matrix, axis=1, keepdims=True)
        q_norm = q_vec / _safe_norm(q_vec)
        
        # Compute scores and clip to [-1, 1]
        scores = np.clip(np.dot(idx_norm, q_norm), -1.0, 1.0)
        
        # Get top K indices
        n_results = min(int(k), len(scores))
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        return [(float(scores[i]), self.metadata[i]) for i in top_indices]

    def save(self, path: Optional[str | Path] = None):
        """Saves the index to disk."""
        target = Path(path) if path else self.index_path
        if not target:
            raise ValueError("No save path specified.")
        data = {
            "format": "LK20.EmbeddingIndex",
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Optional[str | Path] = None):
        """Loads the index from disk."""
        target = Path(path) if path else self.index_path
        if not target or not target.exists():
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            self.embeddings = data.get("embeddings", [])
            self.metadata = data.get("metadata", [])
        except Exception:
            pass

if __name__ == "__main__":
    # Basic test
    index = EmbeddingIndex()
    index.add([1.0, 0.0, 0.0], {"id": "aim_1"})
    index.add([0.0, 0.0, 0.0], {"id": "zero_vec"})
    results = index.search([1.0, 0.0, 0.0])
    print(f"Search Results: {results}")
