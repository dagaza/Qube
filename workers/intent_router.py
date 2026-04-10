import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger("Qube.IntentRouter")

@dataclass
class Intent:
    name: str
    centroid: np.ndarray
    threshold: float
    margin: float

class IntentRegistry:
    def __init__(self):
        self.intents = {}

    def register(self, intent: Intent):
        self.intents[intent.name] = intent

    def all(self):
        return list(self.intents.values())

class EmbeddingCache:
    """
    Ensures a single embedding pass per request lifecycle.
    Prevents the CPU/GPU from wasting cycles re-embedding the same prompt.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache = None
        self._last_text = None

    def get_embedding(self, text: str) -> np.ndarray:
        if self._cache is None or self._last_text != text:
            logger.debug("Generating new embedding vector for intent routing...")
            # Using your existing embed_query method
            self._cache = self.embedder.embed_query(text)
            self._last_text = text
        return self._cache

    def reset(self):
        self._cache = None
        self._last_text = None

class IntentRouter:
    def __init__(self, registry: IntentRegistry):
        self.registry = registry

    def _score_intents(self, user_vec: np.ndarray):
        scores = []
        for intent in self.registry.all():
            # Nomic v1.5 outputs L2 normalized vectors. 
            # Therefore, Dot Product is mathematically identical to Cosine Similarity.
            score = float(np.dot(user_vec, intent.centroid))
            scores.append((intent.name, score, intent))
        
        # Sort highest score first
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def route(self, user_vec: np.ndarray):
        """Returns a tuple of (Intent Name, Confidence Score)"""
        ranked = self._score_intents(user_vec)
        
        top1_name, top1_score, top1_intent = ranked[0]
        top2_name, top2_score, _ = ranked[1] if len(ranked) > 1 else (None, 0.0, None)
        
        margin = top1_score - top2_score
        
        logger.debug(f"Router Analysis: Top 1 [{top1_name}: {top1_score:.3f}], Top 2 [{top2_name}: {top2_score:.3f}], Margin: {margin:.3f}")

        # Decision gating logic
        if top1_score >= top1_intent.threshold and margin >= top1_intent.margin:
            logger.info(f"Intent locked: {top1_name} (Confidence: {top1_score:.3f}, Margin: {margin:.3f})")
            return top1_name, top1_score
            
        logger.info(f"Intent ambiguous or below threshold. Falling back to CHAT. (Top score: {top1_score:.3f})")
        return "CHAT", top1_score

def build_centroid(embedder, examples: list[str]) -> np.ndarray:
    """Embeds a list of examples and computes their mathematical average (centroid)."""
    vectors = [embedder.embed_query(text) for text in examples]
    
    # Calculate the mean across all example vectors
    centroid = np.mean(vectors, axis=0)
    
    # CRITICAL MATH FIX: Re-normalize the centroid vector to a magnitude of 1.
    # Without this, the dot product will break and return values > 1.0 or < -1.0.
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
        
    return centroid