from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from llm_adapter.config import KNNIndexConfig, M3EpisodicMemoryConfig, get_global_config

logger = logging.getLogger('llm_adapter')


class M3EpisodicMemoryRetriever:
    """
    M3-aware episodic memory retrieval.

    - similarity: core.episodic_memory.entropy * engagement
    - top_k: episodic_memory.size / divisor (config.top_k_divisor)
    -  m3_param: (config.m3_param)
    """
    def __init__(self, config: Optional[M3EpisodicMemoryConfig] = None):
        self.config = config or get_global_config().episodic_memory

    def _get_episode_list(self, core) -> List:
        if core is None or not hasattr(core, 'episodic_memory'):
            return []
        em = core.episodic_memory
        if hasattr(em, 'episodes'):
            eps = em.episodes
            if isinstance(eps, dict):
                return list(eps.values())
            return list(eps)
        if hasattr(em, 'memories'):
            mems = em.memories
            if isinstance(mems, dict):
                return list(mems.values())
            return list(mems)
        return []

    def _current_qualia_vector(self, core) -> Optional[np.ndarray]:
        try:
            if core is None or not hasattr(core, 'qualia'):
                return None
            q = core.qualia
            return np.array(
                [
                    float(getattr(q, 'arousal', 0.0)),
                    float(getattr(q, 'valence', 0.0)),
                    float(getattr(q, 'entropy', 0.0)),
                    float(getattr(q, 'engagement', 0.0)),
                    float(getattr(q, 'frustration', 0.0)),
                ],
                dtype=np.float32,
            )
        except Exception:
            return None

    def _episode_embedding(self, episode) -> Optional[np.ndarray]:
        if episode is None:
            return None
        try:
            if hasattr(episode, 'embedding'):
                return np.asarray(episode.embedding, dtype=np.float32).ravel()
            if hasattr(episode, 'qualia_state'):
                return np.array(
                    [
                        getattr(episode.qualia_state, 'arousal', 0.0),
                        getattr(episode.qualia_state, 'valence', 0.0),
                        getattr(episode.qualia_state, 'entropy', 0.0),
                        getattr(episode.qualia_state, 'engagement', 0.0),
                        getattr(episode.qualia_state, 'frustration', 0.0),
                    ],
                    dtype=np.float32,
                )
            if hasattr(episode, 'qualia_vector'):
                return np.asarray(episode.qualia_vector, dtype=np.float32).ravel()
        except Exception:
            return None
        return None

    def _select_query_vector(self, current_embedding: np.ndarray, core, ep_emb: np.ndarray) -> Optional[np.ndarray]:
        if ep_emb is None:
            return None
        try:
            if current_embedding is not None:
                q = np.asarray(current_embedding, dtype=np.float32).ravel()
                if q.size == ep_emb.size:
                    return q
        except Exception:
            pass
        q_qualia = self._current_qualia_vector(core)
        if q_qualia is not None and q_qualia.size == ep_emb.size:
            return q_qualia
        return None

    def _compute_similarity_scores(self, episodes, current_embedding: np.ndarray, core=None):
        """
        Compute similarity scores between the current embedding and episodic memory episodes.

        Args:
            episodes: core.episodic_memory.episodes
            current_embedding: (D,) current context embedding from FeatureBank
        Returns:
            List[(episode, similarity_score)]
        """
        scored_episodes = []

        for episode in episodes:
            try:
                episode_emb = self._episode_embedding(episode)
                if episode_emb is None:
                    continue

                query_vec = self._select_query_vector(current_embedding, core, episode_emb)
                if query_vec is None:
                    continue

                # Cosine similarity
                current_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
                episode_norm = episode_emb / (np.linalg.norm(episode_emb) + 1e-8)
                similarity = float(np.dot(current_norm, episode_norm))

                scored_episodes.append((episode, similarity))
            except Exception as e:
                logger.debug(f"Exception occurred: {e}")
                continue

        return scored_episodes

    def _infer_top_k(self, core) -> int:
        """
        Infer top_k for episodic memory retrieval.

        : memory_size / divisor (config.top_k_divisor)
        """
        try:
            episodes = self._get_episode_list(core)
            mem_size = len(episodes)
            top_k = max(
                self.config.top_k_min,
                min(self.config.top_k_max, mem_size // self.config.memory_size_divisor)
            )
            return top_k
        except Exception as e:
            logger.debug(f"Exception occurred: {e}")
        return self.config.top_k_default  # Fallback: reasonable default

    def retrieve_relevant_episodes(self, core, current_context_embedding: np.ndarray):
        """
        Retrieve relevant episodes using M3-based multi-factor scoring.
        
        Score = (w_c * ContentSim) + (w_a * AffectSim) + (w_d * DriveRelevance)
        Weights are dynamic based on current state intensity.
        """
        if core is None or not hasattr(core, 'episodic_memory'):
            return []

        try:
            # 1. Get current internal states
            affect_state = core.affect_kernel.get_state() if hasattr(core, 'affect_kernel') else {}
            drive_state = core.drives.get_drive_state() if hasattr(core, 'drives') else {}
            
            # Calculate intensities for dynamic weighting
            # Affect intensity: L2 norm of affect values
            affect_vals = np.array(list(affect_state.values())) if affect_state else np.array([])
            w_affect = np.linalg.norm(affect_vals) if affect_vals.size > 0 else 0.0
            
            # Drive intensity: Max drive (urgency)
            drive_vals = np.array(list(drive_state.values())) if drive_state else np.array([])
            w_drive = np.max(drive_vals) if drive_vals.size > 0 else 0.0
            
            # Content weight is baseline (1.0)
            w_content = 1.0
            
            # Normalize weights
            w_sum = w_content + w_affect + w_drive + 1e-8
            w_c, w_a, w_d = w_content / w_sum, w_affect / w_sum, w_drive / w_sum

            episodes = self._get_episode_list(core)
            if not episodes:
                return []

            scored_episodes = []

            for episode in episodes:
                try:
                    # A. Content Similarity
                    content_sim = 0.0
                    ep_emb = self._episode_embedding(episode)
                    if ep_emb is not None:
                        query_vec = self._select_query_vector(current_context_embedding, core, ep_emb)
                        if query_vec is not None:
                            ep_norm = ep_emb / (np.linalg.norm(ep_emb) + 1e-8)
                            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
                            content_sim = float(np.dot(q_norm, ep_norm))
                    
                    # B. Affect Similarity (Mood Congruency)
                    affect_sim = 0.0
                    if w_a > 0 and hasattr(episode, 'affect_state') and episode.affect_state:
                        ep_a_vals = np.array([episode.affect_state.get(k, 0.0) for k in affect_state])
                        if np.linalg.norm(ep_a_vals) > 0:
                            # Cosine similarity of affect
                            curr_a_norm = affect_vals / (np.linalg.norm(affect_vals) + 1e-8)
                            ep_a_norm = ep_a_vals / (np.linalg.norm(ep_a_vals) + 1e-8)
                            affect_sim = float(np.dot(curr_a_norm, ep_a_norm))

                    # C. Drive Relevance (Did this episode help current drives?)
                    drive_rel = 0.0
                    if w_d > 0 and hasattr(episode, 'drive_reduction') and episode.drive_reduction:
                        # Dot product of current drive urgency and past reduction
                        for d_name, d_val in drive_state.items():
                            reduction = episode.drive_reduction.get(d_name, 0.0)
                            if reduction > 0:
                                drive_rel += d_val * reduction
                        drive_rel = min(1.0, drive_rel)

                    # Weighted Sum
                    total_score = (w_c * content_sim) + (w_a * affect_sim) + (w_d * drive_rel)
                    scored_episodes.append((episode, total_score))

                except Exception:
                    continue

            if not scored_episodes:
                return []

            # 2. Dynamic filtering based on Entropy (Cognitive Load)
            entropy = getattr(core.qualia, 'entropy', 0.5)
            engagement = getattr(core.qualia, 'engagement', 0.5)
            
            min_score = (1.0 - entropy) * engagement * 0.8
            try:
                min_floor = float(os.getenv('M3_EPISODIC_MIN_SCORE', '-0.25'))
                min_score = max(min_score, min_floor)
            except Exception:
                pass

            filtered = [
                (ep, score) for ep, score in scored_episodes
                if score > min_score
            ]

            # 3. Sort and Top-K
            filtered.sort(key=lambda x: x[1], reverse=True)
            top_k = self._infer_top_k(core)
            if not filtered:
                scored_episodes.sort(key=lambda x: x[1], reverse=True)
                filtered = scored_episodes

            selected = [ep for ep, _ in filtered[:top_k]]
            if selected:
                for ep in selected:
                    try:
                        if hasattr(ep, 'retrieval_count'):
                            ep.retrieval_count = int(getattr(ep, 'retrieval_count', 0)) + 1
                    except Exception:
                        continue
                try:
                    em = getattr(core, 'episodic_memory', None)
                    if em is not None and hasattr(em, 'total_retrieved'):
                        em.total_retrieved = int(getattr(em, 'total_retrieved', 0)) + len(selected)
                except Exception:
                    pass
            return selected

        except Exception as e:
            logger.error(f"Error in retrieve_relevant_episodes: {e}")
            return []


# === kNN-LM: Conditional Index ===
@dataclass
class KNNItem:
    key: np.ndarray        # (Kdim,)
    value: np.ndarray      # (Vocab,)  softmaxed next-token distribution


class ConditionalKNNIndex:
    """Simple cosine + temperature scaled kNN index with conditional keys.

    Features:
    - LRU eviction when max_items exceeded
    - Periodic downsampling for memory control
    - Logging of KDIM, TAU, and other parameters
    - All parameters configurable via KNNIndexConfig
    """
    def __init__(self, config: Optional[KNNIndexConfig] = None):
        cfg = config or get_global_config().knn_index
        self.tau = float(cfg.tau)
        self.max_items = int(cfg.max_items)
        self.key_dim = int(cfg.key_dim)
        self._keys = []     # List[np.ndarray]
        self._vals = []     # List[np.ndarray]
        self._access_counts = []  # LRU tracking
        self._total_queries = 0
        self._last_downsample_size = 0

        # Log configuration
        logger = logging.getLogger('llm_adapter.knn')
        logger.info(f'ConditionalKNNIndex initialized: KDIM={self.key_dim}, tau={self.tau}, max_items={self.max_items}')

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        n = np.linalg.norm(x) + 1e-8
        return x / n

    def _softmax(self, x: np.ndarray, T: float = 1.0) -> np.ndarray:
        z = (x / (T + 1e-8)) - np.max(x)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-8)

    def add(self, key: np.ndarray, value_logits: np.ndarray):
        """key: (Kdim,), value_logits: (V,) raw logits for next token at teacher-forcing."""
        if key.ndim != 1:
            key = key.reshape(-1)
        k = self._norm(key)
        p = self._softmax(value_logits)  # store as prob dist
        self._keys.append(k)
        self._vals.append(p.astype(np.float32))
        self._access_counts.append(0)

        # Cap enforcement with LRU eviction
        if len(self._keys) > self.max_items:
            evict_idx = int(np.argmin(self._access_counts))
            del self._keys[evict_idx]
            del self._vals[evict_idx]
            del self._access_counts[evict_idx]

        # Periodic downsampling: every 10k items, downsample by 50%
        downsample_interval = int(os.getenv('KNN_DOWNSAMPLE_INTERVAL', '10000'))
        if len(self._keys) > self._last_downsample_size + downsample_interval:
            self._downsample()

    def _downsample(self):
        """Hybrid downsample: keep ~75% (hot + random) to preserve diversity."""
        if len(self._keys) < 100:
            return

        n = len(self._keys)

        # Keep fraction (default 0.75); override via env KNN_DOWNSAMPLE_KEEP_FRACTION
        try:
            keep_frac = float(os.getenv('KNN_DOWNSAMPLE_KEEP_FRACTION', '0.75'))
        except Exception:
            keep_frac = 0.75
        keep_frac = min(max(keep_frac, 0.0), 1.0)
        keep_n = max(1, int(n * keep_frac))

        # Split kept set into hot(top by access) and random remainder
        access = np.asarray(self._access_counts, dtype=np.int64)
        order = np.argsort(access)
        top_n = max(1, int(keep_n * 0.5))
        top_idx = order[-top_n:]

        # Candidates excluding top
        mask = np.ones(n, dtype=bool)
        mask[top_idx] = False
        rest_idx = np.nonzero(mask)[0]
        rnd_n = max(0, keep_n - top_n)
        if rnd_n > 0 and rest_idx.size > 0:
            choose_n = min(rnd_n, rest_idx.size)
            rnd_pick = np.random.choice(rest_idx, size=choose_n, replace=False)
            sel_idx = np.concatenate([top_idx, rnd_pick])
        else:
            sel_idx = top_idx

        sel_idx = sel_idx.tolist()
        self._keys = [self._keys[i] for i in sel_idx]
        self._vals = [self._vals[i] for i in sel_idx]
        self._access_counts = [self._access_counts[i] for i in sel_idx]

        # Move the trigger baseline forward by interval to reduce oscillation
        try:
            downsample_interval = int(os.getenv('KNN_DOWNSAMPLE_INTERVAL', '10000'))
        except Exception:
            downsample_interval = 10000
        self._last_downsample_size += max(1, downsample_interval)

        logger = logging.getLogger('llm_adapter.knn')
        kept = len(self._keys)
        logger.info(f'kNN downsampled: {n} -> {kept} items (top={min(top_n, kept)}, rand={max(0, kept - min(top_n, kept))})')

    def query(self, qkey: np.ndarray, k: int = 8) -> np.ndarray | None:
        if not self._keys:
            return None

        self._total_queries += 1
        q = self._norm(qkey.reshape(-1))

        # cosine similarities
        sims = np.array([float(np.dot(q, kk)) for kk in self._keys], dtype=np.float32)
        idx = np.argsort(-sims)[:max(1, int(k))]
        top = sims[idx]

        # Update access counts for LRU
        for i in idx:
            self._access_counts[i] += 1

        w = self._softmax(top / (self.tau + 1e-8))
        P = np.stack([self._vals[i] for i in idx], axis=0)  # (k, V)

        # Log usage stats every 1000 queries
        if self._total_queries % 1000 == 0:
            logger = logging.getLogger('llm_adapter.knn')
            logger.debug(f'kNN stats: queries={self._total_queries}, items={len(self._keys)}, tau={self.tau:.4f}')

        return (w[:, None] * P).sum(axis=0)  # (V,)
