from __future__ import annotations

import logging
import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_adapter.config import (
    EpisodicANNConfig,
    KNNIndexConfig,
    M3EpisodicMemoryConfig,
    get_global_config,
)

logger = logging.getLogger('llm_adapter')


class _ANNBackendBase:
    """Minimal ANN backend interface."""

    name: str = "numpy"

    def fit(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


class _NumpyANNBackend(_ANNBackendBase):
    name = "numpy"

    def __init__(self):
        self._vectors = np.empty((0, 1), dtype=np.float32)

    def fit(self, vectors: np.ndarray) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors.ndim != 2:
            self._vectors = self._vectors.reshape(self._vectors.shape[0], -1)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8
        self._vectors = self._vectors / norms

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        if self._vectors.size == 0:
            return np.empty((0,), dtype=np.int64)
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self._vectors @ q
        kk = max(1, min(int(k), sims.shape[0]))
        idx = np.argpartition(-sims, kk - 1)[:kk]
        idx = idx[np.argsort(-sims[idx])]
        return idx.astype(np.int64)


class _AnnoyANNBackend(_ANNBackendBase):
    name = "annoy"

    def __init__(self, dim: int, trees: int = 10):
        from annoy import AnnoyIndex  # type: ignore

        self._dim = int(dim)
        self._trees = int(max(1, trees))
        self._AnnoyIndex = AnnoyIndex
        self._index = None
        self._n = 0

    def fit(self, vectors: np.ndarray) -> None:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim != 2:
            vec = vec.reshape(vec.shape[0], -1)
        self._n = int(vec.shape[0])
        idx = self._AnnoyIndex(self._dim, metric='angular')
        for i in range(self._n):
            idx.add_item(i, vec[i].tolist())
        idx.build(self._trees)
        self._index = idx

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        if self._index is None or self._n <= 0:
            return np.empty((0,), dtype=np.int64)
        kk = max(1, min(int(k), self._n))
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        ids = self._index.get_nns_by_vector(q.tolist(), kk, include_distances=False)
        return np.asarray(ids, dtype=np.int64)


class _FaissANNBackend(_ANNBackendBase):
    name = "faiss"

    def __init__(self, dim: int):
        import faiss  # type: ignore

        self._dim = int(dim)
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._n = 0

    def fit(self, vectors: np.ndarray) -> None:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim != 2:
            vec = vec.reshape(vec.shape[0], -1)
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        vec = vec / norms
        self._index.reset()
        self._index.add(vec)
        self._n = int(vec.shape[0])

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        if self._n <= 0:
            return np.empty((0,), dtype=np.int64)
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        kk = max(1, min(int(k), self._n))
        _, idx = self._index.search(q, kk)
        return idx[0].astype(np.int64)


class M3EpisodicMemoryRetriever:
    """
    M3-aware episodic memory retrieval.

    - similarity: core.episodic_memory.entropy * engagement
    - top_k: episodic_memory.size / divisor (config.top_k_divisor)
    -  m3_param: (config.m3_param)
    """
    _MODULE_SUPPORT_CACHE: Dict[str, bool] = {}

    def __init__(self, config: Optional[M3EpisodicMemoryConfig] = None):
        self.config = config or get_global_config().episodic_memory
        self.ann_config: EpisodicANNConfig = get_global_config().episodic_ann
        self._ann_backend: _ANNBackendBase = _NumpyANNBackend()
        self._ann_backend_name: str = "numpy"
        self._ann_vectors: np.ndarray = np.empty((0, 1), dtype=np.float32)
        self._ann_episodes: List[Any] = []
        self._ann_dim: int = 0
        self._ann_refresh_count: int = 0
        self._ann_last_episode_count: int = -1
        self._ann_last_refresh_ts: float = 0.0
        self._ann_last_query_sig: Optional[Tuple[str, int, int]] = None
        self._ann_last_query_log_ts: float = 0.0
        self._ann_selected_once: bool = False
        self._ann_failed_until: Dict[str, float] = {}
        self._ann_probe_backoff_sec: float = float(max(1.0, float(os.getenv("M3_EPISODIC_ANN_PROBE_BACKOFF_SEC", "600"))))
        self._ann_query_log_debounce_sec: float = float(max(0.0, float(os.getenv("M3_EPISODIC_ANN_QUERY_LOG_DEBOUNCE_SEC", "1.0"))))
        backend_name = os.getenv("M3_EPISODIC_ANN_BACKEND", self.ann_config.backend)
        self.set_ann_backend(backend_name)

    def _log_ann_event(self, event: str, **kwargs) -> None:
        payload = {"kind": "ann_backend", "event": str(event), "t": int(time.time() * 1000)}
        payload.update(kwargs)
        try:
            path = os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _resolve_backend_name(self, name: Optional[str]) -> str:
        raw = str(name or "auto").strip().lower()
        if raw in {"np", "numpy", "bruteforce", "brute"}:
            return "numpy"
        if raw == "faiss":
            return "faiss"
        if raw == "annoy":
            return "annoy"
        return "auto"

    def _supports_module(self, module_name: str) -> bool:
        cache_enabled = bool(getattr(self.ann_config, "cache_backend_probe", True))
        if cache_enabled and module_name in self._MODULE_SUPPORT_CACHE:
            return bool(self._MODULE_SUPPORT_CACHE[module_name])
        try:
            __import__(module_name)
            if cache_enabled:
                self._MODULE_SUPPORT_CACHE[module_name] = True
            return True
        except Exception:
            if cache_enabled:
                self._MODULE_SUPPORT_CACHE[module_name] = False
            return False

    def _is_backend_blocked(self, name: str) -> bool:
        try:
            return float(time.time()) < float(self._ann_failed_until.get(str(name), 0.0))
        except Exception:
            return False

    def _mark_backend_failed(self, name: str, reason: str = "") -> None:
        key = str(name or "").strip().lower()
        if not key:
            return
        self._ann_failed_until[key] = float(time.time()) + float(self._ann_probe_backoff_sec)
        if bool(getattr(self.ann_config, "cache_backend_probe", True)):
            self._MODULE_SUPPORT_CACHE[key] = False
        self._log_ann_event("probe_fail", backend=key, reason=str(reason))

    def _resolve_auto_backend(self) -> str:
        if not self._is_backend_blocked("faiss") and self._supports_module("faiss"):
            return "faiss"
        if not self._is_backend_blocked("annoy") and self._supports_module("annoy"):
            return "annoy"
        return "numpy"

    def set_ann_backend(self, name: str):
        resolved = self._resolve_backend_name(name)
        if resolved == "auto":
            resolved = self._resolve_auto_backend()
        log_select_once = bool(getattr(self.ann_config, "log_select_once", True))
        if (
            resolved == self._ann_backend_name
            and self._ann_selected_once
            and log_select_once
        ):
            if str(os.getenv("M3_EPISODIC_ANN_LOG_DEDUP", "0")).lower() in {"1", "true", "yes", "on"}:
                self._log_ann_event(
                    "ann_backend_dedup",
                    backend=str(self._ann_backend_name),
                    requested=str(name),
                )
            return
        self._ann_backend_name = resolved
        # Rebuild backend instance on next refresh to ensure right dimension.
        self._ann_backend = _NumpyANNBackend()
        self._ann_refresh_count = 0
        self._ann_last_episode_count = -1
        logger.info(
            "ann_backend_selected name=%s env=%s",
            self._ann_backend_name,
            os.getenv("M3_EPISODIC_ANN_BACKEND", "auto"),
        )
        if (not log_select_once) or (not self._ann_selected_once):
            self._log_ann_event(
                "select",
                backend=str(self._ann_backend_name),
                requested=str(name),
                env=str(os.getenv("M3_EPISODIC_ANN_BACKEND", "auto")),
            )
        self._ann_selected_once = True

    def _create_backend_for_dim(self, dim: int) -> _ANNBackendBase:
        target = self._ann_backend_name
        if target == "faiss":
            if self._is_backend_blocked("faiss") or not self._supports_module("faiss"):
                target = "annoy"
            else:
                try:
                    return _FaissANNBackend(dim=dim)
                except Exception as e:
                    self._mark_backend_failed("faiss", str(e))
                    logger.warning("FAISS unavailable at runtime; falling back to Annoy/NumPy")
                    target = "annoy"
        if target == "annoy":
            if self._is_backend_blocked("annoy") or not self._supports_module("annoy"):
                target = "numpy"
            else:
                try:
                    trees = int(max(1, getattr(self.ann_config, "annoy_trees", 10)))
                    return _AnnoyANNBackend(dim=dim, trees=trees)
                except Exception as e:
                    self._mark_backend_failed("annoy", str(e))
                    logger.warning("Annoy unavailable at runtime; falling back to NumPy")
                    target = "numpy"
        if target != self._ann_backend_name:
            self._ann_backend_name = str(target)
            self._log_ann_event("fallback_select", backend=str(target))
        return _NumpyANNBackend()

    def _maybe_log_query_event(self, total: int, candidates: int) -> None:
        backend = str(getattr(self._ann_backend, "name", self._ann_backend_name))
        sig = (backend, int(total), int(candidates))
        now = float(time.time())
        if (
            self._ann_last_query_sig == sig
            and (now - float(self._ann_last_query_log_ts)) < float(self._ann_query_log_debounce_sec)
        ):
            return
        self._ann_last_query_sig = sig
        self._ann_last_query_log_ts = now
        self._log_ann_event(
            "query",
            backend=backend,
            total=int(total),
            candidates=int(candidates),
        )

    @staticmethod
    def _dominant_dim(vectors: List[np.ndarray]) -> Optional[int]:
        counts: Dict[int, int] = {}
        for v in vectors:
            if v is None:
                continue
            d = int(v.size)
            counts[d] = counts.get(d, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def refresh_index(self, core):
        episodes = self._get_episode_list(core)
        embs: List[np.ndarray] = []
        eps: List[Any] = []
        for ep in episodes:
            emb = self._episode_embedding(ep)
            if emb is None:
                continue
            embs.append(np.asarray(emb, dtype=np.float32).ravel())
            eps.append(ep)
        dim = self._dominant_dim(embs)
        if dim is None:
            self._ann_vectors = np.empty((0, 1), dtype=np.float32)
            self._ann_episodes = []
            self._ann_dim = 0
            self._ann_refresh_count += 1
            self._ann_last_episode_count = len(episodes)
            return
        filtered = []
        filtered_eps = []
        for ep, emb in zip(eps, embs):
            if emb.size == dim:
                filtered.append(emb)
                filtered_eps.append(ep)
        if not filtered:
            self._ann_vectors = np.empty((0, dim), dtype=np.float32)
            self._ann_episodes = []
            self._ann_dim = int(dim)
            self._ann_refresh_count += 1
            self._ann_last_episode_count = len(episodes)
            return
        vec = np.stack(filtered, axis=0).astype(np.float32)
        try:
            self._ann_backend = self._create_backend_for_dim(int(dim))
            self._ann_backend.fit(vec)
            self._ann_vectors = vec
            self._ann_episodes = filtered_eps
            self._ann_dim = int(dim)
            self._ann_refresh_count += 1
            self._ann_last_episode_count = len(episodes)
            self._ann_last_refresh_ts = float(time.time())
            logger.debug(
                "ann_backend_refresh backend=%s dim=%d items=%d refresh=%d",
                getattr(self._ann_backend, "name", self._ann_backend_name),
                self._ann_dim,
                len(self._ann_episodes),
                self._ann_refresh_count,
            )
            self._log_ann_event(
                "refresh",
                backend=str(getattr(self._ann_backend, "name", self._ann_backend_name)),
                dim=int(self._ann_dim),
                items=int(len(self._ann_episodes)),
                refresh=int(self._ann_refresh_count),
            )
        except Exception as e:
            logger.warning("ANN refresh failed (%s); using NumPy fallback", e)
            self._ann_backend = _NumpyANNBackend()
            self._ann_backend.fit(vec)
            self._ann_vectors = vec
            self._ann_episodes = filtered_eps
            self._ann_dim = int(dim)
            self._ann_refresh_count += 1
            self._ann_last_episode_count = len(episodes)
            self._log_ann_event(
                "fallback_numpy",
                reason=str(e),
                dim=int(self._ann_dim),
                items=int(len(self._ann_episodes)),
            )

    def _needs_refresh(self, core, episodes: List[Any]) -> bool:
        n = len(episodes)
        if n != self._ann_last_episode_count:
            return True
        if self._ann_dim <= 0 or len(self._ann_episodes) == 0:
            return True
        interval = int(max(1, getattr(self.ann_config, "rebuild_interval", 256)))
        if self._ann_refresh_count <= 0:
            return True
        return (self._ann_refresh_count % interval) == 0

    def query_candidates(self, vec: np.ndarray, k: int) -> List[Any]:
        if self._ann_dim <= 0 or len(self._ann_episodes) <= 0:
            return []
        q = np.asarray(vec, dtype=np.float32).ravel()
        if q.size != self._ann_dim:
            return []
        try:
            idx = self._ann_backend.query(q, k=max(1, int(k)))
        except Exception as e:
            # Runtime backend mismatch: rebuild with NumPy fallback and retry.
            try:
                old_backend = str(getattr(self._ann_backend, "name", self._ann_backend_name))
                if old_backend not in {"", "numpy"}:
                    self._mark_backend_failed(old_backend, str(e))
                self._ann_backend = _NumpyANNBackend()
                self._ann_backend_name = "numpy"
                self._ann_backend.fit(self._ann_vectors)
                idx = self._ann_backend.query(q, k=max(1, int(k)))
                self._log_ann_event(
                    "runtime_fallback",
                    from_backend=str(old_backend),
                    to_backend="numpy",
                    reason=str(e),
                )
            except Exception:
                return []
        out: List[Any] = []
        for i in idx.tolist():
            if 0 <= int(i) < len(self._ann_episodes):
                out.append(self._ann_episodes[int(i)])
        return out

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
            candidates = episodes
            min_items_for_ann = int(max(1, getattr(self.ann_config, "min_items_for_ann", 128)))
            if len(episodes) >= min_items_for_ann:
                try:
                    if self._needs_refresh(core, episodes):
                        self.refresh_index(core)
                except Exception:
                    pass
                candidate_k = int(max(1, getattr(self.ann_config, "candidate_k", 64)))
                q_for_ann = None
                if self._ann_dim > 0:
                    try:
                        q = np.asarray(current_context_embedding, dtype=np.float32).ravel()
                        if q.size == self._ann_dim:
                            q_for_ann = q
                    except Exception:
                        q_for_ann = None
                    if q_for_ann is None:
                        try:
                            q_qualia = self._current_qualia_vector(core)
                            if q_qualia is not None and q_qualia.size == self._ann_dim:
                                q_for_ann = q_qualia
                        except Exception:
                            q_for_ann = None
                if q_for_ann is not None:
                    ann_candidates = self.query_candidates(q_for_ann, k=candidate_k)
                    if ann_candidates:
                        candidates = ann_candidates
                logger.debug(
                    "ann_backend_query backend=%s total=%d candidates=%d",
                    getattr(self._ann_backend, "name", self._ann_backend_name),
                    len(episodes),
                    len(candidates),
                )
                self._maybe_log_query_event(total=len(episodes), candidates=len(candidates))

            scored_episodes = []

            for episode in candidates:
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
