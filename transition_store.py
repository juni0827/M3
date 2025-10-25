import math
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _CESConfig:
    shard_prefix_bits: int = 10
    wal_segment_bytes: int = 512 << 20
    cms_depth: int = 4
    cms_width: int = 1 << 20
    alpha: float = 0.3
    topk_c: float = 6.0
    topk_min: int = 8
    topk_max: int = 64
    promote_margin_min: int = 3
    promote_margin_frac: float = 0.01
    cooldown_updates: int = 8192
    decay_half_life_updates: int = 10_000_000
    seed: int = 1337
    row_cache_capacity: int = 200_000
    cms_rehash_window_updates: int = 200_000_000


def _mix64(x: np.uint64) -> np.uint64:
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return x


def _cms_index(hseed: np.uint64, width: int, src: int, dst: int) -> int:
    v = np.uint64((src << 32) ^ (dst & 0xFFFFFFFF)) ^ hseed
    return int(_mix64(v) % np.uint64(width))


class _CmsSketch:
    __slots__ = ("depth", "width", "table", "hseeds", "epoch", "updates", "cfg")

    def __init__(self, cfg: _CESConfig):
        self.cfg = cfg
        self.depth = cfg.cms_depth
        self.width = cfg.cms_width
        rng = np.random.RandomState(cfg.seed)
        self.hseeds = [np.uint64(rng.randint(0, 2**31 - 1)) for _ in range(self.depth)]
        self.table = np.zeros((self.depth, self.width), dtype=np.uint32)
        self.epoch = 0
        self.updates = 0

    def estimate(self, src: int, dst: int) -> int:
        est = np.iinfo(np.uint32).max
        for depth_idx in range(self.depth):
            index = _cms_index(self.hseeds[depth_idx], self.width, src, dst)
            est = min(est, self.table[depth_idx, index])
        return 0 if est == np.iinfo(np.uint32).max else int(est)

    def add(self, src: int, dst: int, weight: int) -> None:
        current = self.estimate(src, dst)
        target = current + int(weight)
        for depth_idx in range(self.depth):
            index = _cms_index(self.hseeds[depth_idx], self.width, src, dst)
            if self.table[depth_idx, index] < target:
                self.table[depth_idx, index] = target
        self.updates += 1
        if (
            self.cfg.cms_rehash_window_updates
            and self.updates % self.cfg.cms_rehash_window_updates == 0
        ):
            self._rehash()

    def _rehash(self) -> None:
        self.scale_down_half()
        rng = np.random.RandomState(self.cfg.seed + self.epoch + 1)
        self.hseeds = [np.uint64(rng.randint(0, 2**31 - 1)) for _ in range(self.depth)]

    def scale_down_half(self) -> None:
        self.table >>= 1
        self.epoch += 1


class _TopKIndex:
    __slots__ = ("map", "heap")

    def __init__(self) -> None:
        self.map: Dict[int, Dict[int, int]] = {}
        self.heap: Dict[int, List[Tuple[int, int]]] = {}

    @staticmethod
    def _dyn_k(cfg: _CESConfig, rowsum: int) -> int:
        return max(
            cfg.topk_min,
            min(cfg.topk_max, int(math.ceil(cfg.topk_c * math.log1p(max(0, rowsum))))),
        )

    def has(self, row: int, dst: int) -> bool:
        row_map = self.map.get(row)
        return row_map is not None and dst in row_map

    def get(self, row: int, dst: int) -> int:
        row_map = self.map.get(row)
        if row_map is None:
            return 0
        return int(row_map.get(dst, 0))

    def list(self, row: int, k: int) -> List[Tuple[int, int]]:
        row_map = self.map.get(row)
        if not row_map:
            return []
        items = sorted(row_map.items(), key=lambda item: item[1], reverse=True)
        return [(dst, int(count)) for dst, count in items[:k]]

    def _rebuild_heap(self, row: int) -> None:
        row_map = self.map.get(row, {})
        self.heap[row] = [(count, dst) for dst, count in row_map.items()]
        heapq.heapify(self.heap[row])

    def inc(self, cfg: _CESConfig, row: int, dst: int, weight: int, rowsum: int) -> None:
        row_map = self.map.setdefault(row, {})
        row_map[dst] = int(row_map.get(dst, 0)) + int(weight)
        k = self._dyn_k(cfg, rowsum)
        if len(row_map) > k:
            top = sorted(row_map.items(), key=lambda item: item[1], reverse=True)[:k]
            self.map[row] = {dst_idx: count for dst_idx, count in top}
            row_map = self.map[row]
        self._rebuild_heap(row)

    def maybe_promote(
        self, cfg: _CESConfig, row: int, dst: int, est_value: int, rowsum: int
    ) -> bool:
        row_map = self.map.get(row)
        if row_map is None:
            self.map[row] = {dst: int(est_value)}
            self._rebuild_heap(row)
            return True
        if dst in row_map:
            return False
        k = self._dyn_k(cfg, rowsum)
        if len(row_map) < k:
            row_map[dst] = int(est_value)
            self._rebuild_heap(row)
            return True
        margin = max(cfg.promote_margin_min, int(cfg.promote_margin_frac * max(1, rowsum)))
        heap = self.heap[row]
        if not heap:
            row_map[dst] = int(est_value)
            self._rebuild_heap(row)
            return True
        min_count, min_dst = heap[0]
        if est_value >= min_count + margin:
            heapq.heappop(heap)
            del row_map[min_dst]
            row_map[dst] = int(est_value)
            heapq.heappush(heap, (int(est_value), dst))
            return True
        return False


class _CsrStore:
    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, row: int, dst: int) -> int:
        data = self.rows.get(row)
        if data is None:
            return 0
        indices, values = data
        if indices.size == 0:
            return 0
        lo, hi = 0, indices.size - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            value = int(indices[mid])
            if value == dst:
                return int(values[mid])
            if value < dst:
                lo = mid + 1
            else:
                hi = mid - 1
        return 0

    def items(self, row: int) -> List[Tuple[int, int]]:
        data = self.rows.get(row)
        if data is None:
            return []
        indices, values = data
        return [(int(indices[i]), int(values[i])) for i in range(int(indices.size))]

    @staticmethod
    def _merge_sorted_arrays(
        a_idx: np.ndarray,
        a_dat: np.ndarray,
        d_idx: np.ndarray,
        d_dat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        na, nd = a_idx.size, d_idx.size
        out_idx: List[int] = []
        out_dat: List[int] = []
        ia = 0
        ib = 0
        while ia < na and ib < nd:
            va = int(a_idx[ia])
            vb = int(d_idx[ib])
            if va == vb:
                out_idx.append(va)
                out_dat.append(int(a_dat[ia]) + int(d_dat[ib]))
                ia += 1
                ib += 1
            elif va < vb:
                out_idx.append(va)
                out_dat.append(int(a_dat[ia]))
                ia += 1
            else:
                out_idx.append(vb)
                out_dat.append(int(d_dat[ib]))
                ib += 1
        while ia < na:
            out_idx.append(int(a_idx[ia]))
            out_dat.append(int(a_dat[ia]))
            ia += 1
        while ib < nd:
            out_idx.append(int(d_idx[ib]))
            out_dat.append(int(d_dat[ib]))
            ib += 1
        return (
            np.array(out_idx, dtype=np.int32),
            np.array(out_dat, dtype=np.uint32),
        )

    def merge_delta(self, delta_row: Dict[int, Dict[int, int]]) -> None:
        for row, delta in delta_row.items():
            if not delta:
                continue
            keys = np.fromiter(sorted(delta.keys()), dtype=np.int32, count=len(delta))
            values = np.fromiter(
                (delta[key] for key in sorted(delta.keys())),
                dtype=np.uint32,
                count=len(delta),
            )
            base = self.rows.get(row)
            if base is None:
                self.rows[row] = (keys, values)
                continue
            a_idx, a_dat = base
            self.rows[row] = self._merge_sorted_arrays(a_idx, a_dat, keys, values)

    def scale_down_half(self) -> None:
        for row, (indices, values) in list(self.rows.items()):
            values = (values >> 1).astype(np.uint32)
            mask = values > 0
            self.rows[row] = (indices[mask], values[mask])


class _WalLog:
    __slots__ = ("src", "dst", "w", "bytes")

    def __init__(self) -> None:
        self.src: List[int] = []
        self.dst: List[int] = []
        self.w: List[int] = []
        self.bytes = 0

    def append(self, src: int, dst: int, weight: int) -> None:
        self.src.append(int(src))
        self.dst.append(int(dst))
        self.w.append(int(weight))
        self.bytes += 12

    def cut_and_aggregate(self) -> Dict[int, Dict[int, int]]:
        if not self.src:
            return {}
        src = np.asarray(self.src, dtype=np.int64)
        dst = np.asarray(self.dst, dtype=np.int64)
        weight = np.asarray(self.w, dtype=np.int64)
        order = np.lexsort((dst, src))
        src = src[order]
        dst = dst[order]
        weight = weight[order]
        out: Dict[int, Dict[int, int]] = {}
        current_src = int(src[0])
        current_dst = int(dst[0])
        current_weight = int(weight[0])
        for idx in range(1, src.size):
            src_i = int(src[idx])
            dst_i = int(dst[idx])
            weight_i = int(weight[idx])
            if src_i == current_src and dst_i == current_dst:
                current_weight += weight_i
            else:
                row = out.setdefault(current_src, {})
                row[current_dst] = row.get(current_dst, 0) + current_weight
                current_src, current_dst, current_weight = src_i, dst_i, weight_i
        row = out.setdefault(current_src, {})
        row[current_dst] = row.get(current_dst, 0) + current_weight
        self.src.clear()
        self.dst.clear()
        self.w.clear()
        self.bytes = 0
        return out


class _RowCache:
    __slots__ = ("cap", "map", "clock", "tick")

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.map: Dict[int, List[Tuple[int, int]]] = {}
        self.clock: Dict[int, int] = {}
        self.tick = 0

    def get(self, row: int) -> Optional[List[Tuple[int, int]]]:
        value = self.map.get(row)
        if value is not None:
            self.tick += 1
            self.clock[row] = self.tick
        return value

    def put(self, row: int, items: List[Tuple[int, int]]) -> None:
        self.tick += 1
        self.map[row] = items
        self.clock[row] = self.tick
        if len(self.map) > self.cap:
            victim = min(self.clock.items(), key=lambda kv: kv[1])[0]
            self.map.pop(victim, None)
            self.clock.pop(victim, None)


class _UnifiedRowStore:
    __slots__ = (
        "cfg",
        "cms",
        "topk",
        "csr",
        "wal",
        "rowsum_map",
        "row_cache",
        "update_counter",
        "last_decay_at",
    )

    def __init__(self, cfg: _CESConfig):
        self.cfg = cfg
        self.cms = _CmsSketch(self.cfg)
        self.topk = _TopKIndex()
        self.csr = _CsrStore()
        self.wal = _WalLog()
        self.rowsum_map: Dict[int, int] = {}
        self.row_cache = _RowCache(self.cfg.row_cache_capacity)
        self.update_counter = 0
        self.last_decay_at = 0
        np.random.seed(self.cfg.seed)

    def _rowsum(self, row: int) -> int:
        return int(self.rowsum_map.get(row, 0))

    def _auto_decay_if_needed(self) -> None:
        if (
            self.update_counter - self.last_decay_at
            >= self.cfg.decay_half_life_updates
        ):
            self.scale_down_half()
            self.last_decay_at = self.update_counter

    def _auto_compact_if_needed(self) -> None:
        if self.wal.bytes >= self.cfg.wal_segment_bytes:
            self.compact()

    def _invalidate_row_cache(self, row: Optional[int] = None) -> None:
        if row is None:
            self.row_cache = _RowCache(self.cfg.row_cache_capacity)
            return
        if row in self.row_cache.map:
            self.row_cache.map.pop(row, None)
            self.row_cache.clock.pop(row, None)

    def add(self, src: int, dst: int, weight: int = 1) -> None:
        weight = int(weight)
        if weight <= 0:
            return
        self.update_counter += 1
        self.rowsum_map[src] = self._rowsum(src) + weight
        self.wal.append(src, dst, weight)
        self._invalidate_row_cache(src)
        if self.topk.has(src, dst):
            self.topk.inc(self.cfg, src, dst, weight, self._rowsum(src))
        else:
            estimate = self.cms.estimate(src, dst) + weight
            if not self.topk.maybe_promote(
                self.cfg, src, dst, estimate, self._rowsum(src)
            ):
                self.cms.add(src, dst, weight)
        self._auto_compact_if_needed()
        self._auto_decay_if_needed()

    def get(self, src: int, dst: int) -> int:
        topk_value = self.topk.get(src, dst)
        if topk_value > 0:
            return topk_value
        csr_value = self.csr.get(src, dst)
        if csr_value > 0:
            return csr_value
        return self.cms.estimate(src, dst)

    def topk_items(self, src: int, k: Optional[int] = None) -> List[Tuple[int, int]]:
        rowsum = self._rowsum(src)
        limit = (
            int(k)
            if k is not None
            else _TopKIndex._dyn_k(self.cfg, rowsum)
        )
        return self.topk.list(src, limit)

    def iter_row(self, src: int) -> List[Tuple[int, int]]:
        cached = self.row_cache.get(src)
        if cached is not None:
            return cached
        row = {dst: count for dst, count in self.csr.items(src)}
        for dst, count in self.topk_items(src):
            row[dst] = count
        items = sorted(row.items(), key=lambda item: item[0])
        self.row_cache.put(src, items)
        return items

    def rowsum(self, src: int) -> int:
        return self._rowsum(src)

    def compact(self) -> None:
        delta = self.wal.cut_and_aggregate()
        if not delta:
            return
        self.csr.merge_delta(delta)
        self._invalidate_row_cache(None)

    def scale_down_half(self) -> None:
        self.cms.scale_down_half()
        for row, row_map in list(self.topk.map.items()):
            new_map = {dst: (count >> 1) for dst, count in row_map.items() if count > 0}
            if new_map:
                self.topk.map[row] = new_map
                self.topk._rebuild_heap(row)
            else:
                self.topk.map.pop(row, None)
                self.topk.heap.pop(row, None)
        self.csr.scale_down_half()
        for row, rowsum in list(self.rowsum_map.items()):
            self.rowsum_map[row] = int(rowsum >> 1)
        self._invalidate_row_cache(None)

    def set_seed(self, seed: int) -> None:
        self.cfg.seed = int(seed)
        np.random.seed(self.cfg.seed)
        rng = np.random.RandomState(self.cfg.seed)
        self.cms.hseeds = [
            np.uint64(rng.randint(0, 2**31 - 1)) for _ in range(self.cms.depth)
        ]


class UnifiedTransitionStore:
    """Sparse heavy-hitter transition tracker built on top of _UnifiedRowStore."""

    __slots__ = (
        "cfg",
        "n_elements",
        "history_length",
        "state_history",
        "_forward",
        "_backward",
        "_last_state",
        "_state_visit_counts",
        "_total_transitions",
    )

    def __init__(
        self,
        n_elements: int,
        history_length: int = 20,
        cfg: Optional[_CESConfig] = None,
    ) -> None:
        self.cfg = cfg or _CESConfig()
        self.n_elements = int(n_elements)
        self.history_length = int(history_length)
        self.state_history: deque = deque(maxlen=self.history_length)
        self._forward = _UnifiedRowStore(self.cfg)
        self._backward = _UnifiedRowStore(self.cfg)
        self._last_state: Optional[np.ndarray] = None
        self._state_visit_counts: Dict[int, int] = {}
        self._total_transitions = 0

    def _state_to_index(self, binary_state: np.ndarray) -> int:
        state = (
            binary_state[: self.n_elements]
            if len(binary_state) > self.n_elements
            else binary_state
        )
        if len(state) < self.n_elements:
            padded = np.zeros(self.n_elements, dtype=int)
            padded[: len(state)] = state
            state = padded
        powers = 2 ** np.arange(len(state))[::-1]
        index = int(np.dot(state, powers))
        max_index = (1 << self.n_elements) - 1
        return min(max(index, 0), max_index)

    def _index_to_state(self, index: int) -> np.ndarray:
        max_index = (1 << self.n_elements) - 1
        safe_index = min(max(int(index), 0), max_index)
        binary = format(safe_index, f"0{self.n_elements}b")
        return np.array([int(bit) for bit in binary], dtype=int)

    def _reset_for_size(self, new_size: int) -> None:
        self.n_elements = int(new_size)
        self.state_history = deque(maxlen=self.history_length)
        self._forward = _UnifiedRowStore(self.cfg)
        self._backward = _UnifiedRowStore(self.cfg)
        self._last_state = None
        self._state_visit_counts = {}
        self._total_transitions = 0

    def update(self, state: np.ndarray) -> None:
        state = np.asarray(state)
        if state.size == 0:
            return
        if len(state) != self.n_elements:
            self._reset_for_size(len(state))
        median = float(np.median(state))
        binary_state = (state > median).astype(int)
        current_idx = self._state_to_index(binary_state)
        if self._last_state is not None and len(self._last_state) == len(binary_state):
            src_idx = self._state_to_index(self._last_state)
            self._forward.add(src_idx, current_idx, 1)
            self._backward.add(current_idx, src_idx, 1)
            self._total_transitions += 1
        self.state_history.append(binary_state.copy())
        self._state_visit_counts[current_idx] = (
            self._state_visit_counts.get(current_idx, 0) + 1
        )
        self._last_state = binary_state.copy()

    def _row_distribution(self, store: _UnifiedRowStore, row: int) -> np.ndarray:
        n_states = 1 << self.n_elements
        if n_states == 0:
            return np.array([], dtype=np.float64)
        distribution = np.zeros(n_states, dtype=np.float64)
        for dst, count in store.iter_row(row):
            if 0 <= dst < n_states:
                distribution[dst] = float(count)
        total = float(distribution.sum())
        if total <= 0:
            return np.ones(n_states, dtype=np.float64) / max(n_states, 1)
        distribution += float(self.cfg.alpha)
        distribution /= float(distribution.sum())
        return distribution

    def get_cause_repertoire(self, current_state: np.ndarray) -> np.ndarray:
        row = self._state_to_index(current_state)
        return self._row_distribution(self._backward, row)

    def get_effect_repertoire(self, current_state: np.ndarray) -> np.ndarray:
        row = self._state_to_index(current_state)
        return self._row_distribution(self._forward, row)

    def has_effect_model(self) -> bool:
        return self._total_transitions > 0 and bool(self._forward.rowsum_map)

    def has_cause_model(self) -> bool:
        return self._total_transitions > 0 and bool(self._backward.rowsum_map)

    def has_any_transitions(self) -> bool:
        return self._total_transitions > 0

    def get_structural_health(self) -> Dict[str, float]:
        n_states = max(1, 1 << self.n_elements)
        observed_rows = len(self._forward.rowsum_map)
        tpm_confidence = float(observed_rows) / float(n_states)
        coherence_scores: List[float] = []
        for row, rowsum in self._forward.rowsum_map.items():
            if rowsum <= 0:
                continue
            items = self._forward.iter_row(row)
            if not items:
                continue
            counts = np.array([count for _, count in items], dtype=float)
            total = counts.sum()
            if total <= 0:
                continue
            coherence_scores.append(float(counts.max() / total))
        structural_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        if self._total_transitions == 0:
            tpm_stability = 0.0
        else:
            decay_period = max(1, self.cfg.decay_half_life_updates)
            stability_factor = min(1.0, self._total_transitions / (decay_period * 0.5))
            tpm_stability = float(0.2 + 0.8 * stability_factor)
        overall_health = float(
            np.mean(
                [
                    float(structural_coherence),
                    float(tpm_stability),
                    float(tpm_confidence),
                ]
            )
        )
        return {
            "structural_coherence": float(structural_coherence),
            "tpm_stability": float(tpm_stability),
            "causal_density": 0.0,
            "tpm_confidence": float(tpm_confidence),
            "overall_health": overall_health,
        }

    def prob(self, src: int, dst: int) -> float:
        rowsum = max(self._forward.rowsum(src), 0)
        v_eff = max(1, len(self._forward.iter_row(src)))
        numerator = self._forward.get(src, dst) + self.cfg.alpha
        denominator = rowsum + self.cfg.alpha * v_eff
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def compact(self) -> None:
        self._forward.compact()
        self._backward.compact()

    def decay(self) -> None:
        self._forward.scale_down_half()
        self._backward.scale_down_half()
        for key, value in list(self._state_visit_counts.items()):
            new_value = value >> 1
            if new_value > 0:
                self._state_visit_counts[key] = new_value
            else:
                self._state_visit_counts.pop(key, None)
        if self._total_transitions > 0:
            self._total_transitions >>= 1

    def set_seed(self, seed: int) -> None:
        self.cfg.seed = int(seed)
        self._forward.set_seed(seed)
        self._backward.set_seed(seed + 1)

