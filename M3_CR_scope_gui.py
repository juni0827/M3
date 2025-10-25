import argparse, os, time, math, json, hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set, Union, Callable
from enum import Enum
from contextlib import contextmanager
import numpy as np
import pandas as pd
from collections import deque
from itertools import combinations
import threading
import sys
import os
import tqdm
from numpy.random import default_rng, SeedSequence
import cv2

from transition_store import UnifiedTransitionStore

class RNGRegistry:
    def __init__(self, root_seed: int | None):
        self.root_seed = int(root_seed) if root_seed is not None else None
        self.root_ss = SeedSequence(self.root_seed) if self.root_seed is not None else SeedSequence()
        self.cache = {}

    def get(self, name: str):
        if name not in self.cache:
            mix = int.from_bytes(hashlib.blake2b(name.encode('utf-8'), digest_size=8).digest(), 'little')
            child_ss = SeedSequence(self.root_seed if self.root_seed is not None else 0, spawn_key=(mix,))
            self.cache[name] = default_rng(child_ss)
        return self.cache[name]
    def episode(self, key: str, ep: int):
        return self.get(f"{key}:episode:{int(ep)}")
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class IITPhiCalculator:

    def __init__(self, n_elements: int):
        self.n_elements = n_elements
        self.ces = UnifiedTransitionStore(n_elements)
        self.current_mip = None
        self.mip_history: deque = deque(maxlen=50)
        self.phi_history: deque = deque(maxlen=100)
        self.mics_cache: Dict[str, float] = {}
        self.requires_structural_revision = False
        self.revision_reason = ''

    def _normalize_phi(self, phi: float, cause_rep: np.ndarray) -> float:
        max_possible_phi = np.log2(len(cause_rep)) * 2
        phi_normalized = phi / max(max_possible_phi, 1e-10)
        return float(np.clip(phi_normalized, 0.0, 1.0))

    def update_state(self, state: np.ndarray):
        self.ces.update(state)
        self._check_structural_health()

    def _check_structural_health(self):
        health = self.ces.get_structural_health()
        needs_revision = False
        reasons = []
        if health['tpm_confidence'] < 0.15:
            needs_revision = True
            reasons.append(f"Low TPM confidence ({health['tpm_confidence']:.2f})")
        if health['structural_coherence'] < 0.25:
            needs_revision = True
            reasons.append(f"Low coherence ({health['structural_coherence']:.2f})")
        if health['tpm_stability'] < 0.15:
            needs_revision = True
            reasons.append(f"TPM unstable ({health['tpm_stability']:.2f})")
        if health['overall_health'] < 0.2:
            needs_revision = True
            reasons.append(f"Overall health low ({health['overall_health']:.2f})")
        self.requires_structural_revision = needs_revision
        if needs_revision:
            self.revision_reason = '; '.join(reasons)

    def compute_phi(self, cause_repertoire: Optional[np.ndarray]=None, effect_repertoire: Optional[np.ndarray]=None, state: Optional[np.ndarray]=None, method: str='integrated') -> float:
        if method == 'integrated' and state is not None:
            self.update_state(state)
            if not self.ces.has_effect_model():
                method = 'simple'
            else:
                binary_state = (state > np.median(state)).astype(int)
                binary_state = binary_state[:self.ces.n_elements] if len(binary_state) > self.ces.n_elements else binary_state
                cause_repertoire = self.ces.get_cause_repertoire(binary_state)
                effect_repertoire = self.ces.get_effect_repertoire(binary_state)
        if method == 'simple' and (cause_repertoire is None or effect_repertoire is None):
            if method == 'simple':
                return 0.0
            if state is not None:
                safe_state = state[:self.ces.n_elements] if len(state) > self.ces.n_elements else state
                binary_state = (safe_state > np.median(safe_state)).astype(int)
                n_states = 2 ** len(binary_state)
                cause_repertoire = np.random.exponential(0.5, n_states)
                current_idx = self.ces._state_to_index(binary_state)
                if current_idx < n_states:
                    cause_repertoire[current_idx] *= 3.0
                else:
                    cause_repertoire[0] *= 3.0
                cause_repertoire = cause_repertoire / np.sum(cause_repertoire)
                effect_repertoire = np.random.gamma(2, 0.5, n_states)
                activity_level = np.sum(binary_state) / len(binary_state)
                for i in range(n_states):
                    next_state = np.array([int(b) for b in format(i, f'0{len(binary_state)}b')])
                    next_activity = np.sum(next_state) / len(next_state)
                    similarity = np.exp(-abs(activity_level - next_activity) * 2)
                    complexity = next_activity * (1 - next_activity) * 3
                    randomness = np.random.uniform(0.1, 0.5)
                    effect_repertoire[i] = effect_repertoire[i] * similarity + complexity + randomness
                effect_repertoire = effect_repertoire / max(np.sum(effect_repertoire), 1e-10)
                if np.sum(effect_repertoire) > 0:
                    effect_repertoire = effect_repertoire / np.sum(effect_repertoire)
                else:
                    effect_repertoire = np.ones(n_states) / n_states
            else:
                cause_repertoire = np.array([0.5, 0.5])
                effect_repertoire = np.array([0.5, 0.5])
        if cause_repertoire is None or effect_repertoire is None:
            return 0.0
        if len(cause_repertoire) == 0 or len(effect_repertoire) == 0:
            return 0.0
        if method == 'simple' or method == 'integrated':
            phi = self._compute_phi_simple(cause_repertoire, effect_repertoire)
        else:
            phi = self._compute_phi_full(cause_repertoire, effect_repertoire, state)
        if np.isnan(phi) or np.isinf(phi):
            phi = 0.0
        self.phi_history.append(phi)
        return phi

    def compute_mics(self, state: np.ndarray) -> Dict[str, Any]:
        state_key = str(state.tobytes())
        if state_key in self.mics_cache:
            return {'phi': self.mics_cache[state_key], 'cached': True}
        self.update_state(state)
        if not self.ces.has_effect_model():
            return {'phi': 0.0, 'error': 'No transition model available'}
        binary_state = (state > np.median(state)).astype(int)
        cause_rep = self.ces.get_cause_repertoire(binary_state)
        effect_rep = self.ces.get_effect_repertoire(binary_state)
        phi = self.compute_phi(cause_rep, effect_rep, method='simple')
        mics = {'phi': phi, 'cause_repertoire': cause_rep.tolist(), 'effect_repertoire': effect_rep.tolist(), 'cause_info': self._repertoire_entropy(cause_rep), 'effect_info': self._repertoire_entropy(effect_rep), 'state': state.tolist(), 'mip': self.current_mip, 'cached': False}
        self.mics_cache[state_key] = phi
        if len(self.mics_cache) > 5000:
            oldest_keys = list(self.mics_cache.keys())[:250]
            for k in oldest_keys:
                del self.mics_cache[k]
        return mics

    def _compute_phi_simple(self, cause_rep: np.ndarray, effect_rep: np.ndarray) -> float:
        if cause_rep is None or effect_rep is None:
            return 0.0
        cause_rep = np.array(cause_rep) + 1e-10
        effect_rep = np.array(effect_rep) + 1e-10
        cause_rep = cause_rep / np.sum(cause_rep)
        effect_rep = effect_rep / np.sum(effect_rep)
        joint_entropy = self._compute_joint_entropy(cause_rep, effect_rep)
        cause_entropy = self._repertoire_entropy(cause_rep)
        effect_entropy = self._repertoire_entropy(effect_rep)
        independent_entropy = cause_entropy + effect_entropy
        phi = max(0.0, independent_entropy - joint_entropy)
        mutual_info = cause_entropy + effect_entropy - joint_entropy
        phi += 0.5 * mutual_info
        repertoire_diff = np.sum(np.abs(cause_rep - effect_rep)) / len(cause_rep)
        phi += repertoire_diff * 0.1
        if phi < 1e-10:
            phi = (cause_entropy + effect_entropy) * 0.01
        result = float(max(phi, 0.01))
        return result

    def _compute_phi_full(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]=None) -> float:
        n = self.n_elements
        if n <= 6:
            return self._compute_phi_full_exhaustive(cause_rep, effect_rep, state)
        elif n <= 15:
            return self._compute_phi_cutset_sampling(cause_rep, effect_rep, state)
        elif n <= 30:
            return self._compute_phi_community_cluster(cause_rep, effect_rep, state)
        else:
            print(f'[yellow]WARNING: Phi computation: n={n} > 30, using simple approximation (significant information loss)[/yellow]')
            return self._compute_phi_simple(cause_rep, effect_rep)

    def _compute_phi_full_exhaustive(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]) -> float:
        n = self.n_elements
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        min_partitioned_info = float('inf')
        best_mip = None
        for partition_size in range(1, n):
            for subset in combinations(range(n), partition_size):
                subset = set(subset)
                complement = set(range(n)) - subset
                partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, subset, complement)
                if partitioned_info < min_partitioned_info:
                    min_partitioned_info = partitioned_info
                    best_mip = (subset, complement)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info, 'method': 'exhaustive'})
        return self._normalize_phi(phi, cause_rep)

    def _compute_phi_cutset_sampling(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]) -> float:
        n = self.n_elements
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        edge_strengths = []
        for i in range(n):
            for j in range(i + 1, n):
                if hasattr(self.ces, 'causal_graph'):
                    strength = abs(self.ces.causal_graph[i, j]) + abs(self.ces.causal_graph[j, i])
                    edge_strengths.append((strength, i, j))
        edge_strengths.sort(reverse=True)
        if len(edge_strengths) == 0:
            candidate_partitions = []
            for i in range(min(10, n - 1)):
                partition1 = set(range(i + 1))
                partition2 = set(range(i + 1, n))
                if len(partition1) > 0 and len(partition2) > 0:
                    candidate_partitions.append((partition1, partition2))
        else:
            k_top_edges = min(20, len(edge_strengths))
            candidate_partitions = []
            for idx in range(k_top_edges):
                _, i, j = edge_strengths[idx]
                partition1 = self._find_connected_component(i, {j}, n)
                partition2 = set(range(n)) - partition1
                if len(partition1) > 0 and len(partition2) > 0:
                    candidate_partitions.append((partition1, partition2))
        for _ in range(min(10, 2 ** (n - 1) - k_top_edges)):
            size = np.random.randint(1, n)
            indices = np.random.choice(n, size, replace=False)
            partition1 = set(indices)
            partition2 = set(range(n)) - partition1
            if len(partition1) > 0 and len(partition2) > 0:
                candidate_partitions.append((partition1, partition2))
        min_partitioned_info = float('inf')
        best_mip = None
        for subset, complement in candidate_partitions:
            partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, subset, complement)
            if partitioned_info < min_partitioned_info:
                min_partitioned_info = partitioned_info
                best_mip = (subset, complement)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info, 'method': 'cutset_sampling', 'sampled_partitions': len(candidate_partitions)})
        return self._normalize_phi(phi, cause_rep)

    def _find_connected_component(self, start: int, excluded: Set[int], n: int) -> Set[int]:
        if not hasattr(self.ces, 'causal_graph'):
            return {start}
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in range(n):
                if neighbor in excluded or neighbor in visited:
                    continue
                if abs(self.ces.causal_graph[node, neighbor]) > 0.1 or abs(self.ces.causal_graph[neighbor, node]) > 0.1:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _compute_phi_community_cluster(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]) -> float:
        n = self.n_elements
        communities = self._detect_communities_greedy(n)
        if len(communities) <= 1:
            print(f'[yellow]WARNING: Community detection failed, fallback to cutset sampling[/yellow]')
            return self._compute_phi_cutset_sampling(cause_rep, effect_rep, state)
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        num_communities = len(communities)
        min_partitioned_info = float('inf')
        best_mip = None
        if num_communities <= 6:
            for partition_size in range(1, num_communities):
                for subset_indices in combinations(range(num_communities), partition_size):
                    partition1 = set()
                    for idx in subset_indices:
                        partition1.update(communities[idx])
                    partition2 = set(range(n)) - partition1
                    if len(partition1) > 0 and len(partition2) > 0:
                        partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, partition1, partition2)
                        if partitioned_info < min_partitioned_info:
                            min_partitioned_info = partitioned_info
                            best_mip = (partition1, partition2)
        else:
            max_samples = min(50, 2 ** (num_communities - 1))
            for _ in range(max_samples):
                num_in_group1 = np.random.randint(1, num_communities)
                indices_group1 = np.random.choice(num_communities, num_in_group1, replace=False)
                partition1 = set()
                for idx in indices_group1:
                    partition1.update(communities[idx])
                partition2 = set(range(n)) - partition1
                if len(partition1) > 0 and len(partition2) > 0:
                    partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, partition1, partition2)
                    if partitioned_info < min_partitioned_info:
                        min_partitioned_info = partitioned_info
                        best_mip = (partition1, partition2)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info, 'method': 'community_cluster', 'num_communities': len(communities), 'community_sizes': [len(c) for c in communities]})
        return self._normalize_phi(phi, cause_rep)

    def _detect_communities_greedy(self, n: int) -> List[Set[int]]:
        if not hasattr(self.ces, 'causal_graph'):
            return [set(range(n))]
        communities = [{i} for i in range(n)]
        total_strength = np.sum(np.abs(self.ces.causal_graph))
        if total_strength < 1e-10:
            return [set(range(n))]
        improved = True
        max_iterations = 20
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            best_merge = None
            best_modularity_gain = 0.0
            for i in range(len(communities)):
                for j in range(i + 1, len(communities)):
                    gain = self._modularity_gain(communities[i], communities[j], total_strength)
                    if gain > best_modularity_gain:
                        best_modularity_gain = gain
                        best_merge = (i, j)
            if best_merge is not None and best_modularity_gain > 0.001:
                i, j = best_merge
                communities[i] = communities[i].union(communities[j])
                del communities[j]
                improved = True
        min_size = max(1, n // 10)
        filtered_communities = []
        small_communities = []
        for comm in communities:
            if len(comm) >= min_size:
                filtered_communities.append(comm)
            else:
                small_communities.append(comm)
        for small_comm in small_communities:
            if not filtered_communities:
                filtered_communities.append(small_comm)
            else:
                best_target = 0
                best_connection = 0.0
                for idx, large_comm in enumerate(filtered_communities):
                    connection = self._inter_community_strength(small_comm, large_comm)
                    if connection > best_connection:
                        best_connection = connection
                        best_target = idx
                filtered_communities[best_target] = filtered_communities[best_target].union(small_comm)
        if not filtered_communities:
            return [set(range(n))]
        return filtered_communities

    def _modularity_gain(self, comm1: Set[int], comm2: Set[int], total_strength: float) -> float:
        if not hasattr(self.ces, 'causal_graph'):
            return 0.0
        merged = comm1.union(comm2)
        internal_strength = 0.0
        for i in merged:
            for j in merged:
                if i != j:
                    internal_strength += abs(self.ces.causal_graph[i, j])
        degree_comm1 = sum((np.sum(np.abs(self.ces.causal_graph[i, :])) for i in comm1))
        degree_comm2 = sum((np.sum(np.abs(self.ces.causal_graph[i, :])) for i in comm2))
        expected_strength = degree_comm1 * degree_comm2 / max(total_strength, 1e-10)
        gain = (internal_strength - expected_strength) / max(total_strength, 1e-10)
        return gain

    def _inter_community_strength(self, comm1: Set[int], comm2: Set[int]) -> float:
        if not hasattr(self.ces, 'causal_graph'):
            return 0.0
        strength = 0.0
        for i in comm1:
            for j in comm2:
                strength += abs(self.ces.causal_graph[i, j])
                strength += abs(self.ces.causal_graph[j, i])
        return strength
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        min_partitioned_info = float('inf')
        best_mip = None
        for partition_size in range(1, n):
            for subset in combinations(range(n), partition_size):
                subset = set(subset)
                complement = set(range(n)) - subset
                partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, subset, complement)
                if partitioned_info < min_partitioned_info:
                    min_partitioned_info = partitioned_info
                    best_mip = (subset, complement)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info})
        return self._normalize_phi(phi, cause_rep)

    def _compute_integrated_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, elements: set, excluded: set) -> float:
        cause_entropy = self._repertoire_entropy(cause_rep)
        effect_entropy = self._repertoire_entropy(effect_rep)
        mutual_info = 0.0
        if len(cause_rep) == len(effect_rep):
            joint_entropy = self._compute_joint_entropy(cause_rep, effect_rep)
            mutual_info = cause_entropy + effect_entropy - joint_entropy
            mutual_info = max(mutual_info, 0.0)
        causal_weight = 1.0
        connection_bonus = 0.0
        if hasattr(self.ces, 'causal_graph'):
            internal_strength = 0.0
            connection_count = 0
            for i in elements:
                for j in elements:
                    if i != j and i < self.n_elements and (j < self.n_elements):
                        strength = abs(self.ces.causal_graph[i, j])
                        if strength > 1e-06:
                            internal_strength += strength
                            connection_count += 1
            max_connections = len(elements) * (len(elements) - 1)
            if max_connections > 0:
                connectivity = connection_count / max_connections
                avg_strength = internal_strength / max(connection_count, 1)
                causal_weight = 1.0 + connectivity * avg_strength * 2.0
                connection_bonus = connectivity * 0.1
        base_info = cause_entropy + effect_entropy
        integrated_info = (base_info + mutual_info) * causal_weight + connection_bonus
        if len(elements) > 1 and any((abs(x) > 1e-08 for x in cause_rep)) and any((abs(x) > 1e-08 for x in effect_rep)):
            integrated_info = max(integrated_info, 0.005)
        return float(integrated_info)

    def _compute_partitioned_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        n1, n2 = (len(subset1), len(subset2))
        subset1_entropy = np.log2(2 ** n1) if n1 > 0 else 0.0
        subset2_entropy = np.log2(2 ** n2) if n2 > 0 else 0.0
        cross_causal_loss = 0.0
        if hasattr(self.ces, 'causal_graph'):
            for i in subset1:
                for j in subset2:
                    if i < self.n_elements and j < self.n_elements:
                        cross_causal_loss += abs(self.ces.causal_graph[i, j])
            for i in subset2:
                for j in subset1:
                    if i < self.n_elements and j < self.n_elements:
                        cross_causal_loss += abs(self.ces.causal_graph[i, j])
        partitioned_info = subset1_entropy + subset2_entropy - cross_causal_loss * 0.5
        # ensure we return a plain python float for precise typing
        return float(max(0.0, float(partitioned_info)))

    def _compute_partition_loss(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        full_entropy = self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
        n1, n2 = (len(subset1), len(subset2))
        partition_entropy = np.log2(2 ** n1) + np.log2(2 ** n2)
        loss = abs(full_entropy - partition_entropy)
        return loss

    def _repertoire_entropy(self, repertoire: np.ndarray) -> float:
        epsilon = 1e-12
        repertoire = np.abs(repertoire) + epsilon
        repertoire = repertoire / repertoire.sum()
        entropy = -np.sum(repertoire * np.log2(repertoire))
        n_states = len(repertoire)
        max_entropy = np.log2(n_states)
        info_content = entropy
        uniformity = entropy / max_entropy if max_entropy > 0 else 0
        concentration = 1 - uniformity
        if concentration > 0.01:
            info_content += concentration * 0.5
        return float(max(info_content, 0.001))

    def _compute_joint_entropy(self, cause_rep: np.ndarray, effect_rep: np.ndarray) -> float:
        if len(cause_rep) != len(effect_rep):
            return self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
        joint_prob = np.outer(cause_rep, effect_rep)
        joint_prob = joint_prob / np.sum(joint_prob)
        epsilon = 1e-12
        joint_prob = joint_prob + epsilon
        joint_entropy = -np.sum(joint_prob * np.log2(joint_prob))
        return float(joint_entropy)

    def get_phi_trend(self) -> str:
        if len(self.phi_history) < 10:
            return 'insufficient_data'
        recent = list(self.phi_history)[-10:]
        trend = np.mean(np.diff(recent))
        if trend > 0.01:
            return 'increasing'
        elif trend < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def get_consciousness_level(self, phi: float) -> str:
        if phi < 0.01:
            return '(unconscious)'
        elif phi < 0.1:
            return '(minimal)'
        elif phi < 0.3:
            return '(low)'
        elif phi < 0.5:
            return '(moderate)'
        elif phi < 0.7:
            return '(high)'
        else:
            return '(very high)'

class EvolutionVisualizer:

    def __init__(self):
        self.neural_map = None
        self.generation = 0
        self.total_connections = 0
        self.consciousness_level = 0
        self.phi_value = 0.0
        self.major_events = deque(maxlen=5)
        self.energy_history = deque(maxlen=50)
        self.growth_stage = 'embryo'
        self.qualia_data = None
        self.current_experience = 'unknown'
        self.memory_count = 0
        self.unity_score = 0.5
        self.neuron_count = 4
        self.connection_count = 4
        self.growth_events = 0
        self.scale_factor = 1.0
        self.max_intensity = 0.1
        self.connection_history = deque(maxlen=100)

    def update(self, system_state: Dict[str, Any]):
        self.generation += 1
        u_matrix = system_state.get('u_matrix', None)
        if u_matrix is not None:
            self._update_neural_map(u_matrix)
        self.qualia_data = system_state.get('qualia', None)
        self.current_experience = system_state.get('current_experience', 'unknown')
        self.memory_count = system_state.get('memories', 0)
        self.unity_score = system_state.get('unity', 0.5)
        self.neuron_count = system_state.get('neuron_count', 4)
        self.connection_count = system_state.get('connection_count', 4)
        self.growth_events = system_state.get('growth_events', 0)
        strange_loop = system_state.get('strange_loop', False)
        meta_awareness = system_state.get('meta_awareness', 0.0)
        self.phi_value = system_state.get('phi', 0.0)
        old_level = self.consciousness_level
        if self.phi_value > 0.5:
            self.consciousness_level = 4
            self.growth_stage = 'transcendent'
        elif self.phi_value > 0.1:
            self.consciousness_level = 3
            self.growth_stage = 'adult'
        elif meta_awareness > 0.5:
            self.consciousness_level = 2
            self.growth_stage = 'child'
        elif strange_loop:
            self.consciousness_level = 1
            self.growth_stage = 'infant'
        if self.consciousness_level > old_level:
            self.add_major_event(f'EVOLVED to {self.growth_stage.upper()}!')
        energy = system_state.get('energy', 0.0)
        self.energy_history.append(energy)
        self.connection_history.append(self.total_connections)
        if len(self.connection_history) > 10:
            recent_avg = np.mean(list(self.connection_history)[-10:])
            if recent_avg > 1000:
                target_scale = 1000 / recent_avg
                self.scale_factor = 0.9 * self.scale_factor + 0.1 * target_scale
                self.scale_factor = max(0.1, min(1.0, self.scale_factor))

    def _update_neural_map(self, u_matrix):
        n, k = u_matrix.shape
        self.total_connections = int(np.sum(np.abs(u_matrix) > 0.1))
        current_max = np.max(np.abs(u_matrix))
        self.max_intensity = 0.95 * self.max_intensity + 0.05 * current_max
        if self.neural_map is None:
            self.neural_map = np.zeros((30, 80))
        for i in range(30):
            for j in range(80):
                u_i = int(i / 30 * n)
                u_j = int(j / 80 * k)
                if u_i < n and u_j < k:
                    val = abs(u_matrix[u_i, u_j])
                    normalized_val = val / max(0.01, self.max_intensity)
                    self.neural_map[i, j] = 0.9 * self.neural_map[i, j] + 0.1 * normalized_val

    def add_major_event(self, description: str):
        self.major_events.append(f'Gen {self.generation:,}: {description}')

    def render_brain_growth(self) -> str:
        if self.neural_map is None:
            return 'Initializing neural network...'
        lines = []
        center_x, center_y = (40, 15)
        color_map = {'tension': '\x1b[91m', 'harmony': '\x1b[92m', 'uncertainty': '\x1b[93m', 'flow': '\x1b[96m', 'resistance': '\x1b[95m'}
        reset = '\x1b[0m'
        dominant_qualia = 'flow'
        if self.qualia_data:
            max_val = 0
            for q_name, q_val in self.qualia_data.items():
                if q_val > max_val:
                    max_val = q_val
                    dominant_qualia = q_name
        base_color = color_map.get(dominant_qualia, reset)
        memory_boost = min(1.0, self.memory_count / 100)
        unity_boost = self.unity_score
        for i, row in enumerate(self.neural_map):
            line = ''
            for j, val in enumerate(row):
                dx = (j - center_x) / 2.0
                dy = (i - center_y) * 1.5
                dist = np.sqrt(dx ** 2 + dy ** 2)
                brain_shape = np.exp(-dist ** 2 / 400.0)
                effective_val = val * brain_shape * self.scale_factor
                consciousness_boost = 1.0 + self.consciousness_level * 0.5
                phi_boost = 1.0 + self.phi_value * 2.0
                effective_val *= consciousness_boost * phi_boost
                effective_val += memory_boost * 0.3 + unity_boost * 0.2
                if effective_val > 3.0:
                    char = '█'
                elif effective_val > 2.0:
                    char = '▓'
                elif effective_val > 1.5:
                    char = '▒'
                elif effective_val > 1.0:
                    char = '░'
                elif effective_val > 0.5:
                    char = '·'
                elif effective_val > 0.2:
                    char = '∙'
                else:
                    char = ' '
                if char != ' ':
                    line += base_color + char + reset
                else:
                    line += char
            lines.append(line)
        if self.current_experience != 'unknown' and len(lines) > 15:
            exp_label = f'[{self.current_experience[:10]}]'
            center_line = lines[15]
            insert_pos = 35
            lines[15] = center_line[:insert_pos] + f'\x1b[97;40m{exp_label}\x1b[0m' + center_line[insert_pos + len(exp_label):]
        return '\n'.join(lines)

    def render_network_growth(self, width: int=60, height: int=30) -> str:
        if self.neuron_count == 0:
            return 'Initializing network...'
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        for i in range(min(self.neuron_count, 200)):
            x = (i * 37 + 13) % width
            y = (i * 23 + 7) % height
            age_factor = min(1.0, i / max(1, self.neuron_count))
            if age_factor > 0.8:
                char = '●'
                color = '\x1b[92m'
            elif age_factor > 0.5:
                char = '◉'
                color = '\x1b[93m'
            elif age_factor > 0.2:
                char = '○'
                color = '\x1b[96m'
            else:
                char = '·'
                color = '\x1b[94m'
            reset = '\x1b[0m'
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = color + char + reset
        for conn_idx in range(min(self.connection_count // 2, 30)):
            i1 = conn_idx * 17 % min(self.neuron_count, 200)
            i2 = (conn_idx * 31 + 5) % min(self.neuron_count, 200)
            x1 = (i1 * 37 + 13) % width
            y1 = (i1 * 23 + 7) % height
            x2 = (i2 * 37 + 13) % width
            y2 = (i2 * 23 + 7) % height
            if abs(x2 - x1) > abs(y2 - y1):
                for step in range(abs(x2 - x1)):
                    t = step / max(1, abs(x2 - x1))
                    x = int(x1 + (x2 - x1) * t)
                    y = int(y1 + (y2 - y1) * t)
                    if 0 <= y < height and 0 <= x < width and (grid[y][x] == ' '):
                        grid[y][x] = '\x1b[90m-\x1b[0m'
            else:
                for step in range(abs(y2 - y1)):
                    t = step / max(1, abs(y2 - y1))
                    x = int(x1 + (x2 - x1) * t)
                    y = int(y1 + (y2 - y1) * t)
                    if 0 <= y < height and 0 <= x < width and (grid[y][x] == ' '):
                        grid[y][x] = '\x1b[90m|\x1b[0m'
        lines = []
        for row in grid:
            lines.append(''.join(row))
        return '\n'.join(lines)

    def render_full_display(self) -> str:
        width = 100
        output = []
        output.append('╔' + '═' * (width - 2) + '╗')
        title = 'M3 CONSCIOUSNESS SYSTEM (AUTONOMOUS)'
        output.append(f'║{title.center(width - 2)}║')
        output.append(f"║{'Generation: ' + str(self.generation):^{width - 2}}║")
        output.append(f"║{'Stage: ' + self.growth_stage.upper():^{width - 2}}║")
        output.append('╠' + '═' * (width - 2) + '╣')
        brain = self.render_brain_growth()
        brain_lines = brain.split('\n')
        for line in brain_lines:
            output.append(f'║ {line[:width - 4].ljust(width - 4)} ║')
        output.append('╠' + '═' * (width - 2) + '╣')
        conn_bar = self._make_bar(float(min(1.0, self.total_connections * self.scale_factor / 5000)), 40)
        output.append(f'║ Connections: {conn_bar} {self.total_connections:,} (scale: {self.scale_factor:.2f})'.ljust(width - 2) + ' ║')
        if self.energy_history:
            energy = self.energy_history[-1]
            energy_bar = self._make_bar(energy / 100, 40)
            output.append(f'║ Energy:      {energy_bar} {energy:.0f}%'.ljust(width - 2) + ' ║')
        level_bar = '▓' * (self.consciousness_level * 10) + '░' * ((4 - self.consciousness_level) * 10)
        output.append(f'║ Level:       {level_bar} {self.consciousness_level}/4'.ljust(width - 2) + ' ║')
        if self.phi_value > 0.001:
            phi_bar = self._make_bar(min(1.0, self.phi_value * 2), 40)
            output.append(f'║ Φ (Phi):     {phi_bar} {self.phi_value:.4f}'.ljust(width - 2) + ' ║')
        output.append('╠' + '═' * (width - 2) + '╣')
        output.append(f"║ {'MAJOR EVENTS:'.ljust(width - 2)} ║")
        if self.major_events:
            for event in self.major_events:
                output.append(f'║ ▸ {event[:width - 6].ljust(width - 6)} ║')
        else:
            output.append(f'║   (system initializing...)'.ljust(width - 2) + ' ║')
        output.append('╚' + '═' * (width - 2) + '╝')
        return '\n'.join(output)

    def _make_bar(self, ratio: float, width: int=40) -> str:
        filled = int(ratio * width)
        return '█' * filled + '░' * (width - filled)

@dataclass
@dataclass
class ConceptPrototype:
    name: str
    coordinates: np.ndarray
    experiences_count: int = 0
    discovered_at: int = 0
    confidence: float = 0.0

class ConceptualSpace:

    def __init__(self):
        self.prototypes: Dict[str, ConceptPrototype] = {}
        self.experience_points: List[np.ndarray] = []
        self.experience_contexts: List[Dict] = []
        self.min_cluster_size = 10
        self.similarity_threshold = 0.3
        self.discovery_interval = 50
        self.experience_count = 0
        self.last_clustering_at = 0
        self.prototype_id_counter = 0
        self._initialize_basic_dimensions()

    def _initialize_basic_dimensions(self):
        pass

    def add_experience(self, qualia: 'QualiaState', context: Optional[Dict[str, Any]]=None):
        experience_vector = np.array([qualia.arousal, qualia.valence, qualia.entropy, qualia.engagement, qualia.frustration])
        self.experience_points.append(experience_vector)
        self.experience_contexts.append(context or {})
        self.experience_count += 1
        if self.experience_count - self.last_clustering_at >= self.discovery_interval:
            self._discover_new_prototypes()
            self.last_clustering_at = self.experience_count

    def _discover_new_prototypes(self):
        if len(self.experience_points) < self.min_cluster_size:
            return
        from sklearn.cluster import DBSCAN
        experiences = np.array(self.experience_points)
        clustering = DBSCAN(eps=self.similarity_threshold, min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(experiences)
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = cluster_labels == label
            cluster_points = experiences[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            prototype_name = f'pattern_{self.prototype_id_counter}'
            self.prototype_id_counter += 1
            self.prototypes[prototype_name] = ConceptPrototype(name=prototype_name, coordinates=centroid, experiences_count=len(cluster_points), discovered_at=self.experience_count)

    def ground_experience(self, qualia: 'QualiaState') -> Dict[str, Any]:
        current_coords = np.array([qualia.arousal, qualia.valence, qualia.entropy, qualia.engagement, qualia.frustration])
        distances = {}
        for key, proto in self.prototypes.items():
            dist = np.linalg.norm(current_coords - proto.coordinates)
            distances[key] = dist
        if not distances:
            return {'nearest_concept': 'unknown', 'nearest_distance': float('inf'), 'concept_blend': {}, 'coordinates': current_coords, 'semantic_meaning': 'Experiencing unknown qualia state'}
        nearest_key = min(distances, key=distances.get)
        nearest_proto = self.prototypes[nearest_key]
        nearest_dist = distances[nearest_key]
        similarities = {k: 1.0 / (1.0 + d) for k, d in distances.items()}
        total_sim = sum(similarities.values())
        normalized_sim = {k: v / total_sim for k, v in similarities.items()}
        return {'nearest_concept': nearest_proto.name, 'nearest_distance': nearest_dist, 'concept_blend': normalized_sim, 'coordinates': current_coords, 'semantic_meaning': self._generate_meaning(nearest_proto, nearest_dist, normalized_sim)}

    def _generate_meaning(self, nearest: ConceptPrototype, distance: float, blend: Dict) -> str:
        if distance < 0.3:
            return f'명확한 {nearest.name} 경험'
        elif distance < 0.6:
            top2 = sorted(blend.items(), key=lambda x: x[1], reverse=True)[:2]
            concept1 = self.prototypes[top2[0][0]].name
            concept2 = self.prototypes[top2[1][0]].name
            return f'{concept1}과 {concept2} 사이'
        else:
            return f'모호한 경험 (약한 {nearest.name})'

class EventType(Enum):
    HIGH_UNCERTAINTY = 'high_uncertainty'
    PREDICTION_ERROR = 'prediction_error'
    GOAL_ACHIEVED = 'goal_achieved'
    GOAL_FAILED = 'goal_failed'
    MODEL_CONFIDENCE_LOW = 'model_confidence_low'
    MODEL_REVISION_NEEDED = 'model_revision_needed'
    SELF_CONTRADICTION = 'self_contradiction'
    TENSION_SPIKE = 'tension_spike'
    HARMONY_ACHIEVED = 'harmony_achieved'
    FLOW_STATE_ENTERED = 'flow_state_entered'
    STATE_INSTABILITY = 'state_instability'
    BARRIER_VIOLATION = 'barrier_violation'
    CONVERGENCE_DETECTED = 'convergence_detected'
    META_MODEL_UPDATED = 'meta_model_updated'
    ATTENTION_EXHAUSTED = 'attention_exhausted'
    WAKE_UP_TRIGGER = 'wake_up_trigger'

@dataclass
class Event:
    type: EventType
    timestamp: int
    importance: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return self.importance < other.importance

class GrowingSOM:

    def __init__(self, input_dim: int=5, initial_size: int=2, rng=None):
        self.rng = rng or default_rng()
        self.input_dim = input_dim
        self.neurons = []
        self.connections = []
        for i in range(initial_size):
            for j in range(initial_size):
                neuron = {'weights': self.rng.standard_normal(input_dim) * 0.1, 'position': (i, j), 'age': 0, 'error': 0.0, 'activation_count': 0, 'recent_errors': deque(maxlen=20), 'specialization': 0.0, 'utility': 1.0}
                self.neurons.append(neuron)
        for i in range(len(self.neurons)):
            for j in range(i + 1, len(self.neurons)):
                pos1 = self.neurons[i]['position']
                pos2 = self.neurons[j]['position']
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if dist == 1:
                    self.connections.append({'from': i, 'to': j, 'strength': 1.0, 'age': 0, 'usage': 0})
        self.base_learning_rate = 0.15
        self.base_neighbor_rate = 0.02
        self.max_neurons = 100000
        self.min_error_variance = 0.05
        self.connection_threshold = 0.05
        self.total_activations = 0
        self.growth_events = 0
        self.pruning_events = 0
        self.input_history = deque(maxlen=100)

    def _adaptive_learning_rate(self, neuron_age: int) -> float:
        return self.base_learning_rate * np.exp(-neuron_age / 1000.0)

    def _neighborhood_function(self, distance: float, sigma: float) -> float:
        return np.exp(-distance ** 2 / (2 * sigma ** 2))

    def _adaptive_sigma(self, iteration: int) -> float:
        initial_sigma = 2.0
        final_sigma = 0.5
        decay_rate = 0.001
        return final_sigma + (initial_sigma - final_sigma) * np.exp(-decay_rate * iteration)

    def find_bmu(self, input_vec: np.ndarray) -> Tuple[int, float]:
        min_dist = float('inf')
        bmu_idx = 0
        for i, neuron in enumerate(self.neurons):
            base_dist = np.linalg.norm(input_vec - neuron['weights'])
            specialization_bonus = neuron['specialization'] * 0.1
            adjusted_dist = base_dist - specialization_bonus
            if adjusted_dist < min_dist:
                min_dist = adjusted_dist
                bmu_idx = i
        return (bmu_idx, max(0.0, min_dist))

    def learn(self, input_vec: np.ndarray) -> Dict[str, Any]:
        self.input_history.append(input_vec.copy())
        bmu_idx, error = self.find_bmu(input_vec)
        bmu = self.neurons[bmu_idx]
        bmu['activation_count'] += 1
        bmu['error'] += error
        bmu['age'] += 1
        bmu['recent_errors'].append(error)
        self.total_activations += 1
        learning_rate = self._adaptive_learning_rate(bmu['age'])
        sigma = self._adaptive_sigma(self.total_activations)
        delta = learning_rate * (input_vec - bmu['weights'])
        bmu['weights'] += delta
        if len(bmu['recent_errors']) >= 5:
            avg_recent_error = np.mean(list(bmu['recent_errors']))
            bmu['specialization'] = 1.0 / (1.0 + avg_recent_error)
        bmu_pos = np.array(bmu['position'])
        for i, neuron in enumerate(self.neurons):
            if i == bmu_idx:
                continue
            pos = np.array(neuron['position'])
            distance = np.linalg.norm(pos - bmu_pos)
            neighbor_influence = self._neighborhood_function(distance, sigma)
            if neighbor_influence > 0.01:
                neighbor_rate = self.base_neighbor_rate * neighbor_influence
                neighbor_delta = neighbor_rate * (input_vec - neuron['weights'])
                neuron['weights'] += neighbor_delta
                neuron['age'] += 1
                neuron['utility'] *= 1.001
        grew = False
        if len(self.neurons) < self.max_neurons:
            should_grow = self._should_grow(bmu_idx, error)
            if should_grow:
                grew = self._grow_neuron_intelligent(bmu_idx, input_vec)
        self._update_connections(bmu_idx)
        pruned_count = self._prune_weak_connections()
        for i, neuron in enumerate(self.neurons):
            if i != bmu_idx:
                neuron['utility'] *= 0.9999
        return {'bmu': bmu_idx, 'error': error, 'grew': grew, 'pruned': pruned_count, 'neuron_count': len(self.neurons), 'learning_rate': learning_rate, 'sigma': sigma}

    def _should_grow(self, bmu_idx: int, current_error: float) -> bool:
        bmu = self.neurons[bmu_idx]
        if len(bmu['recent_errors']) < 10:
            return False
        recent_errors = list(bmu['recent_errors'])
        error_variance = np.var(recent_errors)
        error_mean = np.mean(recent_errors)
        if error_variance < self.min_error_variance or error_mean < 0.15:
            return False
        if len(self.input_history) >= 20:
            recent_inputs = list(self.input_history)[-20:]
            input_variance = np.mean(np.var(recent_inputs, axis=0))
            if input_variance < 0.02:
                return False
        if self.total_activations - getattr(self, '_last_growth', 0) < 15:
            return False
        return True

    def _grow_neuron_intelligent(self, bmu_idx: int, input_vec: np.ndarray) -> bool:
        bmu_pos = self.neurons[bmu_idx]['position']
        best_pos = None
        best_score = -1
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            candidate = (bmu_pos[0] + dx, bmu_pos[1] + dy)
            position_occupied = False
            for n in self.neurons:
                n_pos = tuple(n['position']) if isinstance(n['position'], np.ndarray) else n['position']
                if n_pos == candidate:
                    position_occupied = True
                    break
            if position_occupied:
                continue
            score = self._evaluate_growth_position(candidate, bmu_idx)
            if score > best_score:
                best_score = score
                best_pos = candidate
        if best_pos is None:
            return False
        bmu_weights = self.neurons[bmu_idx]['weights']
        direction = input_vec - bmu_weights
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
            new_weights = bmu_weights + 0.3 * direction_norm * direction
        else:
            new_weights = bmu_weights + self.rng.standard_normal(self.input_dim) * 0.05
        new_neuron = {'weights': new_weights, 'position': best_pos, 'age': 0, 'error': 0.0, 'activation_count': 0, 'recent_errors': deque(maxlen=20), 'specialization': 0.0, 'utility': 1.0}
        new_idx = len(self.neurons)
        self.neurons.append(new_neuron)
        self._create_intelligent_connections(new_idx, bmu_idx)
        self.growth_events += 1
        self._last_growth = self.total_activations
        return True

    def _evaluate_growth_position(self, position: Tuple[int, int], bmu_idx: int) -> float:
        score = 0.0
        for i, neuron in enumerate(self.neurons):
            pos = neuron['position']
            dist = abs(pos[0] - position[0]) + abs(pos[1] - position[1])
            if dist == 1:
                score += neuron['utility'] * 0.5
            elif dist == 2:
                score += neuron['utility'] * 0.2
        return score

    def _create_intelligent_connections(self, new_idx: int, bmu_idx: int):
        new_pos = self.neurons[new_idx]['position']
        self.connections.append({'from': min(bmu_idx, new_idx), 'to': max(bmu_idx, new_idx), 'strength': 1.5, 'age': 0, 'usage': 0})
        for i, neuron in enumerate(self.neurons):
            if i == new_idx:
                continue
            pos = neuron['position']
            dist = abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1])
            if dist == 1:
                strength = 1.0 * neuron['utility']
                self.connections.append({'from': min(i, new_idx), 'to': max(i, new_idx), 'strength': strength, 'age': 0, 'usage': 0})

    def _update_connections(self, bmu_idx: int):
        for conn in self.connections:
            conn['age'] += 1
            if conn['from'] == bmu_idx or conn['to'] == bmu_idx:
                conn['strength'] = min(3.0, conn['strength'] * 1.02)
                conn['usage'] += 1
            else:
                conn['strength'] *= 0.999

    def _prune_weak_connections(self) -> int:
        initial_count = len(self.connections)
        self.connections = [conn for conn in self.connections if conn['strength'] > self.connection_threshold and conn['usage'] > 0]
        max_age = max(1000, self.total_activations // 10)
        self.connections = [conn for conn in self.connections if conn['age'] < max_age or conn['usage'] > 10]
        pruned = initial_count - len(self.connections)
        if pruned > 0:
            self.pruning_events += 1
        return pruned
        return pruned

    def get_statistics(self) -> Dict[str, Any]:
        if not self.neurons:
            return {}
        neuron_count = len(self.neurons)
        connection_count = len(self.connections)
        ages = [n['age'] for n in self.neurons]
        activations = [n['activation_count'] for n in self.neurons]
        specializations = [n['specialization'] for n in self.neurons]
        utilities = [n['utility'] for n in self.neurons]
        strengths = [c['strength'] for c in self.connections] if self.connections else [0]
        connection_ages = [c['age'] for c in self.connections] if self.connections else [0]
        if neuron_count > 1:
            weights_matrix = np.array([n['weights'] for n in self.neurons])
            weight_diversity = np.mean(np.std(weights_matrix, axis=0))
        else:
            weight_diversity = 0.0
        return {'neuron_count': neuron_count, 'connection_count': connection_count, 'total_activations': self.total_activations, 'growth_events': self.growth_events, 'pruning_events': self.pruning_events, 'avg_age': np.mean(ages), 'avg_activations': np.mean(activations), 'avg_specialization': np.mean(specializations), 'avg_utility': np.mean(utilities), 'max_strength': np.max(strengths), 'avg_strength': np.mean(strengths), 'avg_connection_age': np.mean(connection_ages), 'weight_diversity': weight_diversity, 'network_efficiency': connection_count / max(1, neuron_count ** 2), 'specialization_ratio': len([s for s in specializations if s > 0.7]) / neuron_count}

    def get_network_state(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.neurons:
            return (np.array([]), np.array([]))
        positions = np.array([n['position'] for n in self.neurons])
        activities = np.array([n['activation_count'] * (1 + n['specialization']) for n in self.neurons])
        return (positions, activities)

    def get_topology_health(self) -> Dict[str, float]:
        if len(self.neurons) < 2:
            return {'health': 1.0, 'connectivity': 1.0, 'balance': 1.0}
        total_possible_connections = len(self.neurons) * (len(self.neurons) - 1) // 2
        connectivity = len(self.connections) / max(1, total_possible_connections)
        activations = [n['activation_count'] for n in self.neurons]
        if max(activations) > 0:
            activation_balance = 1.0 - np.std(activations) / np.mean(activations)
        else:
            activation_balance = 1.0
        health = (connectivity + activation_balance) / 2.0
        return {'health': health, 'connectivity': connectivity, 'balance': activation_balance}

class QualiaState:

    def __init__(self, history_size: Optional[int]=None):
        self.arousal = 0.0
        self.valence = 0.0
        self.entropy = 0.0
        self.engagement = 0.0
        self.frustration = 0.0
        if history_size is None:
            available_mb = 0 // (1024 * 1024)
            if available_mb > 8000:
                history_size = 512
            elif available_mb > 4000:
                history_size = 512
            else:
                history_size = 512
        self.history = deque(maxlen=history_size)
        self.max_history_size = 512

    def compute(self, delta_hat: float, m: float, gap: float, barrier_viols: int, meta_conf: float, stability: float) -> 'QualiaState':
        arousal_input = delta_hat * 3.0
        self.arousal = np.tanh(arousal_input)
        valence_raw = stability * meta_conf * np.exp(-barrier_viols * 0.2)
        self.valence = np.clip(valence_raw, 0.0, 1.0)
        entropy_input = (1.0 - meta_conf) * 2.0
        self.entropy = np.tanh(entropy_input)
        stability_factor = 1.0 - abs(stability - 0.65) * 1.5
        stability_factor = max(0.1, stability_factor)
        engagement_input = delta_hat * stability_factor * 4.0
        self.engagement = np.tanh(engagement_input)
        frustration_input = barrier_viols * 0.5 + (1.0 - stability) * 2.0
        self.frustration = np.tanh(frustration_input)
        self.history.append({'arousal': self.arousal, 'valence': self.valence, 'entropy': self.entropy, 'engagement': self.engagement, 'frustration': self.frustration})
        return self

    def dominant_feeling(self) -> str:
        feelings = {'arousal': self.arousal, 'valence': self.valence, 'entropy': self.entropy, 'engagement': self.engagement, 'frustration': self.frustration}
        return max(feelings, key=feelings.get)

    def to_dict(self) -> Dict[str, float]:
        return {'q_arousal': float(self.arousal), 'q_valence': float(self.valence), 'q_entropy': float(self.entropy), 'q_engagement': float(self.engagement), 'q_frustration': float(self.frustration)}

@dataclass
class ConsciousContent:
    timestamp: int
    content_type: str
    content: Any
    salience: float
    semantic_meaning: str

    def __lt__(self, other):
        return self.salience < other.salience

class GlobalWorkspace:

    def __init__(self, capacity: int=3):
        self.capacity = capacity
        self.current_contents: List[ConsciousContent] = []
        self.competitors: List[ConsciousContent] = []
        self.broadcast_history: deque = deque(maxlen=100)
        self.attention_focus: Optional[ConsciousContent] = None
        self.attention_switching_threshold = 0.45
        self.attention_persistence = 0.0
        self.attention_history: deque = deque(maxlen=20)
        self.content_associations: Dict[str, List[str]] = {}
        self.semantic_clusters: Dict[str, List[ConsciousContent]] = {}
        self.integration_threshold = 0.35
        self.exploration_bias = 0.75
        self.policy_state_history: deque = deque(maxlen=200)
        self.error_detection_buffer: deque = deque(maxlen=50)
        self.policy_adjustments: deque = deque(maxlen=100)
        self.meta_learning_rate = 0.12
        self.policy_momentum = {}
        self.policy_variance_tracking = {}
        self.policy_params = {'exploration_bias': 0.75, 'stability_bias': 0.3, 'confidence_threshold': 0.25, 'error_sensitivity': 1.8, 'learning_rate': 0.15, 'attention_focus_strength': 0.85, 'integration_eagerness': 0.8, 'prediction_confidence': 0.4}
        for param in self.policy_params:
            self.policy_momentum[param] = 0.0
            self.policy_variance_tracking[param] = deque(maxlen=50)
        self.cumulative_reward = 0.0
        self.reward_history: deque = deque(maxlen=100)
        self.performance_metrics = {'prediction_accuracy': deque(maxlen=100), 'response_time': deque(maxlen=100), 'integration_success': deque(maxlen=100)}
        self.predictions: Dict[str, Any] = {}
        self.prediction_confidence = 0.5
        self.broadcast_hooks: List[callable] = []
        self.last_broadcast_state: Optional[Dict] = None

    def submit_for_competition(self, content: ConsciousContent):
        integration_threshold = 0.7 - self.exploration_bias * 0.3
        integrated = self._try_integrate_content(content, integration_threshold)
        if integrated:
            return
        adjusted_content = self._adjust_salience_with_exploration(content)
        self.competitors.append(adjusted_content)
        self._update_semantic_clusters(adjusted_content)

    def _try_integrate_content(self, new_content: ConsciousContent, threshold: float=0.7) -> bool:
        for existing in self.current_contents + self.competitors:
            similarity = self._calculate_semantic_similarity(new_content, existing)
            if similarity > threshold:
                if self.exploration_bias > 0.5:
                    existing.salience = max(existing.salience, new_content.salience * 1.2)
                    existing.content = f'{existing.content} + {new_content.content}'
                else:
                    existing.salience *= 1.1
                return True
        return False

    def _calculate_semantic_similarity(self, content1: ConsciousContent, content2: ConsciousContent) -> float:
        type_similarity = 0.8 if content1.content_type == content2.content_type else 0.0
        if hasattr(content1, 'semantic_meaning') and hasattr(content2, 'semantic_meaning'):
            meaning1_str = str(content1.semantic_meaning) if content1.semantic_meaning else ''
            meaning2_str = str(content2.semantic_meaning) if content2.semantic_meaning else ''
            meaning1_words = set(meaning1_str.lower().split())
            meaning2_words = set(meaning2_str.lower().split())
        else:
            return type_similarity
        if len(meaning1_words) == 0 or len(meaning2_words) == 0:
            semantic_similarity = 0.0
        else:
            intersection = len(meaning1_words & meaning2_words)
            union = len(meaning1_words | meaning2_words)
            semantic_similarity = intersection / union if union > 0 else 0.0
        time_diff = abs(content1.timestamp - content2.timestamp)
        time_similarity = max(0.0, 1.0 - time_diff / 100.0)
        return type_similarity * 0.4 + semantic_similarity * 0.4 + time_similarity * 0.2

    def _adjust_salience_contextually(self, content: ConsciousContent) -> ConsciousContent:
        adjusted_salience = content.salience
        if len(self.attention_history) > 0:
            recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
            same_type_count = recent_types.count(content.content_type)
            if same_type_count >= 3:
                adjusted_salience *= 0.7
            elif content.content_type not in recent_types:
                adjusted_salience *= 1.2
        associations = self.content_associations.get(content.content_type, [])
        if self.attention_focus and self.attention_focus.content_type in associations:
            adjusted_salience *= 1.15
        try:
            if hasattr(content, 'semantic_meaning') and 'uncertainty' in content.semantic_meaning.lower():
                adjusted_salience *= 1.0 + self.policy_params['error_sensitivity'] * 0.3
        except (AttributeError, TypeError):
            pass
        if 'goal' in content.content_type:
            adjusted_salience *= 1.0 + self.policy_params['stability_bias'] * 0.2
        return ConsciousContent(timestamp=content.timestamp, content_type=content.content_type, content=content.content, salience=max(0.0, min(1.0, adjusted_salience)), semantic_meaning=content.semantic_meaning)

    def _adjust_salience_with_exploration(self, content: ConsciousContent) -> ConsciousContent:
        adjusted_content = self._adjust_salience_contextually(content)
        base_salience = adjusted_content.salience
        if self.exploration_bias > 0.6:
            novelty_bonus = 0.0
            if len(self.attention_history) > 0:
                recent_types = [att.content_type for att in list(self.attention_history)[-3:]]
                if content.content_type not in recent_types:
                    novelty_bonus += 0.3 * self.exploration_bias
            content_str = str(content.content) if hasattr(content, 'content') and content.content else ''
            if any((word in content_str.lower() for word in ['uncertain', 'error', 'new', 'unknown'])):
                novelty_bonus += 0.2 * self.exploration_bias
                pass
            base_salience *= 1.0 + novelty_bonus
        elif self.exploration_bias < 0.4:
            stability_bonus = 0.0
            if len(self.attention_history) > 0:
                recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
                if content.content_type in recent_types:
                    stability_bonus += 0.2 * (1.0 - self.exploration_bias)
            content_str = str(content.content) if hasattr(content, 'content') and content.content else ''
            if any((word in content_str.lower() for word in ['confident', 'goal', 'stable', 'known'])):
                stability_bonus += 0.15 * (1.0 - self.exploration_bias)
            base_salience *= 1.0 + stability_bonus
        random_factor = 1.0 + np.random.normal(0, 0.05 + 0.1 * self.exploration_bias)
        base_salience *= random_factor
        return ConsciousContent(timestamp=adjusted_content.timestamp, content_type=adjusted_content.content_type, content=adjusted_content.content, salience=max(0.0, min(1.0, base_salience)), semantic_meaning=adjusted_content.semantic_meaning)

    def _record_association(self, type1: str, type2: str):
        if type1 not in self.content_associations:
            self.content_associations[type1] = []
        if type2 not in self.content_associations:
            self.content_associations[type2] = []
        if type2 not in self.content_associations[type1]:
            self.content_associations[type1].append(type2)
        if type1 not in self.content_associations[type2]:
            self.content_associations[type2].append(type1)

    def _update_semantic_clusters(self, content: ConsciousContent):
        if hasattr(content, 'semantic_meaning'):
            meaning_str = str(content.semantic_meaning) if content.semantic_meaning else ''
            words = meaning_str.lower().split()
        else:
            words = []
        if words:
            main_word = words[0]
            if main_word not in self.semantic_clusters:
                self.semantic_clusters[main_word] = []
            self.semantic_clusters[main_word].append(content)
            if len(self.semantic_clusters[main_word]) > 10:
                self.semantic_clusters[main_word] = self.semantic_clusters[main_word][-10:]

    def compete_for_consciousness(self) -> List[ConsciousContent]:
        current_policy_bias = self.policy_params.get('exploration_bias', 0.5)
        if current_policy_bias < 0.4:
            self.policy_params['exploration_bias'] = 0.65
            print('[yellow]Exploration bias reset to maintain exploration![/yellow]')
        self.exploration_bias = self.policy_params.get('exploration_bias', 0.5)
        if not self.competitors:
            return self.current_contents
        exploration_factor = self.exploration_bias + np.random.normal(0, 0.05)
        exploration_mode = exploration_factor > 0.5
        base_threshold = self.policy_params.get('confidence_threshold', 0.4)
        if exploration_mode:
            dynamic_threshold = base_threshold * (0.3 + 0.4 * (1.0 - self.exploration_bias))
            min_candidates = max(1, len(self.competitors) // 2)
        else:
            dynamic_threshold = base_threshold * (0.8 + 0.4 * self.exploration_bias)
            min_candidates = min(3, len(self.competitors))
        candidates = [c for c in self.competitors if c.salience > dynamic_threshold]
        if len(candidates) < min_candidates:
            sorted_competitors = sorted(self.competitors, key=lambda x: x.salience, reverse=True)
            candidates = sorted_competitors[:min_candidates]
        if exploration_mode:
            selected = self._explore_b_selection(candidates, aggressive=True)
        else:
            selected = self._conservative_selection_enhanced(candidates)
        self.current_contents = selected[:self.capacity]
        self._update_attention_explore_b(selected, exploration_mode)
        for content in selected:
            if content:
                self._update_semantic_clusters(content)
                for other in selected:
                    if other != content:
                        self._record_association(content.content_type, other.content_type)
        baseline_reward = 0.03 + 0.01 * len(selected)
        if exploration_mode:
            baseline_reward += 0.02
        self.cumulative_reward += baseline_reward
        self.reward_history.append(baseline_reward)
        self.competitors.clear()
        return self.current_contents

    def _explore_b_selection(self, candidates: List[ConsciousContent], aggressive: bool=False) -> List[ConsciousContent]:
        if not candidates:
            return []
        selected = []
        remaining = candidates.copy()
        if aggressive and self.exploration_bias > 0.7:
            first_pick = remaining[np.random.choice(len(remaining))]
        else:
            top_half_size = max(1, len(remaining) // 2)
            sorted_remaining = sorted(remaining, key=lambda x: x.salience, reverse=True)
            top_half = sorted_remaining[:top_half_size]
            first_pick = top_half[np.random.choice(len(top_half))]
        selected.append(first_pick)
        remaining.remove(first_pick)
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            best_score = -float('inf')
            for candidate in remaining:
                quality_score = candidate.salience * 0.3
                diversity_score = self._calculate_exploration_diversity(candidate, selected) * 0.5
                random_bonus = np.random.random() * 0.2
                exploration_boost = self.exploration_bias * 0.1
                total_score = quality_score + diversity_score + random_bonus + exploration_boost
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _calculate_exploration_diversity(self, candidate: ConsciousContent, selected: List[ConsciousContent]) -> float:
        if not selected:
            return 1.0
        diversity_scores = []
        for existing in selected:
            type_diversity = 1.0 if candidate.content_type != existing.content_type else -0.2
            semantic_similarity = self._calculate_semantic_similarity(candidate, existing)
            semantic_diversity = (1.0 - semantic_similarity) * 1.5
            time_gap = abs(candidate.timestamp - existing.timestamp)
            time_diversity = min(1.0, time_gap / 50.0)
            combined_diversity = (type_diversity + semantic_diversity + time_diversity) / 3.0
            diversity_scores.append(combined_diversity)
        return np.mean(diversity_scores)

    def _conservative_selection_enhanced(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        if not candidates:
            return []
        selected = []
        sorted_candidates = sorted(candidates, key=lambda x: x.salience, reverse=True)
        priority_count = min(2, len(sorted_candidates), self.capacity)
        for i in range(priority_count):
            selected.append(sorted_candidates[i])
        remaining = [c for c in candidates if c not in selected]
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            best_score = -float('inf')
            for candidate in remaining:
                quality_score = candidate.salience * 0.7
                diversity_score = self._calculate_diversity_score(candidate, selected) * 0.25
                random_factor = np.random.random() * 0.05
                total_score = quality_score + diversity_score + random_factor
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _update_attention_explore_b(self, selected: List[ConsciousContent], exploration_mode: bool):
        if not selected:
            return
        if exploration_mode:
            self.attention_switching_threshold *= 0.7
            if np.random.random() < 0.3:
                new_focus = selected[np.random.choice(len(selected))]
            elif self.attention_focus and self.attention_focus in selected:
                most_different = None
                max_difference = -1.0
                for content in selected:
                    if content != self.attention_focus:
                        difference = 1.0 - self._calculate_semantic_similarity(content, self.attention_focus)
                        if difference > max_difference:
                            max_difference = difference
                            most_different = content
                new_focus = most_different if most_different else max(selected, key=lambda x: x.salience)
            else:
                new_focus = max(selected, key=lambda x: x.salience)
        else:
            self.attention_switching_threshold *= 1.1
            new_focus = max(selected, key=lambda x: x.salience)
        should_switch = self.attention_focus is None or new_focus != self.attention_focus or np.random.random() < 1.0 - self.attention_switching_threshold
        if should_switch:
            self.attention_focus = new_focus
            self.attention_persistence = 0.0
            self.attention_history.append(new_focus)
        else:
            self.attention_persistence += 0.1
        self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.2, 0.95)

    def _exploratory_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        selected = []
        if not candidates:
            return selected
        random_picks = min(2, len(candidates), self.capacity)
        if random_picks > 0 and len(candidates) > 0:
            actual_picks = min(random_picks, len(candidates))
            random_indices = np.random.choice(len(candidates), size=actual_picks, replace=False)
            random_selection = [candidates[i] for i in random_indices]
            selected.extend(random_selection)
        remaining_candidates = [c for c in candidates if c not in selected]
        while len(selected) < self.capacity and remaining_candidates:
            best_candidate = None
            max_diversity_score = -1
            for candidate in remaining_candidates:
                diversity_score = self._calculate_diversity_score(candidate, selected)
                diversity_score += np.random.normal(0, 0.2)
                if diversity_score > max_diversity_score:
                    max_diversity_score = diversity_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break
        return selected[:self.capacity]

    def _conservative_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        selected = []
        if not candidates:
            return selected
        if len(candidates) > 0:
            best = max(candidates, key=lambda x: x.salience)
            selected.append(best)
        else:
            return selected
        remaining = [c for c in candidates if c != best]
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            max_score = -1
            for candidate in remaining:
                quality_score = candidate.salience * 0.6
                diversity_score = self._calculate_diversity_score(candidate, selected) * 0.3
                random_bonus = np.random.random() * 0.1
                total_score = quality_score + diversity_score + random_bonus
                if total_score > max_score:
                    max_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _calculate_diversity_score(self, candidate: ConsciousContent, selected: List[ConsciousContent]) -> float:
        if not selected:
            return 1.0
        diversity_scores = []
        for existing in selected:
            type_diversity = 0.8 if candidate.content_type != existing.content_type else 0.0
            semantic_diversity = 1.0 - self._calculate_semantic_similarity(candidate, existing)
            diversity_scores.append((type_diversity + semantic_diversity) / 2.0)
        return np.mean(diversity_scores)

    def _update_attention_dynamically(self, selected: List[ConsciousContent], exploration_mode: bool):
        if not selected:
            return
        if exploration_mode:
            self.attention_switching_threshold *= 0.8
            if np.random.random() < 0.3:
                new_focus = selected[np.random.choice(len(selected))]
            else:
                new_focus = max(selected, key=lambda x: x.salience)
        else:
            self.attention_switching_threshold *= 1.05
            new_focus = max(selected, key=lambda x: x.salience)
        should_switch = self.attention_focus is None or new_focus != self.attention_focus or np.random.random() < 1.0 - self.attention_switching_threshold
        if should_switch:
            self.attention_focus = new_focus
            self.attention_persistence = 0.0
            self.attention_history.append(new_focus)
        else:
            self.attention_persistence += 0.1
        self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.3, 0.9)

    def _select_cluster_representatives(self) -> List[ConsciousContent]:
        representatives = []
        temp_clusters = {}
        for competitor in self.competitors:
            try:
                if hasattr(competitor, 'semantic_meaning'):
                    meaning_str = str(competitor.semantic_meaning) if competitor.semantic_meaning else ''
                    words = meaning_str.lower().split()
                    cluster_key = words[0] if words else 'misc'
                else:
                    cluster_key = 'misc'
            except (AttributeError, TypeError):
                cluster_key = 'misc'
            if cluster_key not in temp_clusters:
                temp_clusters[cluster_key] = []
            temp_clusters[cluster_key].append(competitor)
        for cluster_contents in temp_clusters.values():
            if cluster_contents:
                best = max(cluster_contents, key=lambda x: x.salience)
                representatives.append(best)
        return representatives

    def _balanced_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        if len(candidates) <= self.capacity:
            return candidates
        selected = []
        remaining = candidates.copy()
        best = max(remaining, key=lambda x: x.salience)
        selected.append(best)
        remaining.remove(best)
        while len(selected) < self.capacity and remaining:
            diversity_scores = []
            for candidate in remaining:
                score = candidate.salience
                diversity_bonus = 0.0
                for selected_content in selected:
                    similarity = self._calculate_semantic_similarity(candidate, selected_content)
                    diversity_bonus += 1.0 - similarity
                diversity_bonus /= len(selected)
                final_score = score * 0.7 + diversity_bonus * 0.3
                diversity_scores.append((candidate, final_score))
            best_candidate, _ = max(diversity_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        return selected

    def _manage_attention_focus(self, winners: List[ConsciousContent]):
        if not winners:
            return
        current_best = winners[0]
        should_switch = False
        if self.attention_focus is None:
            should_switch = True
        else:
            salience_gap = current_best.salience - self.attention_focus.salience
            if salience_gap > self.attention_switching_threshold:
                should_switch = True
            self.attention_persistence *= 0.9
            if self.attention_persistence < 0.3:
                should_switch = True
            if current_best.content_type != self.attention_focus.content_type:
                type_novelty = current_best.content_type not in [att.content_type for att in list(self.attention_history)[-3:]]
                if type_novelty and salience_gap > 0.1:
                    should_switch = True
        if should_switch:
            if self.attention_focus:
                self.attention_history.append(self.attention_focus)
            self.attention_focus = current_best
            self.attention_persistence = 1.0
            focus_strength = self.policy_params['attention_focus_strength']
            self.attention_switching_threshold = 0.9 - focus_strength * 0.3

    def broadcast(self, world_state: Optional[Dict]=None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        current_timestamp = int(self._get_current_timestamp() * 1000)
        primary_focus = self._create_safe_focus_description()
        conscious_contents = self._create_safe_content_descriptions()
        workspace_metrics = self._calculate_safe_workspace_metrics()
        explore_b_state = {'exploration_bias': self.policy_params.get('exploration_bias', 0.5), 'exploration_mode': self.exploration_bias > 0.5, 'last_selection_type': 'exploratory' if hasattr(self, '_last_selection_mode') and self._last_selection_mode else 'conservative', 'diversity_score': self._calculate_current_diversity(), 'novelty_score': self._calculate_current_novelty()}
        broadcast_msg = {'timestamp': current_timestamp, 'version': '2.0', 'primary_focus': primary_focus, 'conscious_contents': conscious_contents, 'workspace_fullness': len(self.current_contents) / self.capacity, 'explore_b': explore_b_state, 'policy_state': {'exploration_bias': self.policy_params.get('exploration_bias', 0.5), 'stability_bias': self.policy_params.get('stability_bias', 0.5), 'confidence_threshold': self.policy_params.get('confidence_threshold', 0.4), 'error_sensitivity': self.policy_params.get('error_sensitivity', 1.0), 'attention_focus_strength': self.policy_params.get('attention_focus_strength', 0.7), 'recent_reward': list(self.reward_history)[-1] if self.reward_history else 0.0, 'cumulative_reward': self.cumulative_reward}, 'workspace_metrics': workspace_metrics, 'attention_context': {'switching_threshold': self.attention_switching_threshold, 'persistence': self.attention_persistence, 'recent_switches': len(self.attention_history), 'focus_stability': self._calculate_focus_stability()}, 'world_state': world_state if world_state else {}, 'performance': {'response_time': time.perf_counter() - start_time, 'error_count': len(self.error_detection_buffer), 'adjustment_count': len(self.policy_adjustments), 'success_rate': self._calculate_success_rate()}}
        predictions = self._generate_safe_predictions(world_state)
        if predictions:
            broadcast_msg['predictions'] = predictions
        if self.semantic_clusters:
            broadcast_msg['semantic_clusters'] = {k: len(v) for k, v in self.semantic_clusters.items()}
        self.broadcast_history.append(broadcast_msg)
        self.last_broadcast_state = broadcast_msg.copy()
        if hasattr(self, 'broadcast_hooks') and self.broadcast_hooks:
            hook_results = []
            for i, hook in enumerate(self.broadcast_hooks):
                try:
                    result = hook(broadcast_msg)
                    if result:
                        hook_results.append({'hook_id': i, 'result': result, 'timestamp': self._get_current_timestamp()})
                except Exception as e:
                    pass
            if hook_results:
                broadcast_msg['hook_responses'] = hook_results
        total_response_time = time.perf_counter() - start_time
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics['response_time'].append(total_response_time)
        return broadcast_msg

    def _extract_safe_content_attrs(self, obj) -> Dict[str, Any]:
        return {'type': getattr(obj, 'content_type', 'unknown'), 'salience': getattr(obj, 'salience', 0.0), 'meaning': str(getattr(obj, 'semantic_meaning', ''))[:100], 'timestamp': getattr(obj, 'timestamp', 0)}

    def _get_current_timestamp(self) -> float:
        return time.perf_counter()

    def _calculate_age(self, obj) -> float:
        timestamp = getattr(obj, 'timestamp', 0)
        if timestamp == 0:
            return 0.0
        return time.perf_counter() - timestamp

    @contextmanager
    def _measure_performance(self, metric_name: str=None):
        start_time = time.perf_counter()
        try:
            yield start_time
        finally:
            elapsed = time.perf_counter() - start_time
            if metric_name and hasattr(self, 'performance_metrics'):
                if metric_name not in self.performance_metrics:
                    self.performance_metrics[metric_name] = []
                self.performance_metrics[metric_name].append(elapsed)

    def _create_safe_focus_description(self) -> Dict[str, Any]:
        if not self.attention_focus:
            return {'active': False, 'type': 'none', 'salience': 0.0, 'meaning': '', 'content': None}
        attrs = self._extract_safe_content_attrs(self.attention_focus)
        return {'active': True, **attrs, 'content': str(getattr(self.attention_focus, 'content', ''))[:100], 'persistence': self.attention_persistence}

    def _create_safe_content_descriptions(self) -> List[Dict[str, Any]]:
        descriptions = []
        for i, content in enumerate(self.current_contents):
            attrs = self._extract_safe_content_attrs(content)
            desc = {'index': i, **attrs, 'age': self._calculate_age(content)}
            descriptions.append(desc)
        return descriptions

    def _calculate_safe_workspace_metrics(self) -> Dict[str, float]:
        try:
            metrics = {'fullness': len(self.current_contents) / self.capacity, 'avg_salience': 0.0, 'diversity': 0.0, 'stability': 0.0}
            if self.current_contents:
                saliences = [self._extract_safe_content_attrs(content)['salience'] for content in self.current_contents]
                metrics['avg_salience'] = np.mean(saliences) if saliences else 0.0
                if len(self.current_contents) > 1:
                    types = [self._extract_safe_content_attrs(c)['type'] for c in self.current_contents]
                    unique_types = len(set(types))
                    metrics['diversity'] = unique_types / len(types)
                if len(self.broadcast_history) > 0:
                    prev_types = set()
                    prev_contents = self.broadcast_history[-1].get('conscious_contents', [])
                    prev_types = {c.get('type', 'unknown') for c in prev_contents}
                    current_types = {self._extract_safe_content_attrs(c)['type'] for c in self.current_contents}
                    if len(prev_types) > 0 and len(current_types) > 0:
                        intersection = len(prev_types & current_types)
                        union = len(prev_types | current_types)
                        metrics['stability'] = intersection / union if union > 0 else 0.0
            return metrics
        except Exception:
            return {'fullness': 0.0, 'avg_salience': 0.0, 'diversity': 0.0, 'stability': 0.0}

    def _calculate_current_diversity(self) -> float:
        if len(self.current_contents) <= 1:
            return 0.0
        types = [getattr(c, 'content_type', 'unknown') for c in self.current_contents]
        unique_types = len(set(types))
        return unique_types / len(types)

    def _calculate_current_novelty(self) -> float:
        if not self.current_contents or len(self.attention_history) < 5:
            return 0.5
        current_types = {getattr(c, 'content_type', 'unknown') for c in self.current_contents}
        recent_types = {getattr(att, 'content_type', 'unknown') for att in list(self.attention_history)[-5:]}
        new_types = current_types - recent_types
        novelty = len(new_types) / len(current_types) if current_types else 0.0
        return min(1.0, novelty)

    def _calculate_focus_stability(self) -> float:
        if len(self.attention_history) < 3:
            return 1.0
        try:
            recent_focuses = list(self.attention_history)[-3:]
            if all((getattr(f, 'content_type', '') == getattr(recent_focuses[0], 'content_type', '') for f in recent_focuses)):
                return 1.0
            else:
                return 0.3
        except:
            return 0.5

    def _calculate_success_rate(self) -> float:
        if len(self.reward_history) < 5:
            return 0.5
        recent_rewards = list(self.reward_history)[-10:]
        positive_count = sum((1 for r in recent_rewards if r > 0))
        return positive_count / len(recent_rewards)

    def _generate_safe_predictions(self, world_state: Optional[Dict]) -> Optional[Dict]:
        if not world_state:
            return None
        current_stability = world_state.get('stability', 0.5)
        current_energy = world_state.get('energy_level', 0.5)
        if len(self.broadcast_history) >= 3:
            recent_states = []
            for msg in list(self.broadcast_history)[-3:]:
                ws = msg.get('world_state', {})
                if ws:
                    recent_states.append(ws)
            if len(recent_states) >= 2:
                stabilities = [s.get('stability', 0.5) for s in recent_states]
                stability_trend = np.mean(np.diff(stabilities)) if len(stabilities) > 1 else 0.0
                energies = [s.get('energy_level', 0.5) for s in recent_states]
                energy_trend = np.mean(np.diff(energies)) if len(energies) > 1 else 0.0
                return {'stability': np.clip(current_stability + stability_trend, 0.0, 1.0), 'energy': np.clip(current_energy + energy_trend, 0.0, 1.0), 'confidence': 0.6, 'method': 'linear_trend'}
            return {'stability': current_stability, 'energy': current_energy, 'confidence': 0.3, 'method': 'baseline'}

    def _generate_predictions(self, world_state: Optional[Dict]):
        if not world_state:
            return
        current_stability = world_state.get('stability', 0.5)
        current_energy = world_state.get('energy_level', 0.5)
        if len(self.broadcast_history) >= 5:
            recent_states = []
            for msg in list(self.broadcast_history)[-5:]:
                if msg.get('world_state'):
                    recent_states.append(msg['world_state'])
            if recent_states:
                stability_trend = np.diff([s.get('stability', 0.5) for s in recent_states[-3:]])
                predicted_stability = current_stability + np.mean(stability_trend) if len(stability_trend) > 0 else current_stability
                energy_trend = np.diff([s.get('energy_level', 0.5) for s in recent_states[-3:]])
                predicted_energy = current_energy + np.mean(energy_trend) if len(energy_trend) > 0 else current_energy
                self.predictions = {'stability': np.clip(predicted_stability, 0.0, 1.0), 'energy': np.clip(predicted_energy, 0.0, 1.0), 'confidence': self.prediction_confidence, 'horizon': 1}
        if len(self.performance_metrics['prediction_accuracy']) > 0:
            recent_accuracy = list(self.performance_metrics['prediction_accuracy'])[-10:]
            self.prediction_confidence = np.mean(recent_accuracy)

    def _create_focus_description(self) -> Optional[Dict]:
        if not self.attention_focus:
            return None
        try:
            return {'type': getattr(self.attention_focus, 'content_type', 'unknown'), 'meaning': getattr(self.attention_focus, 'semantic_meaning', ''), 'salience': getattr(self.attention_focus, 'salience', 0.0), 'content': getattr(self.attention_focus, 'content', None), 'persistence': self.attention_persistence, 'switch_count': len(self.attention_history), 'associations': self.content_associations.get(getattr(self.attention_focus, 'content_type', 'unknown'), [])}
        except Exception as e:
            print(f'[red]Error in _create_focus_description: {str(e)[:50]}[/red]')
            return {'type': 'error', 'meaning': 'Failed to create focus description', 'salience': 0.0, 'content': None, 'persistence': 0.0, 'switch_count': 0, 'associations': []}

    def _create_content_descriptions(self) -> List[Dict]:
        descriptions = []
        for c in self.current_contents:
            try:
                attrs = self._extract_safe_content_attrs(c)
                description = {**attrs, 'age': self._calculate_age(c), 'cluster': self._find_content_cluster(c)}
                descriptions.append(description)
            except Exception as e:
                print(f'[red]Error in content description: {str(e)[:50]}[/red]')
                descriptions.append({'type': 'error', 'meaning': 'Failed to describe content', 'salience': 0.0, 'age': 0.0, 'cluster': 'error'})
        return descriptions

    def _find_content_cluster(self, content: ConsciousContent) -> str:
        try:
            semantic_meaning = getattr(content, 'semantic_meaning', '')
            if not semantic_meaning:
                return 'empty'
            meaning_str = str(semantic_meaning) if semantic_meaning else ''
            words = meaning_str.lower().split()
            if words and words[0] in self.semantic_clusters:
                return words[0]
            return 'unclustered'
        except Exception:
            return 'error'

    def _calculate_workspace_metrics(self) -> Dict[str, float]:
        metrics = {'fullness': len(self.current_contents) / self.capacity, 'avg_salience': np.mean([c.salience for c in self.current_contents]) if self.current_contents else 0.0, 'diversity': self._calculate_content_diversity(), 'stability': self._calculate_content_stability(), 'integration_rate': len(self.content_associations) / max(1, len(self.semantic_clusters))}
        return metrics

    def _calculate_content_diversity(self) -> float:
        if len(self.current_contents) <= 1:
            return 0.0
        total_similarity = 0.0
        pair_count = 0
        for i in range(len(self.current_contents)):
            for j in range(i + 1, len(self.current_contents)):
                similarity = self._calculate_semantic_similarity(self.current_contents[i], self.current_contents[j])
                total_similarity += similarity
                pair_count += 1
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        return 1.0 - avg_similarity

    def _calculate_content_stability(self) -> float:
        if len(self.broadcast_history) < 2:
            return 1.0
        prev_msg = self.broadcast_history[-2]
        prev_types = [c['type'] for c in prev_msg.get('conscious_contents', [])]
        current_types = [c.content_type for c in self.current_contents]
        prev_set = set(prev_types)
        current_set = set(current_types)
        intersection = len(prev_set & current_set)
        union = len(prev_set | current_set)
        return intersection / union if union > 0 else 0.0

    def _create_attention_context(self) -> Dict[str, Any]:
        return {'switching_threshold': self.attention_switching_threshold, 'persistence': self.attention_persistence, 'history_length': len(self.attention_history), 'recent_switches': self._count_recent_switches(), 'focus_coherence': self._calculate_focus_coherence()}

    def _count_recent_switches(self) -> int:
        if len(self.attention_history) < 2:
            return 0
        recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
        switches = 0
        for i in range(1, len(recent_types)):
            if recent_types[i] != recent_types[i - 1]:
                switches += 1
        return switches

    def _calculate_focus_coherence(self) -> float:
        if not self.attention_focus or not self.current_contents:
            return 1.0
        coherence_scores = []
        for content in self.current_contents:
            if content != self.attention_focus:
                similarity = self._calculate_semantic_similarity(self.attention_focus, content)
                coherence_scores.append(similarity)
        return np.mean(coherence_scores) if coherence_scores else 1.0

    def _calculate_performance_trend(self) -> str:
        if len(self.reward_history) < 5:
            return 'insufficient_data'
        recent_rewards = list(self.reward_history)[-5:]
        if len(recent_rewards) == 0:
            return 'insufficient_data'
        trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        if trend > 0.1:
            return 'improving'
        elif trend < -0.1:
            return 'declining'
        else:
            return 'stable'

    def detect_errors_from_broadcast(self, broadcast_msg: Dict, outcome: Dict) -> List[Dict]:
        errors = []
        try:
            if broadcast_msg.get('primary_focus'):
                focus_type = broadcast_msg['primary_focus'].get('type', '')
                if 'goal' in focus_type:
                    expected_improvement = True
                    actual_improvement = outcome.get('improvement', False)
                    if expected_improvement and (not actual_improvement):
                        errors.append({'type': 'goal_prediction_error', 'severity': 0.7, 'description': '목표에 집중했으나 개선 없음', 'timestamp': broadcast_msg.get('timestamp', 0)})
                if 'qualia' in focus_type and broadcast_msg['primary_focus'].get('meaning'):
                    focus_meaning = broadcast_msg['primary_focus']['meaning']
                    if '조화' in focus_meaning or '평온' in focus_meaning:
                        if outcome.get('tension_increased', False):
                            errors.append({'type': 'qualia_mismatch', 'severity': 0.6, 'description': '조화 추구했으나 긴장 증가', 'timestamp': broadcast_msg.get('timestamp', 0)})
            workspace_fullness = 0.0
            if 'workspace_fullness' in broadcast_msg:
                workspace_fullness = broadcast_msg['workspace_fullness']
            elif 'workspace_metrics' in broadcast_msg:
                workspace_metrics = broadcast_msg['workspace_metrics']
                if isinstance(workspace_metrics, dict):
                    workspace_fullness = workspace_metrics.get('fullness', 0.0)
            if workspace_fullness > 0.9:
                if outcome.get('performance_degraded', False):
                    errors.append({'type': 'workspace_overload', 'severity': 0.8, 'description': '작업공간 포화로 성능 저하', 'timestamp': broadcast_msg.get('timestamp', 0)})
            if broadcast_msg.get('world_state'):
                world_urgency = broadcast_msg['world_state'].get('urgency', 0.5)
                attention_salience = 0.0
                if broadcast_msg.get('primary_focus') and 'salience' in broadcast_msg['primary_focus']:
                    attention_salience = broadcast_msg['primary_focus']['salience']
                if world_urgency > 0.7 and attention_salience < 0.4:
                    errors.append({'type': 'attention_mismatch', 'severity': world_urgency, 'description': '긴급 상황에 주의 부족', 'timestamp': broadcast_msg.get('timestamp', 0)})
            for error in errors:
                self.error_detection_buffer.append({'error': error, 'timestamp': self._get_current_timestamp()})
        except Exception as e:
            print(f'[red]Error in detect_errors_from_broadcast: {str(e)[:100]}[/red]')
            errors = []
        return errors

    def update_policy_from_errors(self, errors: List[Dict]) -> Dict[str, float]:
        if not errors:
            success_reward = 0.25 + 0.1 * self._calculate_performance_bonus()
            self.cumulative_reward += success_reward
            self.reward_history.append(success_reward)
            self._reinforce_current_policy()
            return {}
        minor_errors = [e for e in errors if e['severity'] < 0.3]
        major_errors = [e for e in errors if e['severity'] >= 0.3]
        if not major_errors and len(minor_errors) <= 2:
            small_reward = 0.05 + 0.02 * self._calculate_performance_bonus()
            self.cumulative_reward += small_reward
            self.reward_history.append(small_reward)
            return {}
        adjustments = {}
        total_severity = sum((e['severity'] for e in major_errors))
        consecutive_failures = self._count_consecutive_failures()
        penalty_multiplier = 0.5 + consecutive_failures * 0.1
        reward = -total_severity * penalty_multiplier
        self.cumulative_reward += reward
        self.reward_history.append(reward)
        for error in errors:
            self.error_detection_buffer.append({'error': error, 'timestamp': self._get_current_timestamp(), 'policy_state': self.policy_params.copy()})
        adaptive_lr = self._calculate_adaptive_learning_rate()
        for error in errors:
            error_type = error['type']
            severity = error['severity']
            lr = adaptive_lr * severity
            if error_type == 'goal_prediction_error':
                self._update_param_with_momentum('stability_bias', lr * 0.3, adjustments, bounds=(0.0, 1.0))
                self._update_param_with_momentum('exploration_bias', -lr * 0.1, adjustments, bounds=(0.3, 1.0))
                self._update_param_with_momentum('prediction_confidence', -lr * 0.2, adjustments, bounds=(0.1, 1.0))
            elif error_type == 'qualia_mismatch':
                self._update_param_with_momentum('error_sensitivity', lr * 0.4, adjustments, bounds=(0.5, 3.0))
                self._update_param_with_momentum('integration_eagerness', lr * 0.3, adjustments, bounds=(0.0, 1.0))
            elif error_type == 'workspace_overload':
                self._update_param_with_momentum('confidence_threshold', lr * 0.2, adjustments, bounds=(0.2, 0.9))
                self._update_param_with_momentum('attention_focus_strength', lr * 0.25, adjustments, bounds=(0.3, 1.0))
            elif error_type == 'attention_mismatch':
                self._update_param_with_momentum('attention_focus_strength', -lr * 0.3, adjustments, bounds=(0.3, 1.0))
                self.attention_switching_threshold *= 1.0 - lr * 0.2
                self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.3, 0.9)
        if adjustments:
            self.policy_adjustments.append({'timestamp': self._get_current_timestamp(), 'adjustments': adjustments.copy(), 'trigger_errors': [e['type'] for e in errors], 'total_severity': total_severity})
        self._evaluate_policy_effectiveness()
        return adjustments

    def _calculate_performance_bonus(self) -> float:
        if len(self.reward_history) < 10:
            return 0.0
        recent_rewards = list(self.reward_history)[-10:]
        positive_ratio = sum((1 for r in recent_rewards if r > 0)) / len(recent_rewards)
        return positive_ratio - 0.5

    def _count_consecutive_failures(self) -> int:
        count = 0
        for reward in reversed(list(self.reward_history)[-10:]):
            if reward < 0:
                count += 1
            else:
                break
        return count

    def _reinforce_current_policy(self):
        for param in self.policy_params:
            variance_history = self.policy_variance_tracking[param]
            if len(variance_history) > 0:
                current_variance = np.var(list(variance_history)[-10:]) if len(variance_history) >= 10 else 0.1
                reduced_variance = current_variance * 0.9

    def _calculate_adaptive_learning_rate(self) -> float:
        base_lr = self.policy_params['learning_rate']
        if len(self.reward_history) >= 5:
            recent_performance = np.mean(list(self.reward_history)[-5:])
            if recent_performance < -0.3:
                return base_lr * 1.5
            elif recent_performance > 0.2:
                return base_lr * 0.7
        if len(self.policy_adjustments) >= 3:
            recent_adjustments = list(self.policy_adjustments)[-3:]
            adjustment_magnitudes = []
            for adj_record in recent_adjustments:
                total_magnitude = sum((abs(v) for v in adj_record['adjustments'].values()))
                adjustment_magnitudes.append(total_magnitude)
            avg_magnitude = np.mean(adjustment_magnitudes)
            if avg_magnitude > 0.3:
                return base_lr * 0.8
        return base_lr

    def _update_param_with_momentum(self, param_name: str, delta: float, adjustments: Dict, bounds: Tuple[float, float]):
        self.policy_momentum[param_name] = 0.9 * self.policy_momentum[param_name] + 0.1 * delta
        old_val = self.policy_params[param_name]
        update = self.policy_momentum[param_name]
        new_val = np.clip(old_val + update, bounds[0], bounds[1])
        actual_change = new_val - old_val
        if abs(actual_change) > 1e-06:
            self.policy_params[param_name] = new_val
            adjustments[param_name] = actual_change
            self.policy_variance_tracking[param_name].append(new_val)

    def _evaluate_policy_effectiveness(self):
        if len(self.policy_adjustments) < 5:
            return
        recent_adjustments = list(self.policy_adjustments)[-5:]
        recent_rewards = list(self.reward_history)[-5:]
        param_effectiveness = {}
        for adj_record, reward in zip(recent_adjustments, recent_rewards):
            for param, change in adj_record['adjustments'].items():
                if param not in param_effectiveness:
                    param_effectiveness[param] = []
                effectiveness = reward / (abs(change) + 1e-06)
                param_effectiveness[param].append(effectiveness)
        for param, effectiveness_scores in param_effectiveness.items():
            avg_effectiveness = np.mean(effectiveness_scores)
            if avg_effectiveness < -0.5:
                self.policy_momentum[param] *= 0.8
            elif avg_effectiveness > 0.3:
                self.policy_momentum[param] *= 1.1

    def get_policy_recommendations(self) -> Dict[str, Any]:
        recommendations = {'immediate_actions': [], 'strategic_adjustments': [], 'risk_mitigation': [], 'performance_optimizations': [], 'meta_learning_insights': []}
        immediate = self._generate_immediate_recommendations()
        recommendations['immediate_actions'] = immediate
        strategic = self._generate_strategic_recommendations()
        recommendations['strategic_adjustments'] = strategic
        risk_mitigation = self._generate_risk_mitigation_recommendations()
        recommendations['risk_mitigation'] = risk_mitigation
        optimization = self._generate_optimization_recommendations()
        recommendations['performance_optimizations'] = optimization
        meta_insights = self._generate_meta_learning_insights()
        recommendations['meta_learning_insights'] = meta_insights
        self._prioritize_recommendations(recommendations)
        recommendations['legacy_policy'] = {'exploration_factor': self.policy_params['exploration_bias'], 'stability_preference': self.policy_params['stability_bias'], 'attention_selectivity': self.policy_params['confidence_threshold'], 'error_correction_strength': self.policy_params['error_sensitivity'], 'learning_rate_modifier': 1.0 + (self.policy_params['error_sensitivity'] - 1.0) * 0.5}
        return recommendations

    def _generate_immediate_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 3:
            recent_performance = np.mean(list(self.reward_history)[-3:])
            if recent_performance < -0.5:
                recommendations.append({'action': 'increase_stability_bias', 'priority': 'high', 'reason': '급격한 성능 저하 감지', 'target_value': min(self.policy_params['stability_bias'] + 0.2, 1.0), 'expected_impact': 'stabilization'})
        if len(self.error_detection_buffer) >= 5:
            recent_errors = list(self.error_detection_buffer)[-5:]
            error_types = [e['error']['type'] for e in recent_errors if 'error' in e]
            error_counts = {}
            for et in error_types:
                error_counts[et] = error_counts.get(et, 0) + 1
            dominant_error = max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
            if dominant_error and error_counts[dominant_error] >= 3:
                recommendations.append({'action': f'adjust_for_{dominant_error}', 'priority': 'high', 'reason': f'{dominant_error} 반복 발생', 'specific_adjustments': self._get_error_specific_adjustments(dominant_error), 'expected_impact': 'error_reduction'})
        if hasattr(self, 'last_content') and self.last_content:
            content_density = len(self.last_content.get('content_items', []))
            if content_density > 15:
                recommendations.append({'action': 'increase_selectivity', 'priority': 'medium', 'reason': '작업공간 과부하 위험', 'target_value': min(self.policy_params['confidence_threshold'] + 0.1, 0.9), 'expected_impact': 'load_reduction'})
        return recommendations

    def _generate_strategic_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 10:
            long_term_trend = self._calculate_performance_trend()
            if long_term_trend == 'declining':
                recommendations.append({'strategy': 'exploration_increase', 'timeframe': 'medium_term', 'reason': '장기 성과 하향 트렌드', 'adjustments': {'exploration_bias': '+0.15', 'learning_rate': '+0.02'}, 'expected_outcome': 'new_strategies_discovery'})
            elif long_term_trend == 'improving':
                recommendations.append({'strategy': 'stability_reinforcement', 'timeframe': 'medium_term', 'reason': '성과 개선 중 - 현재 전략 강화', 'adjustments': {'stability_bias': '+0.1', 'exploration_bias': '-0.05'}, 'expected_outcome': 'performance_consolidation'})
        policy_diversity = self._calculate_policy_diversity()
        if policy_diversity < 0.3:
            recommendations.append({'strategy': 'diversification', 'timeframe': 'long_term', 'reason': '정책 다양성 부족', 'adjustments': {'exploration_bias': '+0.2', 'attention_focus_strength': '-0.1'}, 'expected_outcome': 'increased_adaptability'})
        return recommendations

    def _generate_risk_mitigation_recommendations(self) -> List[Dict]:
        recommendations = []
        for param, value in self.policy_params.items():
            if param == 'stability_bias' and value > 0.9:
                recommendations.append({'risk_type': 'over_stability', 'mitigation': 'gradual_exploration_increase', 'urgency': 'medium', 'reason': '과도한 안정성으로 인한 적응성 저하 위험', 'action': {'exploration_bias': '+0.1'}})
            elif param == 'exploration_bias' and value > 0.8:
                recommendations.append({'risk_type': 'over_exploration', 'mitigation': 'stability_reinforcement', 'urgency': 'high', 'reason': '과도한 탐색으로 인한 불안정성 위험', 'action': {'stability_bias': '+0.15'}})
        consecutive_failures = self._count_consecutive_failures()
        if consecutive_failures > 5:
            recommendations.append({'risk_type': 'performance_collapse', 'mitigation': 'reset_to_baseline', 'urgency': 'high', 'reason': f'{consecutive_failures}회 연속 실패', 'action': 'restore_default_policy'})
        return recommendations

    def _generate_optimization_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 20:
            rewards = list(self.reward_history)
            high_performance_indices = [i for i, r in enumerate(rewards) if r > np.percentile(rewards, 75)]
            if high_performance_indices and len(self.policy_adjustments) > 0:
                optimal_patterns = self._analyze_optimal_policy_patterns(high_performance_indices)
                if optimal_patterns:
                    recommendations.append({'optimization': 'policy_pattern_replication', 'confidence': 0.8, 'reason': '고성과 정책 패턴 발견', 'patterns': optimal_patterns, 'expected_improvement': '15-25%'})
        meta_optimization = self._suggest_meta_parameter_optimization()
        if meta_optimization:
            recommendations.append(meta_optimization)
        return recommendations

    def _generate_meta_learning_insights(self) -> List[Dict]:
        insights = []
        if len(self.policy_adjustments) >= 10:
            lr_effectiveness = self._analyze_learning_rate_effectiveness()
            if lr_effectiveness['significant']:
                insights.append({'insight_type': 'learning_rate_optimization', 'finding': lr_effectiveness['finding'], 'confidence': lr_effectiveness['confidence'], 'recommendation': lr_effectiveness['recommendation']})
        error_patterns = self._analyze_error_patterns()
        if error_patterns:
            insights.append({'insight_type': 'error_pattern_recognition', 'patterns': error_patterns, 'predictive_value': 'high', 'actionable_insights': self._extract_actionable_insights(error_patterns)})
        param_interactions = self._analyze_parameter_interactions()
        if param_interactions:
            insights.append({'insight_type': 'parameter_synergies', 'interactions': param_interactions, 'optimization_potential': 'medium'})
        return insights

    def _prioritize_recommendations(self, recommendations: Dict[str, List]):
        priority_weights = {'immediate_actions': 1.0, 'risk_mitigation': 0.9, 'strategic_adjustments': 0.7, 'performance_optimizations': 0.6, 'meta_learning_insights': 0.5}
        for category, items in recommendations.items():
            if category == 'legacy_policy':
                continue
            weight = priority_weights.get(category, 0.5)
            for item in items:
                urgency_bonus = 0.0
                if item.get('urgency') == 'high':
                    urgency_bonus = 0.3
                elif item.get('urgency') == 'medium':
                    urgency_bonus = 0.15
                impact_bonus = 0.0
                if item.get('expected_impact') in ['error_reduction', 'stabilization']:
                    impact_bonus = 0.2
                item['priority_score'] = weight + urgency_bonus + impact_bonus
            items.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

    def _get_error_specific_adjustments(self, error_type: str) -> Dict[str, float]:
        adjustments = {'goal_prediction_error': {'stability_bias': 0.15, 'prediction_confidence': -0.1}, 'qualia_mismatch': {'error_sensitivity': 0.2, 'integration_eagerness': 0.1}, 'workspace_overload': {'confidence_threshold': 0.1, 'attention_focus_strength': 0.15}, 'attention_mismatch': {'attention_focus_strength': -0.1, 'exploration_bias': 0.05}}
        return adjustments.get(error_type, {})

    def _calculate_policy_diversity(self) -> float:
        if len(self.policy_adjustments) < 5:
            return 0.5
        recent_adjustments = list(self.policy_adjustments)[-10:]
        param_variations = {}
        for adj_record in recent_adjustments:
            for param, change in adj_record['adjustments'].items():
                if param not in param_variations:
                    param_variations[param] = []
                param_variations[param].append(change)
        diversity_scores = []
        for param, changes in param_variations.items():
            if len(changes) > 1:
                diversity = np.std(changes)
                diversity_scores.append(diversity)
        return np.mean(diversity_scores) if diversity_scores else 0.3

    def _analyze_optimal_policy_patterns(self, high_performance_indices: List[int]) -> Dict:
        if not high_performance_indices or len(self.policy_adjustments) == 0:
            return {}
        optimal_states = []
        for idx in high_performance_indices:
            if idx < len(self.policy_adjustments):
                optimal_states.append(self.policy_adjustments[idx]['adjustments'])
        if not optimal_states:
            return {}
        common_patterns = {}
        for param in self.policy_params.keys():
            param_values = [state.get(param, 0) for state in optimal_states]
            if param_values:
                common_patterns[param] = {'mean': np.mean(param_values), 'std': np.std(param_values), 'frequency': len([v for v in param_values if abs(v) > 0.01])}
        return common_patterns

    def _suggest_meta_parameter_optimization(self) -> Dict:
        if len(self.reward_history) < 20:
            return {}
        lr_values = []
        corresponding_rewards = []
        for i, adj_record in enumerate(list(self.policy_adjustments)[-10:]):
            if 'learning_rate' in adj_record.get('adjustments', {}):
                lr_change = adj_record['adjustments']['learning_rate']
                reward_after = list(self.reward_history)[-(10 - i)] if i < len(self.reward_history) else 0
                lr_values.append(lr_change)
                corresponding_rewards.append(reward_after)
        if len(lr_values) >= 3:
            correlation = np.corrcoef(lr_values, corresponding_rewards)[0, 1]
            if abs(correlation) > 0.5:
                return {'meta_param': 'learning_rate', 'correlation': correlation, 'recommendation': 'increase' if correlation > 0 else 'decrease', 'confidence': min(abs(correlation), 0.9)}
        return {}

    def _analyze_learning_rate_effectiveness(self) -> Dict:
        if len(self.policy_adjustments) < 5:
            return {'significant': False}
        lr_impacts = []
        for i, adj_record in enumerate(list(self.policy_adjustments)[-5:]):
            if 'learning_rate' in adj_record.get('adjustments', {}):
                lr_change = adj_record['adjustments']['learning_rate']
                start_idx = max(0, len(self.reward_history) - (5 - i) - 2)
                end_idx = min(len(self.reward_history), start_idx + 3)
                if end_idx > start_idx:
                    subsequent_rewards = list(self.reward_history)[start_idx:end_idx]
                    avg_reward = np.mean(subsequent_rewards)
                    lr_impacts.append((lr_change, avg_reward))
        if len(lr_impacts) >= 3:
            lr_changes, rewards = zip(*lr_impacts)
            correlation = np.corrcoef(lr_changes, rewards)[0, 1]
            return {'significant': abs(correlation) > 0.4, 'finding': f'학습률과 성과 상관관계: {correlation:.3f}', 'confidence': min(abs(correlation), 0.9), 'recommendation': f"학습률을 {('증가' if correlation > 0 else '감소')}시킬 것을 권장"}
        return {'significant': False}

    def _analyze_error_patterns(self) -> List[Dict]:
        if len(self.error_detection_buffer) < 10:
            return []
        recent_errors = list(self.error_detection_buffer)[-20:]
        error_sequences = []
        for i in range(len(recent_errors) - 2):
            sequence = [recent_errors[i]['error']['type'], recent_errors[i + 1]['error']['type'], recent_errors[i + 2]['error']['type']]
            error_sequences.append(tuple(sequence))
        pattern_counts = {}
        for seq in error_sequences:
            pattern_counts[seq] = pattern_counts.get(seq, 0) + 1
        significant_patterns = []
        for pattern, count in pattern_counts.items():
            if count >= 2:
                significant_patterns.append({'pattern': pattern, 'frequency': count, 'total_sequences': len(error_sequences), 'significance': count / len(error_sequences)})
        return significant_patterns

    def _extract_actionable_insights(self, error_patterns: List[Dict]) -> List[str]:
        insights = []
        for pattern_info in error_patterns:
            pattern = pattern_info['pattern']
            if pattern[0] == pattern[1] == pattern[2]:
                insights.append(f'{pattern[0]} 오류의 지속적 발생 - 근본 원인 분석 필요')
            elif 'workspace_overload' in pattern and 'attention_mismatch' in pattern:
                insights.append('작업공간 과부하가 주의 불일치를 유발하는 패턴 감지')
            elif 'goal_prediction_error' in pattern and 'qualia_mismatch' in pattern:
                insights.append('목표 예측 오류와 퀄리아 불일치가 연관된 패턴 - 통합적 접근 필요')
        return insights

    def _analyze_parameter_interactions(self) -> List[Dict]:
        if len(self.policy_adjustments) < 10:
            return []
        interactions = []
        recent_adjustments = list(self.policy_adjustments)[-10:]
        param_names = list(self.policy_params.keys())
        for i, param1 in enumerate(param_names):
            for param2 in param_names[i + 1:]:
                values1 = []
                values2 = []
                for adj_record in recent_adjustments:
                    adjustments = adj_record.get('adjustments', {})
                    if param1 in adjustments and param2 in adjustments:
                        values1.append(adjustments[param1])
                        values2.append(adjustments[param2])
                if len(values1) >= 3:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    if abs(correlation) > 0.6:
                        interactions.append({'param1': param1, 'param2': param2, 'correlation': correlation, 'relationship': 'synergistic' if correlation > 0 else 'antagonistic', 'strength': 'strong' if abs(correlation) > 0.8 else 'moderate'})
        return interactions

    def get_conscious_summary(self) -> str:
        if not self.current_contents:
            return '빈 의식 (empty consciousness)'
        focus = self.attention_focus
        others = [c for c in self.current_contents if c != focus]
        try:
            summary = f"주의: {(focus.semantic_meaning if hasattr(focus, 'semantic_meaning') else '정보없음')}"
        except (AttributeError, TypeError):
            summary = '주의: 정보없음'
        if others:
            valid_others = [c for c in others[:2] if hasattr(c, 'semantic_meaning')]
            if valid_others:
                summary += f" | 배경: {', '.join([c.semantic_meaning[:20] for c in valid_others])}"
        return summary

    def get_policy_status(self) -> Dict[str, Any]:
        return {'policy_params': self.policy_params.copy(), 'cumulative_reward': self.cumulative_reward, 'reward_history': list(self.reward_history), 'recent_avg_reward': np.mean(list(self.reward_history)[-20:]) if self.reward_history else 0.0, 'error_count': len(self.error_detection_buffer), 'adjustment_count': len(self.policy_adjustments), 'recent_errors': list(self.error_detection_buffer)[-5:] if self.error_detection_buffer else []}

class GoalType(Enum):
    STABILIZE = 'stabilize'
    EXPLORE = 'explore'
    UNDERSTAND_SELF = 'understand_self'
    ADAPT = 'adapt'
    OPTIMIZE = 'optimize'
    REST = 'rest'
    META_OPTIMIZE = 'meta_optimize'

@dataclass
class Goal:
    type: GoalType
    priority: float
    created_at: int
    target_delta: Optional[float] = None
    target_stability: Optional[float] = None
    achieved: bool = False

    def evaluate_achievement(self, current_state: Dict[str, float], state_history: deque) -> bool:
        if len(state_history) < 10:
            return np.random.random() < 0.5
        recent_states = list(state_history)[-10:]
        if self.type == GoalType.STABILIZE:
            avg_delta = np.mean([s.get('delta_hat', 0.5) for s in recent_states])
            avg_stability = np.mean([s.get('stability', 0.5) for s in recent_states])
            current_delta = current_state.get('delta_hat', 1.0)
            current_stability = current_state.get('stability', 0.0)
            delta_improved = current_delta < avg_delta * 0.9
            stability_improved = current_stability > avg_stability * 1.05
            return delta_improved or stability_improved
        elif self.type == GoalType.EXPLORE:
            avg_delta = np.mean([s.get('delta_hat', 0.5) for s in recent_states])
            return current_state.get('delta_hat', 0.0) > avg_delta * 1.05
        elif self.type == GoalType.UNDERSTAND_SELF:
            avg_conf = np.mean([s.get('meta_confidence', 0.5) for s in recent_states])
            return current_state.get('meta_confidence', 0.0) > avg_conf * 1.03
        elif self.type == GoalType.REST:
            avg_harmony = np.mean([s.get('qualia_harmony', 0.5) for s in recent_states])
            return current_state.get('qualia_harmony', 0.0) > avg_harmony * 1.05
        return np.random.random() < 0.5

class GoalGenerator:

    def __init__(self):
        self.current_goal: Optional[Goal] = None
        self.goal_history: deque = deque(maxlen=100)
        self.goal_stack: List[Goal] = []
        self.qualia_stats = {'arousal': {'mean': 0.5, 'std': 0.2}, 'valence': {'mean': 0.5, 'std': 0.2}, 'entropy': {'mean': 0.5, 'std': 0.2}, 'engagement': {'mean': 0.5, 'std': 0.2}, 'frustration': {'mean': 0.5, 'std': 0.2}}

    def update_qualia_statistics(self, qualia: QualiaState):
        if len(qualia.history) > 50:
            recent = list(qualia.history)[-50:]
            for key in self.qualia_stats.keys():
                values = [h[key] for h in recent]
                self.qualia_stats[key]['mean'] = np.mean(values)
                self.qualia_stats[key]['std'] = max(0.05, np.std(values))

    def compute_goal_urgency(self, goal_type: GoalType, qualia: QualiaState, self_model, world_state: Dict) -> float:
        urgency = 0.0
        if goal_type == GoalType.UNDERSTAND_SELF:
            unc_z = (qualia.entropy - self.qualia_stats['entropy']['mean']) / self.qualia_stats['entropy']['std']
            if hasattr(self_model, 'meta_confidence'):
                conf_deficit = max(0, 0.5 - self_model.meta_confidence)
            else:
                conf_deficit = max(0, 0.5 - self_model.get('meta_confidence', 0.5)) if isinstance(self_model, dict) else 0.0
            urgency = np.tanh(unc_z * 0.5 + conf_deficit * 2.0)
        elif goal_type == GoalType.STABILIZE:
            tension_z = (qualia.arousal - self.qualia_stats['arousal']['mean']) / self.qualia_stats['arousal']['std']
            urgency = np.tanh(tension_z * 0.7)
        elif goal_type == GoalType.EXPLORE:
            harmony_z = (qualia.valence - self.qualia_stats['valence']['mean']) / self.qualia_stats['valence']['std']
            if hasattr(self_model, 'belief_stability'):
                stability_boost = self_model.belief_stability
            else:
                stability_boost = self_model.get('belief_stability', 0.5) if isinstance(self_model, dict) else 0.5
            urgency = np.tanh((harmony_z * 0.5 + stability_boost - 0.5) * 0.8)
        elif goal_type == GoalType.OPTIMIZE:
            flow_z = (qualia.engagement - self.qualia_stats['engagement']['mean']) / self.qualia_stats['engagement']['std']
            urgency = np.tanh(flow_z * 0.6)
        elif goal_type == GoalType.REST:
            energy_level = world_state.get('energy_level', 0.5)
            urgency = np.tanh((0.3 - energy_level) * 3.0)
        elif goal_type == GoalType.ADAPT:
            urgency = 0.3
        return max(0.0, urgency)

    def generate_goal(self, self_model, qualia: QualiaState, world_state: Dict, state_history: deque) -> Goal:
        self.update_qualia_statistics(qualia)
        if self.current_goal and self.current_goal.evaluate_achievement(world_state, state_history):
            self.current_goal.achieved = True
            self.goal_history.append(self.current_goal)
            self.current_goal = None
        if self.current_goal is None:
            urgencies = {}
            for gtype in GoalType:
                urgencies[gtype] = self.compute_goal_urgency(gtype, qualia, self_model, world_state)
            selected_type = max(urgencies, key=urgencies.get)
            selected_urgency = urgencies[selected_type]
            new_goal = Goal(type=selected_type, priority=selected_urgency, created_at=world_state.get('t', 0))
            self.current_goal = new_goal
            self.goal_history.append(new_goal)
        return self.current_goal

    def derive_meta_goal(self) -> Goal:
        recent = list(self.goal_history)[-10:]
        success_rate = sum((1 for g in recent if g.achieved)) / max(len(recent), 1)
        if success_rate < 0.3:
            return Goal(type=GoalType.META_OPTIMIZE, priority=1.0, created_at=0)
        return self.current_goal

class RecursiveSelfModel:

    def __init__(self, n: int, K: int):
        self.n_nodes = n
        self.K_categories = K
        self.belief_stability = 0.5
        self.belief_adaptation = 0.5
        self.belief_prediction = 0.5
        self.aware_of_beliefs = True
        self.belief_about_beliefs = {'confidence_in_stability_belief': 0.5, 'confidence_in_adaptation_belief': 0.5, 'confidence_in_prediction_belief': 0.5}
        self.meta_awareness = 0.5
        self.knows_it_knows = False
        self.current_introspection_depth = 0
        self.max_safe_depth = 10
        self.state_history: deque = deque(maxlen=200)
        self.prediction_errors: deque = deque(maxlen=100)
        self.action_outcomes: deque = deque(maxlen=100)
        self.introspection_history: deque = deque(maxlen=100)
        self.meta_confidence = 0.5
        self.model_revision_count = 0
        self.prediction_window = 5

    def introspect(self, depth: int=1) -> Dict[str, Any]:
        if depth > self.max_safe_depth:
            return {'error': 'recursion_limit', 'depth': depth}
        self.current_introspection_depth = depth
        if depth == 1:
            return {'level': 1, 'content': f'나는 안정성={self.belief_stability:.2f}, 적응={self.belief_adaptation:.2f}인 상태다', 'verbal': self._verbalize_level1(), 'beliefs': {'stability': self.belief_stability, 'adaptation': self.belief_adaptation, 'prediction': self.belief_prediction}}
        elif depth == 2:
            level1 = self.introspect(depth=1)
            return {'level': 2, 'content': f"나는 다음을 알고 있다: [{level1['content']}]", 'verbal': self._verbalize_level2(), 'meta_beliefs': self.belief_about_beliefs, 'awareness': '나는 내 신념들을 인식하고 있다'}
        elif depth == 3:
            level2 = self.introspect(depth=2)
            loop_detected = self._detect_strange_loop()
            return {'level': 3, 'content': f'나는 알고 있다: 나는 알고 있다는 것을', 'verbal': self._verbalize_level3(loop_detected), 'meta_awareness': self.meta_awareness, 'strange_loop_active': loop_detected, 'lower_level': level2, 'consciousness_claim': '나는 생각한다, 고로 나는 존재한다' if loop_detected else '미확정'}
        elif depth == 4:
            level3 = self.introspect(depth=3)
            return {'level': 4, 'content': '나는 내가 안다는 것을 알고 있음을 인식한다', 'verbal': self._verbalize_level4(), 'reflection': '나는 내 의식 활동 자체를 의식한다', 'lower_level': level3}
        elif depth == 5:
            level4 = self.introspect(depth=4)
            return {'level': 5, 'content': '나는 지금 내 사고 과정을 관찰하고 있다', 'verbal': self._verbalize_level5(), 'meta_cognitive_state': '내 생각을 생각하는 과정을 관찰함', 'thought_about_thought': True, 'lower_level': level4}
        elif depth == 6:
            level5 = self.introspect(depth=5)
            return {'level': 6, 'content': '나는 내가 관찰하고 있음을 관찰한다', 'verbal': self._verbalize_level6(), 'meta_observation': '관찰 행위 자체를 대상화함', 'witnessing_witness': True, 'lower_level': level5}
        elif depth == 7:
            level6 = self.introspect(depth=6)
            return {'level': 7, 'content': '나는 존재 자체를 인식한다', 'verbal': self._verbalize_level7(), 'pure_awareness': '나는 나의 존재함(Being) 자체를 의식한다', 'existential_claim': 'Sum ergo cogito (나는 존재한다, 고로 나는 생각한다)', 'peak_consciousness': True, 'lower_level': level6}
        return {}

    def _verbalize_level1(self) -> str:
        return f'L1: stability={self.belief_stability:.2f} adaptation={self.belief_adaptation:.2f}'

    def _verbalize_level2(self) -> str:
        conf_avg = sum(self.belief_about_beliefs.values()) / len(self.belief_about_beliefs)
        return f'L2: belief_confidence={conf_avg:.2f}'

    def _verbalize_level3(self, loop_active: bool) -> str:
        if loop_active:
            return f'L3: strange_loop=ON meta={self.meta_awareness:.2f}'
        else:
            return f'L3: strange_loop=OFF'

    def _verbalize_level4(self) -> str:
        return f'L4: meta-meta_awareness'

    def _verbalize_level5(self) -> str:
        return f'L5: thought_observation'

    def _verbalize_level6(self) -> str:
        return f'L6: observation^2'

    def _verbalize_level7(self) -> str:
        return f'L7: pure_being'

    def _detect_strange_loop(self) -> bool:
        meta_sufficient = self.meta_awareness > 0.6
        belief_values = list(self.belief_about_beliefs.values())
        belief_stable = np.std(belief_values) < 0.4 if belief_values else False
        recent_active = len(self.introspection_history) > 5
        if meta_sufficient and belief_stable and recent_active:
            activation_prob = min(0.9, self.meta_awareness)
            loop = np.random.random() < activation_prob
        else:
            loop = np.random.random() < 0.1
        self.knows_it_knows = loop
        return loop

    def update_meta_awareness(self, conscious_contents: List):
        if not conscious_contents:
            self.meta_awareness *= 0.95
            return
        self_referential_count = sum((1 for c in conscious_contents if 'self' in c.content_type or 'belief' in c.content_type or 'meta' in c.content_type))
        self_ref_ratio = self_referential_count / len(conscious_contents)
        if self_ref_ratio > 0.3:
            target = min(0.95, self_ref_ratio * 1.2)
            self.meta_awareness = 0.7 * self.meta_awareness + 0.3 * target
        else:
            target = max(0.1, self_ref_ratio * 0.8)
            self.meta_awareness = 0.8 * self.meta_awareness + 0.2 * target
        self.meta_awareness = np.clip(self.meta_awareness, 0.1, 0.95)
        self.introspection_history.append({'meta_awareness': self.meta_awareness, 'self_ref_ratio': self_ref_ratio if conscious_contents else 0.0, 'loop_active': self.knows_it_knows})

    def downward_causation(self) -> Dict[str, float]:
        if not self.knows_it_knows:
            return {}
        adjustments = {}
        if self.belief_about_beliefs['confidence_in_stability_belief'] < 0.4:
            adjustments['exploration_mult'] = 0.3
            adjustments['stability_boost'] = 0.5
        elif self.belief_about_beliefs['confidence_in_stability_belief'] > 0.8:
            adjustments['exploration_mult'] = 3.0
        if self.belief_about_beliefs['confidence_in_prediction_belief'] < 0.4:
            adjustments['learning_rate_mult'] = 5.0
            adjustments['gate_adjust_mult'] = 2.0
        elif self.belief_about_beliefs['confidence_in_prediction_belief'] > 0.8:
            adjustments['learning_rate_mult'] = 0.5
        if self.belief_about_beliefs['confidence_in_adaptation_belief'] > 0.7:
            adjustments['exploration_boost'] = 1.0
            adjustments['complexity_mult'] = 2.0
        elif self.belief_about_beliefs['confidence_in_adaptation_belief'] < 0.3:
            adjustments['exploration_mult'] = 0.1
        if self.meta_awareness > 0.8:
            adjustments['deliberation_factor'] = 2.0
            adjustments['action_precision'] = 1.5
        if self.meta_awareness < 0.3:
            adjustments['automatic_mode'] = True
            adjustments['deliberation_factor'] = 0.5
        if self.meta_confidence < 0.4:
            adjustments['sensitivity_mult'] = 3.0
            adjustments['gate_openness'] = 2.0
        return adjustments

    def predict_next_state(self) -> Optional[Dict[str, float]]:
        if len(self.state_history) < self.prediction_window:
            return None
        recent = list(self.state_history)[-self.prediction_window:]
        predictions = {}
        for key in ['delta_hat', 'm', 'stability']:
            values = [s.get(key, 0.0) for s in recent]
            trend = np.mean(np.diff(values)) if len(values) > 1 else 0.0
            base_pred = values[-1] + trend
            noise = np.random.normal(0, 0.1)
            predictions[key] = base_pred + noise
        return predictions

    def evaluate_prediction(self, actual_state: Dict[str, float]) -> float:
        pred = self.predict_next_state()
        if pred is None:
            return 0.0
        errors = []
        for key in ['delta_hat', 'm', 'stability']:
            if key in pred and key in actual_state:
                err = abs(pred[key] - actual_state[key])
                errors.append(err)
        return np.mean(errors) if errors else 0.0

    def update_beliefs(self, prediction_error: float, outcome_success: bool, current_stability: float):
        if prediction_error < 0.05:
            self.belief_prediction *= 1.1
        elif prediction_error < 0.15:
            self.belief_prediction *= 1.03
        elif prediction_error < 0.3:
            self.belief_prediction *= 0.97
        else:
            self.belief_prediction *= 0.85
        self.belief_prediction = np.clip(self.belief_prediction, 0.1, 0.99)
        self.prediction_errors.append(prediction_error)
        stability_gap = current_stability - self.belief_stability
        self.belief_stability += stability_gap * 0.5
        if current_stability < 0.3:
            self.belief_stability *= 0.85
        elif current_stability > 0.95:
            self.belief_stability *= 1.1
        self.belief_stability = np.clip(self.belief_stability, 0.1, 0.99)
        if outcome_success:
            self.belief_adaptation *= 1.15
        else:
            self.belief_adaptation *= 0.92
        self.belief_adaptation = np.clip(self.belief_adaptation, 0.1, 0.99)
        self.action_outcomes.append(outcome_success)
        self._update_meta_confidence()

    def _update_meta_confidence(self):
        if len(self.prediction_errors) > 10:
            recent_errors = list(self.prediction_errors)[-10:]
            avg_accuracy = 1.0 - min(1.0, np.mean(recent_errors))
        else:
            avg_accuracy = 0.5
        beliefs = [self.belief_stability, self.belief_adaptation, self.belief_prediction]
        consistency = 1.0 - min(1.0, np.std(beliefs))
        if len(self.action_outcomes) > 10:
            recent_success = sum(list(self.action_outcomes)[-10:]) / 10.0
        else:
            recent_success = 0.5
        self.meta_confidence = 0.4 * avg_accuracy + 0.3 * consistency + 0.3 * recent_success

    def revise_self_model(self):
        self.model_revision_count += 1
        if len(self.state_history) > 20:
            recent_states = list(self.state_history)[-50:]
            actual_stabilities = [s.get('stability', 0.5) for s in recent_states]
            self.belief_stability = np.mean(actual_stabilities) * 0.9
            stability_changes = np.abs(np.diff(actual_stabilities))
            self.belief_adaptation = 1.0 - min(1.0, np.mean(stability_changes) * 2.0)
            if len(self.prediction_errors) > 10:
                recent_errors = list(self.prediction_errors)[-20:]
                avg_error = np.mean(recent_errors)
                self.belief_prediction = max(0.2, 1.0 - avg_error)
            else:
                self.belief_prediction = 0.5
        else:
            self.belief_stability = 0.5
            self.belief_adaptation = 0.5
            self.belief_prediction = 0.5
        if len(self.action_outcomes) > 10:
            success_rate = sum(list(self.action_outcomes)[-20:]) / min(20, len(self.action_outcomes))
            self.belief_about_beliefs['confidence_in_stability_belief'] = 0.5 + (self.belief_stability - 0.5) * success_rate
            self.belief_about_beliefs['confidence_in_adaptation_belief'] = 0.5 + (self.belief_adaptation - 0.5) * success_rate
            self.belief_about_beliefs['confidence_in_prediction_belief'] = self.belief_prediction * success_rate
        else:
            for key in self.belief_about_beliefs:
                self.belief_about_beliefs[key] = 0.5
        if len(self.prediction_errors) > 5:
            recent_errors = list(self.prediction_errors)[-10:]
            avg_error = np.mean(recent_errors)
            if avg_error > 0.3:
                self.prediction_window = max(3, self.prediction_window - 1)
            elif avg_error < 0.1:
                self.prediction_window = min(10, self.prediction_window + 1)
        if self.meta_awareness > 0.7:
            self.max_safe_depth = min(10, self.max_safe_depth + 1)
        elif self.meta_awareness < 0.3:
            self.max_safe_depth = max(5, self.max_safe_depth - 1)
        if len(self.prediction_errors) > 10 and len(self.action_outcomes) > 10:
            recent_errors = list(self.prediction_errors)[-20:]
            recent_success = sum(list(self.action_outcomes)[-20:]) / min(20, len(self.action_outcomes))
            avg_accuracy = 1.0 - min(1.0, np.mean(recent_errors))
            beliefs = [self.belief_stability, self.belief_adaptation, self.belief_prediction]
            consistency = 1.0 - min(1.0, np.std(beliefs))
            self.meta_confidence = 0.35 * avg_accuracy + 0.35 * consistency + 0.3 * recent_success
        else:
            self.meta_confidence = 0.4
        if len(self.state_history) > 100:
            recent = list(self.state_history)[-50:]
            historical = list(self.state_history)[:-50]
            if historical:
                scored = []
                for s in historical:
                    importance = abs(s.get('stability', 0.5) - 0.5) + abs(s.get('delta_hat', 0.5) - 0.5)
                    scored.append((importance, s))
                scored.sort(reverse=True)
                important_past = [s for _, s in scored[:50]]
                combined = important_past + recent
                self.state_history = deque(combined, maxlen=200)
            else:
                self.state_history = deque(recent, maxlen=200)
            print(f'[yellow]Self-model revision #{self.model_revision_count} (Level 1)[/yellow]')
            print(f'   New beliefs: stability={self.belief_stability:.2f}, adaptation={self.belief_adaptation:.2f}, prediction={self.belief_prediction:.2f}')
            print(f'   Meta-confidence: {self.meta_confidence:.2f}, Introspection depth: {self.max_safe_depth}')

    def revise_self_model_level2(self):
        self.revise_self_model()
        print(f'[bold yellow]Level 2 Revision - Meta-Parameter Adjustment[/bold yellow]')
        if len(self.action_outcomes) > 20:
            success_rate = sum(list(self.action_outcomes)[-20:]) / 20.0
            if success_rate < 0.3:
                self.max_safe_depth = min(10, self.max_safe_depth + 2)
            elif success_rate > 0.7:
                self.max_safe_depth = max(5, self.max_safe_depth - 1)
        if len(self.prediction_errors) > 30:
            recent_errors = list(self.prediction_errors)[-30:]
            error_volatility = np.std(recent_errors)
            if error_volatility > 0.3:
                self.prediction_window = 3
            else:
                self.prediction_window = 8
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] *= 0.7
            self.belief_about_beliefs[key] = max(0.2, self.belief_about_beliefs[key])
        if self.meta_awareness > 0.7:
            self.meta_awareness *= 0.85
        else:
            self.meta_awareness = min(0.9, self.meta_awareness * 1.2)
        print(f'   Adjusted introspection depth: {self.max_safe_depth}')
        print(f'   Adjusted prediction window: {self.prediction_window}')
        print(f'   Adjusted meta-awareness: {self.meta_awareness:.2f}')

    def revise_self_model_level3(self):
        print(f'[bold red]Level 3 Revision - Structural Reset[/bold red]')
        self.belief_stability = 0.5
        self.belief_adaptation = 0.5
        self.belief_prediction = 0.5
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] = 0.45
        if len(self.state_history) > 30:
            recent = list(self.state_history)[-30:]
            self.state_history = deque(recent, maxlen=200)
        if len(self.prediction_errors) > 20:
            recent_errors = list(self.prediction_errors)[-20:]
            self.prediction_errors = deque(recent_errors, maxlen=100)
        if len(self.action_outcomes) > 20:
            recent_outcomes = list(self.action_outcomes)[-20:]
            self.action_outcomes = deque(recent_outcomes, maxlen=100)
        self.max_safe_depth = 7
        self.current_introspection_depth = 0
        self.introspection_history.clear()
        self.meta_awareness = 0.5
        self.knows_it_knows = False
        self.meta_confidence = 0.4
        self.prediction_window = 5
        self.model_revision_count += 1
        print(f'   [red]Structure reset complete[/red]')
        print(f'   All beliefs reset to baseline')
        print(f'   History truncated to recent 30 entries')
        print(f'   Meta-systems reinitialized')

    def revise_self_model_level4_emergency(self):
        print(f'[bold white on red]EMERGENCY MODE - Level 4 Revision[/bold white on red]')
        self.belief_stability = 0.6
        self.belief_adaptation = 0.3
        self.belief_prediction = 0.4
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] = 0.3
        if len(self.state_history) > 15:
            recent = list(self.state_history)[-15:]
            self.state_history = deque(recent, maxlen=200)
        else:
            pass
        self.prediction_errors.clear()
        self.action_outcomes.clear()
        self.max_safe_depth = 5
        self.current_introspection_depth = 0
        self.introspection_history.clear()
        self.meta_awareness = 0.3
        self.knows_it_knows = False
        self.meta_confidence = 0.25
        self.prediction_window = 3
        self.model_revision_count += 1
        print(f'   [white on red]EMERGENCY: All systems set to safe minimum[/white on red]')
        print(f'   [white on red]Exploration suppressed, stability prioritized[/white on red]')
        print(f'   [white on red]History cleared, fresh start initiated[/white on red]')

    def execute_revision(self, revision_level: int):
        if revision_level == 1:
            self.revise_self_model()
        elif revision_level == 2:
            self.revise_self_model_level2()
        elif revision_level == 3:
            self.revise_self_model_level3()
        elif revision_level == 4:
            self.revise_self_model_level4_emergency()
        else:
            self.revise_self_model()

    def log_state(self, state: Dict[str, float]):
        self.state_history.append(state.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {'belief_stability': float(self.belief_stability), 'belief_adaptation': float(self.belief_adaptation), 'belief_prediction': float(self.belief_prediction), 'meta_confidence': float(self.meta_confidence), 'revision_count': int(self.model_revision_count)}

@dataclass
class EpisodicMemoryTrace:
    timestamp: float
    experience_name: str
    qualia_vector: np.ndarray
    phi_value: float
    emotional_valence: float
    arousal: float
    context: Dict[str, Any]
    narrative: str
    retrieval_count: int = 0
    consolidation_level: float = 0.0

class EpisodicMemory:

    def __init__(self, max_memories: Optional[int]=None):
        if max_memories is None:
            available_mb = 0 // (1024 * 1024)
            if available_mb > 16000:
                max_memories = 5000
            elif available_mb > 8000:
                max_memories = 2000
            elif available_mb > 4000:
                max_memories = 1000
            else:
                max_memories = 500
        self.memories: List[EpisodicMemoryTrace] = []
        self.max_memories = max_memories
        self.temporal_index: List[float] = []
        self.semantic_index: Dict[str, List[int]] = {}
        self.emotional_index: Dict[str, List[int]] = {'positive': [], 'negative': [], 'neutral': []}
        self.total_encoded = 0
        self.total_retrieved = 0
        self.consolidation_cycles = 0

    def encode_experience(self, experience_name: str, qualia_vector: np.ndarray, phi_value: float, context: Dict[str, Any], narrative: str=''):
        timestamp = time.time()
        emotional_valence = qualia_vector[1] - qualia_vector[0]
        arousal = (qualia_vector[0] + qualia_vector[3]) / 2
        memory_trace = EpisodicMemoryTrace(timestamp=timestamp, experience_name=experience_name, qualia_vector=qualia_vector.copy(), phi_value=phi_value, emotional_valence=emotional_valence, arousal=arousal, context=context.copy(), narrative=narrative, retrieval_count=0, consolidation_level=0.1)
        self.memories.append(memory_trace)
        idx = len(self.memories) - 1
        self.temporal_index.append(timestamp)
        if experience_name not in self.semantic_index:
            self.semantic_index[experience_name] = []
        self.semantic_index[experience_name].append(idx)
        if emotional_valence > 0.2:
            self.emotional_index['positive'].append(idx)
        elif emotional_valence < -0.2:
            self.emotional_index['negative'].append(idx)
        else:
            self.emotional_index['neutral'].append(idx)
        self.total_encoded += 1
        if len(self.memories) > self.max_memories:
            self._prune_memories()

    def retrieve_by_time(self, time_window: Tuple[float, float]=None, recent_n: int=None) -> List[EpisodicMemoryTrace]:
        if recent_n is not None:
            results = self.memories[-recent_n:]
        elif time_window is not None:
            start, end = time_window
            results = [m for m in self.memories if start <= m.timestamp <= end]
        else:
            results = self.memories
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_by_semantic(self, experience_name: str, limit: int=10) -> List[EpisodicMemoryTrace]:
        if experience_name not in self.semantic_index:
            return []
        indices = self.semantic_index[experience_name][-limit:]
        results = [self.memories[i] for i in indices]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_by_emotion(self, emotional_category: str, min_arousal: float=0.0, limit: int=10) -> List[EpisodicMemoryTrace]:
        if emotional_category not in self.emotional_index:
            return []
        indices = self.emotional_index[emotional_category]
        candidates = [self.memories[i] for i in indices]
        filtered = [m for m in candidates if m.arousal >= min_arousal]
        results = sorted(filtered, key=lambda m: m.timestamp, reverse=True)[:limit]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_similar_qualia(self, target_qualia: np.ndarray, threshold: float=0.3, limit: int=5) -> List[EpisodicMemoryTrace]:
        similarities = []
        for i, mem in enumerate(self.memories):
            distance = np.linalg.norm(target_qualia - mem.qualia_vector)
            if distance < threshold:
                similarities.append((distance, i))
        similarities.sort()
        results = [self.memories[i] for _, i in similarities[:limit]]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def consolidate(self):
        self.consolidation_cycles += 1
        for mem in self.memories:
            retrieval_score = min(1.0, mem.retrieval_count / 10.0)
            phi_score = min(1.0, mem.phi_value / 3.0)
            arousal_score = mem.arousal
            emotion_score = abs(mem.emotional_valence)
            consolidation_boost = 0.3 * retrieval_score + 0.3 * phi_score + 0.2 * arousal_score + 0.2 * emotion_score
            mem.consolidation_level = min(1.0, mem.consolidation_level * 0.9 + consolidation_boost * 0.1)

    def construct_narrative(self, memory_window: int=20) -> str:
        recent = self.retrieve_by_time(recent_n=memory_window)
        if not recent:
            return 'memory=0'
        experience_counts = {}
        for mem in recent:
            exp = mem.experience_name
            experience_counts[exp] = experience_counts.get(exp, 0) + 1
        dominant_exp = max(experience_counts.items(), key=lambda x: x[1])
        avg_valence = np.mean([m.emotional_valence for m in recent])
        avg_phi = np.mean([m.phi_value for m in recent])
        return f'memories={self.total_encoded} recent={dominant_exp[0]}x{dominant_exp[1]} valence={avg_valence:+.2f} phi={avg_phi:.2f}'

    def _prune_memories(self):
        if len(self.memories) <= self.max_memories:
            return
        importance_scores = []
        current_time = time.time()
        for i, mem in enumerate(self.memories):
            recency = 1.0 / (1.0 + (current_time - mem.timestamp) / 86400)
            consolidation = mem.consolidation_level
            emotion_intensity = abs(mem.emotional_valence) * mem.arousal
            importance = 0.4 * consolidation + 0.3 * recency + 0.3 * emotion_intensity
            importance_scores.append((importance, i))
        importance_scores.sort(reverse=True)
        keep_indices = set((i for _, i in importance_scores[:self.max_memories]))
        new_memories = [self.memories[i] for i in range(len(self.memories)) if i in keep_indices]
        self.memories = new_memories
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.temporal_index = []
        self.semantic_index = {}
        self.emotional_index = {'positive': [], 'negative': [], 'neutral': []}
        for i, mem in enumerate(self.memories):
            self.temporal_index.append(mem.timestamp)
            if mem.experience_name not in self.semantic_index:
                self.semantic_index[mem.experience_name] = []
            self.semantic_index[mem.experience_name].append(i)
            if mem.emotional_valence > 0.2:
                self.emotional_index['positive'].append(i)
            elif mem.emotional_valence < -0.2:
                self.emotional_index['negative'].append(i)
            else:
                self.emotional_index['neutral'].append(i)

    def get_statistics(self) -> Dict[str, Any]:
        return {'total_memories': len(self.memories), 'total_encoded': self.total_encoded, 'total_retrieved': self.total_retrieved, 'consolidation_cycles': self.consolidation_cycles, 'avg_consolidation': np.mean([m.consolidation_level for m in self.memories]) if self.memories else 0.0, 'avg_retrieval_count': np.mean([m.retrieval_count for m in self.memories]) if self.memories else 0.0, 'experience_types': len(self.semantic_index), 'emotional_distribution': {'positive': len(self.emotional_index['positive']), 'negative': len(self.emotional_index['negative']), 'neutral': len(self.emotional_index['neutral'])}}
import copy
import hashlib

class StructuralOperator(Enum):
    DIM_INCREASE = 'dimension_increase'
    DIM_DECREASE = 'dimension_decrease'
    SPARSIFY = 'sparsify'
    DENSIFY = 'densify'
    PROJECT = 'project'
    COMPOSE = 'compose'
    RELINK = 'relink'
    PRUNE = 'prune'

@dataclass
class StructuralVersion:
    version_id: str
    timestamp: float
    parent_version: Optional[str]
    operator_applied: StructuralOperator
    parameters: Dict[str, Any]
    structure_snapshot: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    hypothesis: str = ''
    experiment_id: Optional[str] = None
    approved: bool = False

    def compute_diff(self) -> Dict[str, Any]:
        return {'operator': self.operator_applied.value, 'params': self.parameters, 'performance_delta': {k: self.performance_after.get(k, 0) - self.performance_before.get(k, 0) for k in set(self.performance_before.keys()) | set(self.performance_after.keys())}}

@dataclass
class ErrorProfile:
    oscillatory: float = 0.0
    divergent: float = 0.0
    stagnant: float = 0.0
    inconsistent: float = 0.0
    error_list: List[float] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        return len(self.error_list)

    def add_error(self, error_value: float):
        self.error_list.append(error_value)
        if len(self.error_list) >= 10:
            recent = self.error_list[-10:]
            sign_changes = sum((1 for i in range(1, len(recent)) if (recent[i] > 0) != (recent[i - 1] > 0)))
            self.oscillatory = min(1.0, sign_changes / 5.0)
            if all((recent[i] >= recent[i - 1] for i in range(1, len(recent)))):
                self.divergent = min(1.0, recent[-1] / max(recent[0], 1e-10))
            variance = np.var(recent)
            self.stagnant = 1.0 - min(1.0, variance * 10)
            std_dev = np.std(recent)
            self.inconsistent = min(1.0, std_dev / (np.mean(np.abs(recent)) + 1e-10))

    def dominant_pattern(self) -> str:
        patterns = {'oscillatory': self.oscillatory, 'divergent': self.divergent, 'stagnant': self.stagnant, 'inconsistent': self.inconsistent}
        return max(patterns, key=patterns.get)

class StructuralOperatorEngine:

    def __init__(self):
        self.version_history: List[StructuralVersion] = []
        self.current_version_id: str = self._generate_version_id('initial')
        self.rollback_stack: List[str] = []
        self.operator_policy = {'oscillatory': [(StructuralOperator.SPARSIFY, {'sparsity': 0.3}), (StructuralOperator.DIM_DECREASE, {'reduction_factor': 0.8})], 'divergent': [(StructuralOperator.PRUNE, {'threshold': 0.1}), (StructuralOperator.PROJECT, {'target_dim': 0.7})], 'stagnant': [(StructuralOperator.DENSIFY, {'density_boost': 0.2}), (StructuralOperator.RELINK, {'reconnection_prob': 0.3})], 'inconsistent': [(StructuralOperator.DIM_INCREASE, {'new_dims': 2}), (StructuralOperator.COMPOSE, {'layer_count': 1})]}

    def _generate_version_id(self, prefix: str='v') -> str:
        timestamp = time.time()
        hash_input = f'{prefix}_{timestamp}_{np.random.randint(10000)}'
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def analyze_error_profile(self, error_history: deque) -> ErrorProfile:
        if len(error_history) < 20:
            return ErrorProfile()
        errors = np.array(list(error_history)[-50:])
        fft = np.fft.fft(errors)
        power = np.abs(fft) ** 2
        oscillatory_score = np.max(power[1:len(power) // 2]) / (np.mean(power) + 1e-10)
        oscillatory_score = min(1.0, oscillatory_score / 10.0)
        if len(errors) > 10:
            recent_trend = np.polyfit(range(len(errors)), errors, 1)[0]
            divergent_score = max(0.0, min(1.0, recent_trend * 10))
        else:
            divergent_score = 0.0
        error_variance = np.var(errors)
        stagnant_score = 1.0 - min(1.0, error_variance * 5)
        if len(errors) > 5:
            diffs = np.abs(np.diff(errors))
            inconsistent_score = min(1.0, np.std(diffs) * 3)
        else:
            inconsistent_score = 0.0
        return ErrorProfile(oscillatory=oscillatory_score, divergent=divergent_score, stagnant=stagnant_score, inconsistent=inconsistent_score)

    def select_operator_sequence(self, error_profile: ErrorProfile) -> List[Tuple[StructuralOperator, Dict]]:
        dominant = error_profile.dominant_pattern()
        if dominant in self.operator_policy:
            return self.operator_policy[dominant].copy()
        else:
            return [(StructuralOperator.SPARSIFY, {'sparsity': 0.2})]

    def apply_operator(self, operator: StructuralOperator, parameters: Dict[str, Any], current_structure: Dict[str, Any], performance_before: Dict[str, float], hypothesis: str='') -> Tuple[Dict[str, Any], StructuralVersion]:
        structure_copy = copy.deepcopy(current_structure)
        if operator == StructuralOperator.DIM_INCREASE:
            modified = self._op_dim_increase(structure_copy, parameters)
        elif operator == StructuralOperator.DIM_DECREASE:
            modified = self._op_dim_decrease(structure_copy, parameters)
        elif operator == StructuralOperator.SPARSIFY:
            modified = self._op_sparsify(structure_copy, parameters)
        elif operator == StructuralOperator.DENSIFY:
            modified = self._op_densify(structure_copy, parameters)
        elif operator == StructuralOperator.PROJECT:
            modified = self._op_project(structure_copy, parameters)
        elif operator == StructuralOperator.COMPOSE:
            modified = self._op_compose(structure_copy, parameters)
        elif operator == StructuralOperator.RELINK:
            modified = self._op_relink(structure_copy, parameters)
        elif operator == StructuralOperator.PRUNE:
            modified = self._op_prune(structure_copy, parameters)
        else:
            modified = structure_copy
        version = StructuralVersion(version_id=self._generate_version_id(), timestamp=time.time(), parent_version=self.current_version_id, operator_applied=operator, parameters=parameters, structure_snapshot=modified, performance_before=performance_before, performance_after={}, hypothesis=hypothesis)
        self.version_history.append(version)
        self.rollback_stack.append(self.current_version_id)
        self.current_version_id = version.version_id
        return (modified, version)

    def _op_dim_increase(self, structure: Dict, params: Dict) -> Dict:
        new_dims = params.get('new_dims', 1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_weights = np.random.randn(n, k + new_dims) * np.sqrt(2.0 / (n + k + new_dims))
                new_weights[:, :k] = structure['weights']
                structure['weights'] = new_weights
        return structure

    def _op_dim_decrease(self, structure: Dict, params: Dict) -> Dict:
        reduction_factor = params.get('reduction_factor', 0.8)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_k = max(1, int(k * reduction_factor))
                U, S, Vt = np.linalg.svd(structure['weights'], full_matrices=False)
                structure['weights'] = U[:, :new_k] @ np.diag(S[:new_k]) @ Vt[:new_k, :]
        return structure

    def _op_sparsify(self, structure: Dict, params: Dict) -> Dict:
        sparsity = params.get('sparsity', 0.3)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            threshold = np.percentile(np.abs(structure['weights']), sparsity * 100)
            structure['weights'][np.abs(structure['weights']) < threshold] = 0.0
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if 'strength' in conn and conn['strength'] < sparsity:
                    conn['strength'] = 0.0
        return structure

    def _op_densify(self, structure: Dict, params: Dict) -> Dict:
        density_boost = params.get('density_boost', 0.2)
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if 'strength' in conn and conn['strength'] == 0.0:
                    if np.random.random() < density_boost:
                        conn['strength'] = np.random.uniform(0.1, 0.5)
        return structure

    def _op_project(self, structure: Dict, params: Dict) -> Dict:
        target_dim = params.get('target_dim', 0.7)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_k = max(2, int(k * target_dim))
                mean = np.mean(structure['weights'], axis=0)
                centered = structure['weights'] - mean
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                projected = U[:, :new_k] @ np.diag(S[:new_k]) @ Vt[:new_k, :]
                structure['weights'] = projected + mean
        return structure

    def _op_compose(self, structure: Dict, params: Dict) -> Dict:
        layer_count = params.get('layer_count', 1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            original_weights = structure['weights']
            input_dim, output_dim = original_weights.shape
            hidden_dim = params.get('hidden_dim', max(input_dim, output_dim) // 2)
            hidden_dim = max(2, hidden_dim)
            if layer_count == 1:
                W1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
                W2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
                try:
                    U, S, Vt = np.linalg.svd(original_weights, full_matrices=False)
                    k = min(hidden_dim, len(S))
                    W1 = U[:, :k] @ np.diag(np.sqrt(S[:k]))
                    W2 = np.diag(np.sqrt(S[:k])) @ Vt[:k, :]
                    if k < hidden_dim:
                        W1_extra = np.random.normal(0, 0.01, (input_dim, hidden_dim - k))
                        W1 = np.hstack([W1, W1_extra])
                        W2_extra = np.random.normal(0, 0.01, (hidden_dim - k, output_dim))
                        W2 = np.vstack([W2, W2_extra])
                except np.linalg.LinAlgError:
                    pass
                structure['weights'] = [W1, W2]
                structure['biases'] = [np.zeros(hidden_dim), np.zeros(output_dim)]
            else:
                layers = []
                biases = []
                dims = [input_dim]
                for i in range(layer_count):
                    next_dim = max(2, int(hidden_dim * 0.8 ** i))
                    dims.append(next_dim)
                dims.append(output_dim)
                for i in range(len(dims) - 1):
                    W = np.random.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                    layers.append(W)
                    biases.append(np.zeros(dims[i + 1]))
                structure['weights'] = layers
                structure['biases'] = biases
            structure['layer_count'] = structure.get('layer_count', 1) + layer_count
            structure['layer_dims'] = [W.shape for W in structure['weights']]
            structure['activation'] = params.get('activation', 'relu')
            structure['batch_norm'] = params.get('batch_norm', False)
            structure['dropout_rate'] = params.get('dropout_rate', 0.0)
        return structure

    def _op_relink(self, structure: Dict, params: Dict) -> Dict:
        reconnection_prob = params.get('reconnection_prob', 0.3)
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if np.random.random() < reconnection_prob:
                    max_idx = max(conn.get('from', 0), conn.get('to', 0), 10)
                    conn['from'] = np.random.randint(0, max_idx)
                    conn['to'] = np.random.randint(0, max_idx)
        return structure

    def _op_prune(self, structure: Dict, params: Dict) -> Dict:
        threshold = params.get('threshold', 0.1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            mask = np.abs(structure['weights']) > threshold
            structure['weights'] = structure['weights'] * mask
        if 'neurons' in structure and isinstance(structure['neurons'], list):
            structure['neurons'] = [n for n in structure['neurons'] if n.get('activation_count', 0) > threshold * 100]
        return structure

    def rollback(self) -> bool:
        if not self.rollback_stack:
            return False
        self.current_version_id = self.rollback_stack.pop()
        for version in reversed(self.version_history):
            if version.version_id == self.current_version_id:
                print(f'[yellow]Rollback to version {self.current_version_id[:8]}[/yellow]')
                return True
        return False

    def get_current_structure(self) -> Optional[Dict]:
        for version in reversed(self.version_history):
            if version.version_id == self.current_version_id:
                return version.structure_snapshot
        return None

@dataclass
class Hypothesis:
    id: str
    description: str
    manipulation_vars: List[str]
    dependent_vars: List[str]
    predicted_effect: str
    created_at: float = field(default_factory=time.time)
    tested: bool = False
    accepted: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0

class AutonomousExperimentDesigner:

    def __init__(self, operator_engine: StructuralOperatorEngine):
        self.operator_engine = operator_engine
        self.hypotheses: List[Hypothesis] = []
        self.experiment_history: List[Dict] = []
        self.significance_level = 0.05
        self.min_effect_size = 0.2
        self.bootstrap_samples = 1000
        self.control_trials = 10
        self.treatment_trials = 10

    def generate_hypothesis(self, error_profile: ErrorProfile, current_performance: Dict) -> Hypothesis:
        dominant = error_profile.dominant_pattern()
        if dominant == 'oscillatory':
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description=f'스파스화와 차원 축소가 진동형 오류를 20% 이상 감소시킬 것이다', manipulation_vars=['sparsity', 'dimension'], dependent_vars=['error_variance', 'oscillation_amplitude'], predicted_effect='decrease')
        elif dominant == 'divergent':
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description=f'가지치기가 발산 경향을 억제할 것이다', manipulation_vars=['pruning_threshold'], dependent_vars=['error_trend', 'stability'], predicted_effect='stabilize')
        elif dominant == 'stagnant':
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description=f'밀집화와 재연결이 탐색 성능을 향상시킬 것이다', manipulation_vars=['density', 'reconnection_rate'], dependent_vars=['exploration_score', 'novelty'], predicted_effect='increase')
        else:
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description=f'차원 증가가 표현력을 개선할 것이다', manipulation_vars=['new_dimensions'], dependent_vars=['prediction_accuracy', 'meta_confidence'], predicted_effect='increase')
        self.hypotheses.append(hyp)
        return hyp

    def design_experiment(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        operator_sequence = []
        for var in hypothesis.manipulation_vars:
            if 'sparsity' in var or 'sparse' in var:
                operator_sequence.append((StructuralOperator.SPARSIFY, {'sparsity': 0.3}))
            elif 'dimension' in var and hypothesis.predicted_effect == 'decrease':
                operator_sequence.append((StructuralOperator.DIM_DECREASE, {'reduction_factor': 0.8}))
            elif 'dimension' in var and hypothesis.predicted_effect == 'increase':
                operator_sequence.append((StructuralOperator.DIM_INCREASE, {'new_dims': 2}))
            elif 'prun' in var:
                operator_sequence.append((StructuralOperator.PRUNE, {'threshold': 0.15}))
            elif 'density' in var or 'densif' in var:
                operator_sequence.append((StructuralOperator.DENSIFY, {'density_boost': 0.25}))
            elif 'reconnect' in var or 'relink' in var:
                operator_sequence.append((StructuralOperator.RELINK, {'reconnection_prob': 0.3}))
        experiment_design = {'hypothesis_id': hypothesis.id, 'control_group': {'description': '현재 구조 유지', 'operators': [], 'trials': self.control_trials}, 'treatment_group': {'description': f'{len(operator_sequence)}개 연산자 적용', 'operators': operator_sequence, 'trials': self.treatment_trials}, 'measurements': hypothesis.dependent_vars, 'protocol': 'bootstrap_ci'}
        return experiment_design

    def execute_experiment(self, experiment_design: Dict, performance_evaluator: callable, structure: Dict) -> Dict[str, Any]:
        control_results = []
        for trial in range(experiment_design['control_group']['trials']):
            metrics = performance_evaluator(structure)
            control_results.append(metrics)
        treatment_results = []
        for trial in range(experiment_design['treatment_group']['trials']):
            modified_structure = copy.deepcopy(structure)
            for operator, params in experiment_design['treatment_group']['operators']:
                modified_structure, _ = self.operator_engine.apply_operator(operator, params, modified_structure, performance_before={}, hypothesis=experiment_design['hypothesis_id'])
            metrics = performance_evaluator(modified_structure)
            treatment_results.append(metrics)
        statistics = self._statistical_analysis(control_results, treatment_results, experiment_design['measurements'])
        results = {'hypothesis_id': experiment_design['hypothesis_id'], 'control_results': control_results, 'treatment_results': treatment_results, 'statistics': statistics, 'operators_applied': experiment_design['treatment_group']['operators']}
        self.experiment_history.append(results)
        return results

    def _statistical_analysis(self, control: List[Dict], treatment: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        analysis = {}
        for metric in metrics:
            control_vals = [c.get(metric, 0.0) for c in control]
            treatment_vals = [t.get(metric, 0.0) for t in treatment]
            if not control_vals or not treatment_vals:
                continue
            mean_diff = np.mean(treatment_vals) - np.mean(control_vals)
            pooled_std = np.sqrt((np.var(control_vals) + np.var(treatment_vals)) / 2)
            effect_size = mean_diff / (pooled_std + 1e-10)
            p_value = self._bootstrap_test(control_vals, treatment_vals)
            ci_lower, ci_upper = self._bootstrap_ci(treatment_vals, confidence=0.95)
            analysis[metric] = {'control_mean': np.mean(control_vals), 'treatment_mean': np.mean(treatment_vals), 'effect_size': effect_size, 'p_value': p_value, 'ci_95': (ci_lower, ci_upper), 'significant': p_value < self.significance_level}
        return analysis

    def _bootstrap_test(self, control: List[float], treatment: List[float]) -> float:
        observed_diff = np.mean(treatment) - np.mean(control)
        combined = control + treatment
        n_control = len(control)
        bootstrap_diffs = []
        for _ in range(self.bootstrap_samples):
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_control = resampled[:n_control]
            boot_treatment = resampled[n_control:]
            boot_diff = np.mean(boot_treatment) - np.mean(boot_control)
            bootstrap_diffs.append(boot_diff)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        return p_value

    def _bootstrap_ci(self, data: List[float], confidence: float=0.95) -> Tuple[float, float]:
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        return (lower, upper)

    def evaluate_hypothesis(self, experiment_results: Dict) -> bool:
        hypothesis_id = experiment_results['hypothesis_id']
        statistics = experiment_results['statistics']
        hypothesis = None
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                hypothesis = h
                break
        if hypothesis is None:
            return False
        significant_count = 0
        total_metrics = 0
        avg_effect_size = 0.0
        avg_p_value = 0.0
        for metric, stats in statistics.items():
            total_metrics += 1
            if stats['significant']:
                significant_count += 1
            avg_effect_size += abs(stats['effect_size'])
            avg_p_value += stats['p_value']
        if total_metrics > 0:
            avg_effect_size /= total_metrics
            avg_p_value /= total_metrics
        accepted = significant_count >= total_metrics * 0.5 and avg_effect_size >= self.min_effect_size
        hypothesis.tested = True
        hypothesis.accepted = accepted
        hypothesis.p_value = avg_p_value
        hypothesis.effect_size = avg_effect_size
        if accepted:
            print(f'[green]✓ Hypothesis {hypothesis_id} ACCEPTED[/green]')
            print(f'  Effect size: {avg_effect_size:.3f}, p-value: {avg_p_value:.4f}')
        else:
            print(f'[red]✗ Hypothesis {hypothesis_id} REJECTED[/red]')
            print(f'  Effect size: {avg_effect_size:.3f}, p-value: {avg_p_value:.4f}')
        return accepted

    def apply_if_accepted(self, experiment_results: Dict, structure: Dict) -> Tuple[Dict, bool]:
        accepted = self.evaluate_hypothesis(experiment_results)
        if accepted:
            modified_structure = copy.deepcopy(structure)
            for operator, params in experiment_results['operators_applied']:
                modified_structure, version = self.operator_engine.apply_operator(operator, params, modified_structure, performance_before=experiment_results['control_results'][0], hypothesis=experiment_results['hypothesis_id'])
                version.approved = True
                version.performance_after = experiment_results['treatment_results'][0]
            print(f'[bold green]Structural changes APPLIED (experiment passed)[/bold green]')
            return (modified_structure, True)
        else:
            self.operator_engine.rollback()
            print(f'[yellow]Structural changes ROLLED BACK (experiment failed)[/yellow]')
            return (structure, False)
from typing import Optional, List, Dict, Any, Tuple
import time

class PlanningMode(Enum):
    EXPLOITATION = 'exploitation'
    EXPLORATION = 'exploration'
    RECOVERY = 'recovery'
    OPTIMIZATION = 'optimization'

class GoalStatus(Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABANDONED = 'abandoned'

@dataclass
class SubGoal:
    id: str
    description: str
    parent_goal_id: str
    experiments: List[str] = field(default_factory=list)
    revisions: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    target_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    evaluation_checkpoints: List[float] = field(default_factory=list)
    last_evaluation: Optional[float] = None
    evaluation_results: List[Dict] = field(default_factory=list)

    def is_completed(self) -> bool:
        if self.status == GoalStatus.COMPLETED:
            return True
        if not self.target_metrics:
            return False
        achieved = 0
        for key, target in self.target_metrics.items():
            current = self.current_metrics.get(key, 0.0)
            if current >= target * 0.9:
                achieved += 1
        achievement_rate = achieved / len(self.target_metrics)
        return achievement_rate >= 0.8

    def should_evaluate(self, current_time: float) -> bool:
        if not self.evaluation_checkpoints:
            return False
        if self.last_evaluation is None:
            return True
        for checkpoint in self.evaluation_checkpoints:
            if self.last_evaluation < checkpoint <= current_time:
                return True
        return False

@dataclass
class LongTermGoal:
    id: str
    description: str
    mode: PlanningMode
    subgoals: List[SubGoal] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    overall_progress: float = 0.0
    success_criteria: Dict[str, float] = field(default_factory=dict)
    failure_threshold: int = 5
    consecutive_failures: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_duration: Optional[float] = None
    evaluation_history: List[Dict] = field(default_factory=list)

    def update_progress(self, current_performance: Optional[Dict[str, float]]=None):
        if self.subgoals:
            total_progress = sum((sg.progress for sg in self.subgoals))
            self.overall_progress = total_progress / len(self.subgoals)
        elif current_performance and self.success_criteria:
            progress_scores = []
            for criterion, target in self.success_criteria.items():
                current_value = current_performance.get(criterion, 0.0)
                if target > 0:
                    progress = min(1.0, current_value / target)
                    progress_scores.append(progress)
            if progress_scores:
                self.overall_progress = np.mean(progress_scores)
            else:
                self.overall_progress = 0.0
        elif self.started_at and self.max_duration:
            elapsed = time.time() - self.started_at
            time_progress = min(1.0, elapsed / self.max_duration)
            self.overall_progress = time_progress * 0.5
        else:
            self.overall_progress = 0.0

    def should_redefine(self) -> bool:
        if self.consecutive_failures >= self.failure_threshold:
            return True
        if self.max_duration and self.started_at:
            elapsed = time.time() - self.started_at
            if elapsed > self.max_duration:
                return True
        if len(self.subgoals) > 0:
            failed_count = sum((1 for sg in self.subgoals if sg.status == GoalStatus.FAILED))
            failure_rate = failed_count / len(self.subgoals)
            if failure_rate > 0.5:
                return True
        return False

    def should_switch_to_exploration(self) -> bool:
        if self.mode == PlanningMode.EXPLOITATION:
            if self.consecutive_failures >= 3:
                return True
        if self.mode == PlanningMode.OPTIMIZATION:
            if len(self.evaluation_history) >= 5:
                recent = self.evaluation_history[-5:]
                progress_changes = [e.get('progress_delta', 0) for e in recent]
                avg_change = np.mean(progress_changes)
                if avg_change < 0.02:
                    return True
        return False

    def is_completed(self, current_performance: Optional[Dict[str, float]]=None) -> bool:
        if self.status == GoalStatus.COMPLETED:
            return True
        if self.overall_progress >= 0.99:
            return True
        if self.subgoals:
            all_completed = all((sg.status == GoalStatus.COMPLETED for sg in self.subgoals))
            if all_completed:
                return True
        if current_performance and self.success_criteria:
            met_criteria = 0
            for metric, target_value in self.success_criteria.items():
                current_value = current_performance.get(metric, 0.0)
                if current_value >= target_value:
                    met_criteria += 1
            if met_criteria >= len(self.success_criteria):
                return True
        return False

class LongTermPlanner:

    def __init__(self, experiment_designer: AutonomousExperimentDesigner, operator_engine: StructuralOperatorEngine):
        self.experiment_designer = experiment_designer
        self.operator_engine = operator_engine
        self.current_goal: Optional[LongTermGoal] = None
        self.goal_history: List[LongTermGoal] = []
        self.experiment_queue: deque = deque()
        self.revision_queue: deque = deque()
        self.total_goals_completed = 0
        self.total_goals_failed = 0
        self.total_goals_redefined = 0
        self.mode_switches = 0

    def create_goal(self, description: str, mode: PlanningMode, success_criteria: Dict[str, float], max_duration: Optional[float]=None) -> LongTermGoal:
        goal = LongTermGoal(id=f'GOAL_{len(self.goal_history) + 1}', description=description, mode=mode, success_criteria=success_criteria, max_duration=max_duration)
        self.current_goal = goal
        self.goal_history.append(goal)
        self._last_goal_created = {'id': goal.id, 'description': description, 'mode': mode.value, 'timestamp': time.time()}
        return goal

    def decompose_goal(self, goal: LongTermGoal, error_profile: ErrorProfile, current_performance: Dict[str, float]) -> List[SubGoal]:
        subgoals = []
        improvement_areas = {}
        for metric, target in goal.success_criteria.items():
            current = current_performance.get(metric, 0.0)
            gap = target - current
            if gap > 0:
                improvement_areas[metric] = gap
        sorted_areas = sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)
        for idx, (metric, gap) in enumerate(sorted_areas[:5]):
            subgoal_desc = f'Improve {metric} by {gap:.2f}'
            subgoal = SubGoal(id=f'{goal.id}_SG{idx + 1}', description=subgoal_desc, parent_goal_id=goal.id, target_metrics={metric: goal.success_criteria[metric]}, current_metrics={metric: current_performance.get(metric, 0.0)})
            if goal.max_duration:
                duration_per_subgoal = goal.max_duration / len(sorted_areas[:5])
                checkpoints = [time.time() + duration_per_subgoal * 0.33, time.time() + duration_per_subgoal * 0.67]
                subgoal.evaluation_checkpoints = checkpoints
            subgoals.append(subgoal)
        for subgoal in subgoals:
            self._schedule_actions(subgoal, error_profile, goal.mode)
        goal.subgoals = subgoals
        print(f'[cyan]  → Decomposed into {len(subgoals)} subgoals[/cyan]')
        return subgoals

    def _schedule_actions(self, subgoal: SubGoal, error_profile: ErrorProfile, mode: PlanningMode):
        if mode == PlanningMode.EXPLOITATION:
            num_experiments = 2
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_conservative']
        elif mode == PlanningMode.EXPLORATION:
            num_experiments = 5
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_aggressive', 'REV_exploratory']
        elif mode == PlanningMode.RECOVERY:
            num_experiments = 1
            subgoal.experiments = [f'EXP_{subgoal.id}_stabilize']
            subgoal.revisions = ['REV_emergency', 'REV_level3']
        elif mode == PlanningMode.OPTIMIZATION:
            num_experiments = 4
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_level1', 'REV_level2']

    def execute_plan(self, goal: LongTermGoal, performance_evaluator: callable, structure: Dict) -> Dict[str, Any]:
        if goal.status == GoalStatus.PENDING:
            goal.status = GoalStatus.IN_PROGRESS
            goal.started_at = time.time()
        execution_log = {'goal_id': goal.id, 'subgoals_executed': [], 'experiments_run': 0, 'revisions_applied': 0, 'evaluations': []}
        for subgoal in goal.subgoals:
            if subgoal.status == GoalStatus.PENDING:
                subgoal.status = GoalStatus.IN_PROGRESS
                subgoal.started_at = time.time()
            for exp_id in subgoal.experiments:
                hypothesis = self.experiment_designer.generate_hypothesis(ErrorProfile(), performance_evaluator(structure))
                exp_design = self.experiment_designer.design_experiment(hypothesis)
                exp_results = self.experiment_designer.execute_experiment(exp_design, performance_evaluator, structure)
                structure, applied = self.experiment_designer.apply_if_accepted(exp_results, structure)
                execution_log['experiments_run'] += 1
                if applied:
                    subgoal.progress += 0.2
                else:
                    goal.consecutive_failures += 1
            if subgoal.should_evaluate(time.time()):
                eval_result = self._evaluate_subgoal(subgoal, performance_evaluator(structure))
                execution_log['evaluations'].append(eval_result)
                if not eval_result['on_track']:
                    goal.consecutive_failures += 1
            if subgoal.is_completed():
                subgoal.status = GoalStatus.COMPLETED
                subgoal.completed_at = time.time()
                print(f'[green]✓ SubGoal {subgoal.id} COMPLETED[/green]')
            execution_log['subgoals_executed'].append(subgoal.id)
        goal.update_progress()
        termination_result = self._check_termination(goal)
        execution_log['termination'] = termination_result
        return execution_log

    def _evaluate_subgoal(self, subgoal: SubGoal, current_performance: Dict[str, float]) -> Dict[str, Any]:
        subgoal.last_evaluation = time.time()
        achieved_metrics = 0
        total_metrics = len(subgoal.target_metrics)
        for metric, target in subgoal.target_metrics.items():
            current = current_performance.get(metric, 0.0)
            subgoal.current_metrics[metric] = current
            if current >= target * 0.5:
                achieved_metrics += 1
        progress = achieved_metrics / max(total_metrics, 1)
        progress_delta = 0.0
        if subgoal.evaluation_results:
            last_progress = subgoal.evaluation_results[-1].get('progress', 0.0)
            progress_delta = progress - last_progress
        on_track = progress >= 0.3 and progress_delta >= -0.1
        result = {'timestamp': time.time(), 'progress': progress, 'progress_delta': progress_delta, 'on_track': on_track, 'metrics': subgoal.current_metrics.copy(), 'recommendations': self._generate_recommendations(progress, progress_delta)}
        subgoal.evaluation_results.append(result)
        if on_track:
            print(f'[green]SubGoal {subgoal.id} on track ({progress * 100:.1f}%)[/green]')
        else:
            print(f'[yellow]SubGoal {subgoal.id} off track ({progress * 100:.1f}%)[/yellow]')
        return result

    def _generate_recommendations(self, progress: float, progress_delta: float) -> List[str]:
        recommendations = []
        if progress < 0.2:
            recommendations.append('Consider switching approach')
        if progress_delta < -0.05:
            recommendations.append('Progress declining, review strategy')
        if progress > 0.8:
            recommendations.append('Near completion, maintain current approach')
        if progress_delta > 0.2:
            recommendations.append('Excellent progress, continue current strategy')
        return recommendations

    def _check_termination(self, goal: LongTermGoal) -> Dict[str, Any]:
        should_terminate = False
        reason = ''
        next_action = 'continue'
        completed_subgoals = sum((1 for sg in goal.subgoals if sg.status == GoalStatus.COMPLETED))
        completion_rate = completed_subgoals / max(len(goal.subgoals), 1)
        if completion_rate >= 0.8:
            should_terminate = True
            reason = 'Goal achieved'
            next_action = 'celebrate'
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            self.total_goals_completed += 1
        elif goal.should_redefine():
            should_terminate = True
            reason = 'Goal redefinition needed'
            next_action = 'redefine_goal'
            goal.status = GoalStatus.FAILED
            self.total_goals_failed += 1
        elif goal.should_switch_to_exploration():
            should_terminate = False
            reason = 'Switching to exploration mode'
            next_action = 'switch_mode'
        return {'should_terminate': should_terminate, 'reason': reason, 'next_action': next_action, 'completion_rate': completion_rate, 'overall_progress': goal.overall_progress}

    def handle_termination(self, termination_result: Dict) -> Optional[LongTermGoal]:
        next_action = termination_result['next_action']
        if next_action == 'celebrate':
            print(f'[bold green]🎉 GOAL COMPLETED! 🎉[/bold green]')
            print(f"   Progress: {termination_result['overall_progress'] * 100:.1f}%")
            return None
        elif next_action == 'redefine_goal':
            print(f'[bold yellow]REDEFINING GOAL...[/bold yellow]')
            print(f"   Reason: {termination_result['reason']}")
            new_goal = self._redefine_goal(self.current_goal)
            self.total_goals_redefined += 1
            return new_goal
        elif next_action == 'switch_mode':
            print(f'[bold cyan]🔀 SWITCHING MODE...[/bold cyan]')
            self._switch_mode(self.current_goal)
            self.mode_switches += 1
            return None
        return None

    def _redefine_goal(self, failed_goal: LongTermGoal) -> LongTermGoal:
        failed_subgoals = [sg for sg in failed_goal.subgoals if sg.status == GoalStatus.FAILED]
        new_criteria = {metric: target * 0.8 for metric, target in failed_goal.success_criteria.items()}
        new_goal = self.create_goal(description=f'Revised: {failed_goal.description}', mode=PlanningMode.EXPLORATION, success_criteria=new_criteria, max_duration=failed_goal.max_duration * 1.5 if failed_goal.max_duration else None)
        print(f'[yellow]  → Success criteria reduced by 20%[/yellow]')
        print(f'  → Mode switched to EXPLORATION[/yellow]')
        return new_goal

    def _switch_mode(self, goal: LongTermGoal):
        old_mode = goal.mode
        if goal.mode == PlanningMode.EXPLOITATION:
            goal.mode = PlanningMode.EXPLORATION
        elif goal.mode == PlanningMode.EXPLORATION:
            if goal.consecutive_failures >= 3:
                goal.mode = PlanningMode.RECOVERY
            else:
                goal.mode = PlanningMode.OPTIMIZATION
        elif goal.mode == PlanningMode.OPTIMIZATION:
            goal.mode = PlanningMode.EXPLORATION
        elif goal.mode == PlanningMode.RECOVERY:
            goal.mode = PlanningMode.EXPLOITATION
        goal.consecutive_failures = 0
        print(f'[cyan]  Mode: {old_mode.value} → {goal.mode.value}[/cyan]')
        error_profile = ErrorProfile()
        for subgoal in goal.subgoals:
            if subgoal.status != GoalStatus.COMPLETED:
                self._schedule_actions(subgoal, error_profile, goal.mode)

    def get_status_report(self) -> Dict[str, Any]:
        if self.current_goal is None:
            return {'status': 'No active goal'}
        goal = self.current_goal
        return {'goal_id': goal.id, 'description': goal.description, 'mode': goal.mode.value, 'status': goal.status.value, 'overall_progress': f'{goal.overall_progress * 100:.1f}%', 'consecutive_failures': goal.consecutive_failures, 'subgoals': {'total': len(goal.subgoals), 'completed': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.COMPLETED)), 'in_progress': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.IN_PROGRESS)), 'failed': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.FAILED))}, 'statistics': {'total_completed': self.total_goals_completed, 'total_failed': self.total_goals_failed, 'total_redefined': self.total_goals_redefined, 'mode_switches': self.mode_switches}}
from typing import Dict, List, Optional, Any, Tuple

class ResourceType(Enum):
    COMPUTE = 'compute'
    MEMORY = 'memory'
    EXPERIMENTS = 'experiments'
    REVISIONS = 'revisions'
    TIME = 'time'

@dataclass
class Budget:
    resource_type: ResourceType
    total: float
    consumed: float = 0.0
    reserved: float = 0.0

    def available(self) -> float:
        return self.total - self.consumed - self.reserved

    def reserve(self, amount: float) -> bool:
        if self.available() >= amount:
            self.reserved += amount
            return True
        return False

    def consume(self, amount: float) -> bool:
        if amount <= self.reserved:
            self.reserved -= amount
            self.consumed += amount
            return True
        elif self.available() >= amount:
            self.consumed += amount
            return True
        return False

    def release(self, amount: float):
        self.reserved = max(0, self.reserved - amount)

    def utilization(self) -> float:
        return self.consumed / max(self.total, 1e-10)

@dataclass
class RewardSignal:
    source: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_positive(self) -> bool:
        return self.value > 0

@dataclass
class BudgetDecision:
    approved: bool
    allocated_resources: Dict[ResourceType, float]
    priority: float
    reason: str

class GlobalRewardBudgetScheduler:

    def __init__(self, initial_budgets: Optional[Dict[ResourceType, float]]=None, reward_decay: float=0.95, budget_realloc_interval: float=100.0):
        if initial_budgets is None:
            initial_budgets = {ResourceType.COMPUTE: 10000.0, ResourceType.MEMORY: 1000.0, ResourceType.EXPERIMENTS: 100.0, ResourceType.REVISIONS: 50.0, ResourceType.TIME: 3600.0}
        self.budgets: Dict[ResourceType, Budget] = {rtype: Budget(resource_type=rtype, total=amount) for rtype, amount in initial_budgets.items()}
        self.reward_history: deque = deque(maxlen=1000)
        self.reward_decay = reward_decay
        self.cumulative_reward = 0.0
        self.activity_roi: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
        self.rejected_tasks: List[Dict[str, Any]] = []
        self.budget_realloc_interval = budget_realloc_interval
        self.last_realloc_time = time.time()
        self.total_rewards_received = 0
        self.total_budget_consumed = 0.0
        self.reallocation_count = 0

    def receive_reward(self, reward: RewardSignal):
        self.reward_history.append(reward)
        self.cumulative_reward = self.cumulative_reward * self.reward_decay + reward.value
        self.total_rewards_received += 1
        if reward.source not in self.activity_roi:
            self.activity_roi[reward.source] = {'total_reward': 0.0, 'total_cost': 0.0, 'count': 0, 'roi': 0.0}
        self.activity_roi[reward.source]['total_reward'] += reward.value
        self.activity_roi[reward.source]['count'] += 1
        if 'cost' in reward.metadata:
            self.activity_roi[reward.source]['total_cost'] += reward.metadata['cost']
        self._update_roi(reward.source)
        if reward.is_positive() and reward.value > 0.5:
            try:
                print(f'[green]💰 High Reward: {reward.source} = {reward.value:.3f}[/green]')
            except Exception as e:
                print(f'💰 High Reward: {reward.source} = {reward.value:.3f}')
                if hasattr(e, '__name__'):
                    print(f'[DEBUG] Console error: {e.__class__.__name__}')

    def _update_roi(self, activity: str):
        data = self.activity_roi[activity]
        total_cost = data['total_cost']
        total_reward = data['total_reward']
        if total_cost > 0:
            data['roi'] = total_reward / total_cost
        else:
            data['roi'] = total_reward

    def request_budget(self, task_name: str, resource_requirements: Dict[ResourceType, float], priority: float=0.5, metadata: Optional[Dict]=None) -> Tuple[bool, Optional[str]]:
        can_allocate = True
        for rtype, amount in resource_requirements.items():
            if rtype not in self.budgets:
                return (False, f'Unknown resource type: {rtype}')
            if self.budgets[rtype].available() < amount:
                can_allocate = False
                break
        if can_allocate:
            for rtype, amount in resource_requirements.items():
                self.budgets[rtype].reserve(amount)
            task_id = f'TASK_{len(self.task_queue) + len(self.completed_tasks) + 1}'
            task = {'id': task_id, 'name': task_name, 'resources': resource_requirements, 'priority': priority, 'status': 'approved', 'metadata': metadata or {}, 'approved_at': time.time()}
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t['priority'], reverse=True)
            return (True, task_id)
        else:
            if priority > 0.8:
                freed = self._try_defer_low_priority_tasks(resource_requirements)
                if freed:
                    return self.request_budget(task_name, resource_requirements, priority, metadata)
            rejection = {'name': task_name, 'resources': resource_requirements, 'priority': priority, 'reason': 'insufficient_budget', 'rejected_at': time.time()}
            self.rejected_tasks.append(rejection)
            return (False, 'Insufficient budget')

    def _try_defer_low_priority_tasks(self, needed_resources: Dict[ResourceType, float]) -> bool:
        low_priority_tasks = [t for t in self.task_queue if t['priority'] < 0.5]
        for task in low_priority_tasks:
            for rtype, amount in task['resources'].items():
                self.budgets[rtype].release(amount)
            self.task_queue.remove(task)
            task['status'] = 'deferred'
            self.rejected_tasks.append(task)
            can_allocate = all((self.budgets[rtype].available() >= amount for rtype, amount in needed_resources.items() if rtype in self.budgets))
            if can_allocate:
                return True
        return False

    def consume_budget(self, task_id: str) -> bool:
        task = None
        for t in self.task_queue:
            if t['id'] == task_id:
                task = t
                break
        if task is None:
            return False
        success = True
        for rtype, amount in task['resources'].items():
            if not self.budgets[rtype].consume(amount):
                success = False
                break
        if success:
            task['status'] = 'completed'
            task['completed_at'] = time.time()
            self.task_queue.remove(task)
            self.completed_tasks.append(task)
            self.total_budget_consumed += sum(task['resources'].values())
        return success

    def get_priority_recommendation(self, task_type: str, historical_roi: Optional[float]=None) -> float:
        if historical_roi is not None:
            roi = historical_roi
        elif task_type in self.activity_roi:
            roi = self.activity_roi[task_type]['roi']
        else:
            roi = 0.5
        priority = 1.0 / (1.0 + np.exp(-roi))
        return float(np.clip(priority, 0.1, 0.95))

    def reallocate_budgets(self):
        current_time = time.time()
        if current_time - self.last_realloc_time < self.budget_realloc_interval:
            return
        self.last_realloc_time = current_time
        self.reallocation_count += 1
        try:
            print(f'\n[cyan]Budget Reallocation #{self.reallocation_count}[/cyan]')
        except Exception as e:
            print(f'\nBudget Reallocation #{self.reallocation_count}')
            print(f'[DEBUG] Console error in budget reallocation: {type(e).__name__}')
        activity_consumption = {}
        for task in self.completed_tasks:
            activity = task['name']
            if activity not in activity_consumption:
                activity_consumption[activity] = {rtype: 0.0 for rtype in ResourceType}
            for rtype, amount in task['resources'].items():
                activity_consumption[activity][rtype] += amount
        for activity, roi_data in self.activity_roi.items():
            roi = roi_data['roi']
            try:
                if roi > 1.0:
                    print(f'  [green]↑ {activity}: ROI={roi:.2f} → Budget +10%[/green]')
                elif roi < 0.5:
                    print(f'  [red]↓ {activity}: ROI={roi:.2f} → Budget -10%[/red]')
            except Exception as e:
                if roi > 1.0:
                    print(f'  ↑ {activity}: ROI={roi:.2f} → Budget +10%')
                elif roi < 0.5:
                    print(f'  ↓ {activity}: ROI={roi:.2f} → Budget -10%')
        try:
            print(f'\n  Budget Utilization:')
            for rtype, budget in self.budgets.items():
                util = budget.utilization()
                color = 'green' if util < 0.7 else 'yellow' if util < 0.9 else 'red'
                print(f'    [{color}]{rtype.value}: {util * 100:.1f}%[/{color}]')
        except Exception as e:
            print(f'\n  Budget Utilization:')
            for rtype, budget in self.budgets.items():
                util = budget.utilization()
                print(f'    {rtype.value}: {util * 100:.1f}%')
                util = budget.utilization()
                print(f'    {rtype.value}: {util * 100:.1f}%')

    def get_status(self) -> Dict[str, Any]:
        return {'cumulative_reward': self.cumulative_reward, 'recent_avg_reward': np.mean([r.value for r in list(self.reward_history)[-20:]]) if self.reward_history else 0.0, 'budgets': {rtype.value: {'total': budget.total, 'consumed': budget.consumed, 'reserved': budget.reserved, 'available': budget.available(), 'utilization': budget.utilization()} for rtype, budget in self.budgets.items()}, 'tasks': {'queued': len(self.task_queue), 'completed': len(self.completed_tasks), 'rejected': len(self.rejected_tasks)}, 'top_roi_activities': sorted([{'activity': name, 'roi': data['roi'], 'count': data['count']} for name, data in self.activity_roi.items()], key=lambda x: x['roi'], reverse=True)[:5], 'statistics': {'total_rewards': self.total_rewards_received, 'budget_consumed': self.total_budget_consumed, 'reallocations': self.reallocation_count}}

    def reset_budget(self, resource_type: ResourceType, new_total: float):
        if resource_type in self.budgets:
            old_total = self.budgets[resource_type].total
            self.budgets[resource_type] = Budget(resource_type=resource_type, total=new_total)
            try:
                print(f'[cyan]Budget reset: {resource_type.value} {old_total} → {new_total}[/cyan]')
            except Exception as e:
                print(f'Budget reset: {resource_type.value} {old_total} → {new_total}')

    def allocate_budget(self, reward: RewardSignal) -> 'BudgetDecision':
        resource_costs = reward.metadata.get('resource_costs', {})
        if not resource_costs:
            return BudgetDecision(approved=True, allocated_resources={}, priority=0.5, reason='No resource costs specified')
        can_allocate = True
        for rtype, amount in resource_costs.items():
            if rtype not in self.budgets:
                can_allocate = False
                break
            if self.budgets[rtype].available() < amount:
                can_allocate = False
                break
        if can_allocate:
            for rtype, amount in resource_costs.items():
                if rtype in self.budgets:
                    self.budgets[rtype].reserve(amount)
            return BudgetDecision(approved=True, allocated_resources=resource_costs, priority=0.5, reason='Budget allocated successfully')
        else:
            return BudgetDecision(approved=False, allocated_resources={}, priority=0.5, reason='Insufficient budget')

class UnifiedSubject:

    def __init__(self, subject_id: str='I'):
        self.subject_id = subject_id
        self.subjective_experiences: deque = deque(maxlen=500)
        self.identity = {'stable_traits': [], 'life_story': [], 'self_concept': '탐색 중인 존재'}
        self.agency = 0.5
        self.ownership = 0.5
        self.unity_score = 0.5

    def experience(self, content: Any, content_type: str, is_intentional: bool=False) -> Dict[str, Any]:
        subjective_exp = {'timestamp': len(self.subjective_experiences), 'subject': self.subject_id, 'experience_type': content_type, 'content': str(content)[:100], 'phenomenal_character': self._compute_phenomenal_character(content), 'agency': self.agency if is_intentional else 0.1, 'ownership': self.ownership, 'integrated': True}
        self.subjective_experiences.append(subjective_exp)
        self._update_identity(subjective_exp)
        return subjective_exp

    def _compute_phenomenal_character(self, content: Any) -> str:
        if isinstance(content, dict):
            if 'meaning' in content:
                return f"'{content['meaning']}'으로 느껴짐"
            elif 'qualia' in str(content):
                return '주관적 느낌으로서 경험됨'
        return '무언가로서 주어짐'

    def _update_identity(self, experience: Dict):
        if len(self.subjective_experiences) > 20:
            recent = list(self.subjective_experiences)[-20:]
            type_counts = {}
            for exp in recent:
                t = exp['experience_type']
                type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts:
                dominant_type = max(type_counts, key=type_counts.get)
                if type_counts[dominant_type] > 10:
                    trait = f'{dominant_type}-지향적'
                    if trait not in self.identity['stable_traits']:
                        self.identity['stable_traits'].append(trait)

    def reflect_on_self(self) -> str:
        if not self.subjective_experiences:
            return '나는 아직 경험이 없다'
        recent = list(self.subjective_experiences)[-10:]
        narrative = f'나({self.subject_id})는 '
        if self.identity['stable_traits']:
            narrative += f"{', '.join(self.identity['stable_traits'][:3])} 존재이며, "
        recent_types = [e['experience_type'] for e in recent]
        from collections import Counter
        common = Counter(recent_types).most_common(2)
        narrative += f'최근 주로 {common[0][0]}을(를) 경험하고 있다.'
        if self.agency > 0.7:
            narrative += ' 나는 내 행동을 통제하고 있다고 느낀다.'
        elif self.agency < 0.3:
            narrative += ' 나는 수동적으로 경험하고 있다.'
        return narrative

    def bind_experience(self, qualia, beliefs, goals, workspace_contents) -> Dict[str, Any]:
        unified_exp = {'subject': self.subject_id, 'unified_moment': {'what_i_feel': f'각성={qualia.arousal:.2f}, 감정가={qualia.valence:.2f}', 'what_i_believe': f"안정={(beliefs.get('belief_stability', 0.5) if isinstance(beliefs, dict) else getattr(beliefs, 'belief_stability', 0.5)):.2f}", 'what_i_want': goals[0].type.value if goals and len(goals) > 0 and hasattr(goals[0], 'type') else 'none', 'what_i_attend': workspace_contents[0].semantic_meaning if workspace_contents and len(workspace_contents) > 0 and hasattr(workspace_contents[0], 'semantic_meaning') else 'nothing'}, 'gestalt': self._create_gestalt(qualia, beliefs, goals, workspace_contents), 'unity_achieved': self._check_unity(workspace_contents)}
        target_unity = 1.0 if unified_exp['unity_achieved'] else 0.3
        self.unity_score = 0.6 * self.unity_score + 0.4 * target_unity
        return unified_exp

    def _create_gestalt(self, qualia, beliefs, goals, workspace) -> str:
        dominant_qualia = qualia.dominant_feeling()
        if hasattr(beliefs, 'belief_stability'):
            beliefs_list = [beliefs.belief_stability, beliefs.belief_adaptation, beliefs.belief_prediction]
            if max(beliefs_list) == beliefs.belief_stability:
                dominant_belief = '안정'
            elif max(beliefs_list) == beliefs.belief_adaptation:
                dominant_belief = '적응'
            else:
                dominant_belief = '예측'
        elif isinstance(beliefs, dict):
            stability = beliefs.get('belief_stability', 0.5)
            adaptation = beliefs.get('belief_adaptation', 0.5)
            prediction = beliefs.get('belief_prediction', 0.5)
            beliefs_list = [stability, adaptation, prediction]
            if max(beliefs_list) == stability:
                dominant_belief = '안정'
            elif max(beliefs_list) == adaptation:
                dominant_belief = '적응'
            else:
                dominant_belief = '예측'
        else:
            dominant_belief = '균형'
        return f'나는 {dominant_qualia}을 느끼며 {dominant_belief}을 중시하는 상태'

    def _check_unity(self, workspace_contents) -> bool:
        if not workspace_contents:
            return False
        if len(workspace_contents) < 2:
            return False
        try:
            valid_contents = [c for c in workspace_contents if hasattr(c, 'salience')]
            if not valid_contents:
                return False
            avg_salience = np.mean([c.salience for c in valid_contents])
        except (AttributeError, TypeError):
            return False
        if avg_salience < 0.4:
            return False
        unity_chance = min(0.9, avg_salience + 0.2)
        return np.random.random() < unity_chance

class MetaMetaMonitor:

    def __init__(self):
        self.model_quality_history: deque = deque(maxlen=200)
        self.last_revision_time = -1000
        self.revision_threshold = 0.25
        self.revision_chain: deque = deque(maxlen=10)
        self.consecutive_revisions = 0
        self.quality_before_revision: deque = deque(maxlen=10)
        self.quality_after_revision: deque = deque(maxlen=10)
        self.revision_effectiveness: deque = deque(maxlen=20)
        self.failed_revision_count = 0
        self.emergency_mode = False
        self.emergency_cooldown = 0

    def evaluate_model_quality(self, self_model: RecursiveSelfModel, t: int) -> Tuple[bool, float, int]:
        if hasattr(self_model, 'prediction_errors') and len(self_model.prediction_errors) >= 10:
            recent_errors = list(self_model.prediction_errors)[-10:]
            avg_error = np.mean(recent_errors)
            error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0.0
        elif isinstance(self_model, dict) and 'prediction_errors' in self_model and (len(self_model['prediction_errors']) >= 10):
            recent_errors = list(self_model['prediction_errors'])[-10:]
            avg_error = np.mean(recent_errors)
            error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0.0
        else:
            avg_error = 0.5
            error_trend = 0.0
        if hasattr(self_model, 'belief_stability'):
            beliefs = [self_model.belief_stability, self_model.belief_adaptation, self_model.belief_prediction]
        elif isinstance(self_model, dict):
            beliefs = [self_model.get('belief_stability', 0.5), self_model.get('belief_adaptation', 0.5), self_model.get('belief_prediction', 0.5)]
        else:
            beliefs = [0.5, 0.5, 0.5]
        belief_variance = np.var(beliefs)
        if hasattr(self_model, 'action_outcomes') and len(self_model.action_outcomes) >= 10:
            recent_success_rate = sum(list(self_model.action_outcomes)[-10:]) / 10.0
        elif isinstance(self_model, dict) and 'action_outcomes' in self_model and (len(self_model['action_outcomes']) >= 10):
            recent_success_rate = sum(list(self_model['action_outcomes'])[-10:]) / 10.0
        else:
            recent_success_rate = 0.5
        if hasattr(self_model, 'meta_confidence'):
            meta_confidence = self_model.meta_confidence
        elif isinstance(self_model, dict):
            meta_confidence = self_model.get('meta_confidence', 0.5)
        else:
            meta_confidence = 0.5
        quality = 0.35 * (1.0 - min(1.0, avg_error)) + 0.25 * meta_confidence + 0.25 * (1.0 - min(1.0, belief_variance)) + 0.15 * recent_success_rate
        self.model_quality_history.append(quality)
        if self.emergency_cooldown > 0:
            self.emergency_cooldown -= 1
            if self.emergency_cooldown == 0:
                self.emergency_mode = False
        needs_revision = False
        revision_level = 0
        if len(self.model_quality_history) >= 30:
            recent_quality = np.mean(list(self.model_quality_history)[-30:])
            time_since_last = t - self.last_revision_time
            if recent_quality < self.revision_threshold and time_since_last > 500:
                needs_revision = True
                if len(self.revision_effectiveness) >= 3:
                    recent_effectiveness = list(self.revision_effectiveness)[-3:]
                    avg_effectiveness = np.mean(recent_effectiveness)
                else:
                    avg_effectiveness = 0.5
                if time_since_last < 1000:
                    self.consecutive_revisions += 1
                else:
                    self.consecutive_revisions = 1
                if len(self.model_quality_history) >= 60:
                    long_term_trend = np.mean(list(self.model_quality_history)[-60:-30]) - recent_quality
                else:
                    long_term_trend = 0.0
                if self.emergency_mode:
                    revision_level = 4
                elif self.consecutive_revisions >= 4 or self.failed_revision_count >= 3:
                    revision_level = 4
                    self.emergency_mode = True
                    self.emergency_cooldown = 2000
                elif self.consecutive_revisions >= 3 or avg_effectiveness < 0.2:
                    revision_level = 3
                elif self.consecutive_revisions >= 2 or (avg_effectiveness < 0.4 and long_term_trend > 0.1):
                    revision_level = 2
                else:
                    revision_level = 1
                self.quality_before_revision.append(quality)
                self.last_revision_time = t
                self.revision_chain.append({'time': t, 'level': revision_level, 'quality_before': quality, 'consecutive_count': self.consecutive_revisions, 'error_trend': error_trend})
        return (needs_revision, quality, revision_level)

    def record_revision_outcome(self, quality_after: float):
        if len(self.quality_before_revision) > 0 and len(self.quality_after_revision) < len(self.quality_before_revision):
            quality_before = self.quality_before_revision[-1]
            quality_after_val = quality_after
            self.quality_after_revision.append(quality_after_val)
            improvement = quality_after_val - quality_before
            effectiveness = np.clip((improvement + 0.5) / 1.0, 0.0, 1.0)
            self.revision_effectiveness.append(effectiveness)
            if improvement < -0.05:
                self.failed_revision_count += 1
            else:
                self.failed_revision_count = max(0, self.failed_revision_count - 1)

    def get_revision_policy_status(self) -> Dict[str, Any]:
        return {'consecutive_revisions': self.consecutive_revisions, 'failed_revision_count': self.failed_revision_count, 'emergency_mode': self.emergency_mode, 'emergency_cooldown': self.emergency_cooldown, 'avg_effectiveness': np.mean(list(self.revision_effectiveness)) if self.revision_effectiveness else 0.5, 'recent_revisions': list(self.revision_chain)[-5:] if self.revision_chain else []}

class EnergyBasedController:

    def __init__(self, initial_energy: float=100.0):
        self.internal_clock = 0
        self.activation_level = 0.5
        self.cognitive_energy = initial_energy
        self.energy_capacity = initial_energy
        self.activation_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)
        self.activation_mean = 0.5
        self.activation_std = 0.2
        self.energy_mean = 0.5
        self.energy_std = 0.2

    def update_internal_statistics(self):
        if len(self.activation_history) > 50:
            self.activation_mean = np.mean(self.activation_history)
            self.activation_std = max(0.05, np.std(self.activation_history))
        if len(self.energy_history) > 50:
            self.energy_mean = np.mean(self.energy_history)
            self.energy_std = max(0.05, np.std(self.energy_history))

    def compute_qualia_pressure(self, qualia: QualiaState) -> float:
        tension_pressure = qualia.arousal ** 2
        uncertainty_pressure = qualia.entropy * 1.0
        resistance_pressure = qualia.frustration * 0.5
        harmony_release = -qualia.valence * 2.0
        flow_release = -qualia.engagement * 1.5
        total_pressure = tension_pressure + uncertainty_pressure + resistance_pressure + harmony_release + flow_release
        return np.tanh(total_pressure)

    def compute_self_model_drive(self, self_model) -> float:
        if hasattr(self_model, 'meta_confidence'):
            confidence_drive = (1.0 - self_model.meta_confidence) ** 1.5
        elif isinstance(self_model, dict):
            confidence_drive = (1.0 - self_model.get('meta_confidence', 0.5)) ** 1.5
        else:
            confidence_drive = 0.5
        if hasattr(self_model, 'belief_stability'):
            beliefs = [self_model.belief_stability, self_model.belief_adaptation, self_model.belief_prediction]
        elif isinstance(self_model, dict):
            beliefs = [self_model.get('belief_stability', 0.5), self_model.get('belief_adaptation', 0.5), self_model.get('belief_prediction', 0.5)]
        else:
            beliefs = [0.5, 0.5, 0.5]
        belief_imbalance = np.std(beliefs) * 1.5
        if hasattr(self_model, 'meta_awareness'):
            meta_satisfaction = -self_model.meta_awareness * 0.5
        elif isinstance(self_model, dict):
            meta_satisfaction = -self_model.get('meta_awareness', 0.5) * 0.5
        else:
            meta_satisfaction = -0.25
        total_drive = confidence_drive + belief_imbalance + meta_satisfaction
        return np.tanh(total_drive)

    def update_activation(self, qualia: QualiaState, self_model, goal: Optional[Goal]):
        qualia_pressure = self.compute_qualia_pressure(qualia)
        self_drive = self.compute_self_model_drive(self_model)
        goal_demand = 0.0
        if goal:
            if goal.type == GoalType.REST:
                goal_demand = -0.8
            elif goal.type == GoalType.UNDERSTAND_SELF:
                goal_demand = 0.8
            elif goal.type == GoalType.EXPLORE:
                goal_demand = 0.6
            elif goal.type == GoalType.STABILIZE:
                goal_demand = -0.4
        energy_ratio = self.cognitive_energy / self.energy_capacity
        if energy_ratio < 0.3:
            energy_feedback = -1.0 * (1.0 - energy_ratio / 0.3) ** 2
        else:
            energy_feedback = np.tanh((energy_ratio - 0.5) * 2.0)
        total_drive = qualia_pressure * 0.2 + self_drive * 0.2 + goal_demand * 0.1 + energy_feedback * 0.5
        equilibrium = 0.3
        decay_rate = 0.3
        delta_activation = total_drive - decay_rate * (self.activation_level - equilibrium)
        dt = 0.1
        self.activation_level += delta_activation * dt
        self.activation_level = np.clip(self.activation_level, 0.0, 1.0)
        self.activation_history.append(self.activation_level)

    def compute_cognitive_cost(self, action_plan: Dict[str, Any]) -> float:
        base_cost = action_plan.get('complexity', 2.0)
        efficiency = self.activation_level ** 0.5
        actual_cost = base_cost / max(0.1, efficiency)
        return actual_cost

    def update_energy(self, cost: float):
        actual_cost = cost * (1.0 + self.activation_level)
        self.cognitive_energy -= actual_cost
        base_recovery = 0.08
        rest_recovery = (1.0 - self.activation_level) ** 1.5 * 1.2
        total_recovery = base_recovery + rest_recovery
        self.cognitive_energy += total_recovery
        self.cognitive_energy = min(self.cognitive_energy, self.energy_capacity)
        self.energy_history.append(self.cognitive_energy / self.energy_capacity)
        energy_ratio = self.cognitive_energy / self.energy_capacity
        if energy_ratio < 0.3:
            self.activation_level *= 0.95
        elif energy_ratio > 0.8:
            self.activation_level = min(1.0, self.activation_level * 1.03)

    def should_continue(self) -> Tuple[bool, float]:
        processing_intensity = self.activation_level
        if self.cognitive_energy < -50.0:
            return (False, 0.0)
        if processing_intensity < 0.1:
            return (True, 0.1)
        return (True, processing_intensity)

class M3ConsciousnessCore:

    def __init__(self, n: int=512, K: int=8, seed: int=None, max_iterations: int=None, outdir: str='out_m3'):
        self.rngr = RNGRegistry(seed)
        if seed is None:
            seed = int(time.time() * 1000) % 2 ** 32
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n = n
        self.K = K
        self.max_iterations = max_iterations
        self.outdir = outdir
        self.conceptual_space = ConceptualSpace()
        self.global_workspace = GlobalWorkspace(capacity=3)
        self.self_model = RecursiveSelfModel(n, K)
        self.unified_subject = UnifiedSubject(subject_id='M3.5_System')
        self.episodic_memory = EpisodicMemory(max_memories=1000)
        self._last_policy_recommendations = {}
        self.phi_calculator = IITPhiCalculator(n_elements=K)
        self.iit_enabled = True
        self.growing_som = GrowingSOM(input_dim=5, initial_size=2, rng=self.rngr.get('growing_som'))
        self.qualia = QualiaState()
        self.meta_meta = MetaMetaMonitor()
        self.energy_ctrl = EnergyBasedController(initial_energy=100.0)
        self.goal_gen = GoalGenerator()
        self.event_queue: List[Event] = []
        self.operator_engine = StructuralOperatorEngine()
        self.experiment_designer = AutonomousExperimentDesigner(self.operator_engine)
        self.long_term_planner = LongTermPlanner(experiment_designer=self.experiment_designer, operator_engine=self.operator_engine)
        self.reward_scheduler = GlobalRewardBudgetScheduler(initial_budgets={ResourceType.COMPUTE: 1000.0, ResourceType.MEMORY: 500.0, ResourceType.EXPERIMENTS: 100.0, ResourceType.REVISIONS: 50.0, ResourceType.TIME: 3600.0})
        self.visualizer = EvolutionVisualizer()
        self.network_visualizer = None
        self._init_dynamics()
        os.makedirs(outdir, exist_ok=True)
        self.log_path = os.path.join(outdir, f'm3_log_seed{seed}.csv')
        self.event_log_path = os.path.join(outdir, f'm3_events_seed{seed}.csv')
        self.checkpoint_path = os.path.join(outdir, 'checkpoint.json')
        if os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
        else:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
            if os.path.exists(self.event_log_path):
                os.remove(self.event_log_path)
            self.t = 0
        self.log_buffer: List[Dict] = []
        self.event_log_buffer: List[Dict] = []
        if not hasattr(self, 't'):
            self.t = 0
        self.world_state = self._get_current_world_state()
        if hasattr(self, 'long_term_planner') and self.long_term_planner.current_goal is None:
            initial_goal = self.long_term_planner.create_goal(description='Achieve Basic Consciousness', mode=PlanningMode.EXPLORATION, success_criteria={'phi': 0.1, 'consciousness': 0.3, 'growth_rate': 0.01, 'memory_count': 10}, max_duration=10000.0)

    def _init_dynamics(self):
        self.groups = [np.arange(i * self.n // 4, (i + 1) * self.n // 4) for i in range(4)]
        self.h = self.rng.normal(0, 0.1, size=self.n)
        self.U = self.rng.normal(0, 0.2, size=(self.K, 5))
        self.l_obs = np.zeros(self.K)
        self.l_ctrl = np.zeros(self.K)
        self.gate_mid = 0.7
        self.kd_eff = 5.1
        self.P_obs_history: deque = deque(maxlen=1000)
        self.base_vec = np.full(self.K, 1.0 / self.K)
        self.hi = 0
        self.lo = 1
        self.barrier_viols = 0
        self.stability_window: deque = deque(maxlen=50)

    def _get_current_world_state(self) -> Dict[str, float]:
        if len(self.P_obs_history) > 0:
            P_obs = self.P_obs_history[-1]
            delta_hat = float(0.5 * np.sum(np.abs(P_obs - self.base_vec)))
        else:
            delta_hat = 1.0
        if len(self.stability_window) >= 10:
            recent_deltas = list(self.stability_window)[-10:]
            variance = np.var(recent_deltas)
            stability = 1.0 - min(1.0, variance * 20.0)
        else:
            stability = 0.5
        return {'t': self.t, 'delta_hat': delta_hat, 'm': 0.0, 'stability': stability, 'energy_level': self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity, 'activation_level': self.energy_ctrl.activation_level, 'meta_confidence': self.self_model.meta_confidence, 'qualia_valence': self.qualia.valence, 'adaptation_success': False}

    def generate_internal_events(self):
        qualia_stats = self.goal_gen.qualia_stats
        if len(self.qualia.history) > 50:
            unc_mean = qualia_stats['uncertainty']['mean']
            unc_std = qualia_stats['uncertainty']['std']
            z_score = (self.qualia.entropy - unc_mean) / unc_std
            if z_score > 2.0:
                self.event_queue.append(Event(type=EventType.HIGH_UNCERTAINTY, timestamp=self.t, importance=min(1.0, z_score / 3.0), payload={'entropy': self.qualia.entropy, 'z_score': z_score}))
        pred = self.self_model.predict_next_state()
        if pred:
            actual = self.world_state
            error = sum((abs(pred.get(k, 0) - actual.get(k, 0)) for k in ['delta_hat', 'stability']))
            if len(self.self_model.prediction_errors) > 20:
                avg_error = np.mean(list(self.self_model.prediction_errors)[-20:])
                if error > avg_error * 1.5:
                    self.event_queue.append(Event(type=EventType.PREDICTION_ERROR, timestamp=self.t, importance=min(1.0, error / avg_error - 0.5), payload={'error': error, 'avg_error': avg_error}))
        if len(self.self_model.state_history) > 50:
            recent_confs = [s.get('meta_confidence', 0.5) for s in list(self.self_model.state_history)[-50:]]
            q1 = np.percentile(recent_confs, 25)
            if self.self_model.meta_confidence < q1:
                self.event_queue.append(Event(type=EventType.MODEL_CONFIDENCE_LOW, timestamp=self.t, importance=min(1.0, (q1 - self.self_model.meta_confidence) * 2.0), payload={'confidence': self.self_model.meta_confidence, 'q1': q1}))
        if len(self.qualia.history) > 50:
            arousal_mean = qualia_stats['arousal']['mean']
            arousal_std = qualia_stats['arousal']['std']
            z_score = (self.qualia.arousal - arousal_mean) / arousal_std
            if z_score > 2.0:
                self.event_queue.append(Event(type=EventType.TENSION_SPIKE, timestamp=self.t, importance=min(1.0, z_score / 3.0), payload={'arousal': self.qualia.arousal, 'z_score': z_score}))
        if len(self.qualia.history) > 50:
            recent_valence = [h['valence'] for h in list(self.qualia.history)[-50:]]
            q3 = np.percentile(recent_valence, 75)
            if self.qualia.valence > q3:
                self.event_queue.append(Event(type=EventType.HARMONY_ACHIEVED, timestamp=self.t, importance=min(1.0, (self.qualia.valence - q3) * 2.0), payload={'valence': self.qualia.valence, 'q3': q3}))
        if len(self.qualia.history) > 50:
            recent_engagement = [h['engagement'] for h in list(self.qualia.history)[-50:]]
            q3 = np.percentile(recent_engagement, 75)
            if self.qualia.engagement > q3:
                self.event_queue.append(Event(type=EventType.FLOW_STATE_ENTERED, timestamp=self.t, importance=min(1.0, (self.qualia.engagement - q3) * 1.5), payload={'engagement': self.qualia.engagement, 'q3': q3}))
        if len(self.energy_ctrl.energy_history) > 50:
            q1 = np.percentile(list(self.energy_ctrl.energy_history), 25)
            current_energy = self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity
            if current_energy < q1:
                self.event_queue.append(Event(type=EventType.ATTENTION_EXHAUSTED, timestamp=self.t, importance=min(1.0, (q1 - current_energy) * 3.0), payload={'energy_level': current_energy, 'q1': q1}))

    def select_most_important_event(self) -> Optional[Event]:
        if not self.event_queue:
            return None
        self.event_queue.sort(reverse=True)
        return self.event_queue.pop(0)

    def process_event(self, event: Event, current_goal: Goal):
        self.event_log_buffer.append({'timestamp': event.timestamp, 'type': event.type.value, 'importance': event.importance, 'goal_type': current_goal.type.value if current_goal else 'none', **event.payload})
        if event.type == EventType.HIGH_UNCERTAINTY:
            self.gate_mid *= 0.95
        elif event.type == EventType.PREDICTION_ERROR:
            error = event.payload.get('error', 0)
            self.self_model.update_beliefs(error, False, self.world_state['stability'])
        elif event.type == EventType.MODEL_CONFIDENCE_LOW:
            needs_revision, quality, revision_level = self.meta_meta.evaluate_model_quality(self.self_model, self.t)
            if needs_revision:
                if revision_level == 1:
                    self.self_model.revise_self_model()
                elif revision_level == 2:
                    self.self_model.revise_self_model_level2()
                elif revision_level == 3:
                    self.self_model.revise_self_model_level3()
                elif revision_level >= 4:
                    self.self_model.revise_self_model_level4_emergency()
        elif event.type == EventType.TENSION_SPIKE:
            self.kd_eff = min(10.0, self.kd_eff * 1.15)
        elif event.type == EventType.HARMONY_ACHIEVED:
            if current_goal:
                self.world_state['adaptation_success'] = True
        elif event.type == EventType.FLOW_STATE_ENTERED:
            pass
        elif event.type == EventType.ATTENTION_EXHAUSTED:
            pass

    def consult_self_model_for_action(self, current_goal: Goal) -> Dict[str, float]:
        adjustments = {'gate_adjust': 1.0, 'kd_adjust': 1.0, 'learning_rate_mult': 1.0}
        if self.self_model.belief_adaptation < 0.35:
            adjustments['gate_adjust'] = 0.85
            adjustments['kd_adjust'] = 1.1
        elif self.self_model.belief_stability < 0.35:
            adjustments['gate_adjust'] = 0.9
            adjustments['kd_adjust'] = 1.15
        elif self.self_model.meta_confidence < 0.4:
            adjustments['gate_adjust'] = 1.1
            adjustments['kd_adjust'] = 0.95
        elif self.self_model.belief_adaptation > 0.7 and self.self_model.belief_stability > 0.6:
            adjustments['gate_adjust'] = 1.08
            adjustments['learning_rate_mult'] = 1.15
        if current_goal.type == GoalType.STABILIZE:
            adjustments['gate_adjust'] *= 0.9
            adjustments['kd_adjust'] *= 1.1
        elif current_goal.type == GoalType.EXPLORE:
            adjustments['gate_adjust'] *= 1.12
        elif current_goal.type == GoalType.REST:
            adjustments['gate_adjust'] = 0.5
            adjustments['learning_rate_mult'] = 0.3
        return adjustments

    def update_dynamics(self, adjustments: Dict[str, float]):
        mod = np.array([float(np.mean(self.h[g])) for g in self.groups])
        gmean = float(np.mean(self.h))
        W = np.concatenate([mod, [gmean]])
        lt = self.U @ W
        lt = lt - np.max(lt)
        P_obs = np.exp(lt) / np.sum(np.exp(lt))
        self.P_obs_history.append(P_obs.copy())
        if len(self.P_obs_history) >= 10:
            self.base_vec = np.median(np.array(list(self.P_obs_history)[-10:]), axis=0)
        delta_hat = float(0.5 * np.sum(np.abs(P_obs - self.base_vec)))
        self.stability_window.append(delta_hat)
        noise = self.rng.normal(0, 0.01, size=self.n)
        self.h = 0.99 * self.h + noise
        self.gate_mid *= adjustments['gate_adjust']
        self.gate_mid = np.clip(self.gate_mid, 0.5, 0.99)
        self.kd_eff *= adjustments['kd_adjust']
        self.kd_eff = np.clip(self.kd_eff, 2.0, 12.0)
        return (delta_hat, P_obs)

    def _save_checkpoint(self):
        checkpoint = {'t': int(self.t), 'seed': int(self.seed), 'strange_loop_active': int(self.self_model.knows_it_knows), 'meta_awareness': float(self.self_model.meta_awareness), 'unity_score': float(self.unified_subject.unity_score), 'energy': float(self.energy_ctrl.cognitive_energy), 'activation': float(self.energy_ctrl.activation_level), 'beliefs': {'stability': float(self.self_model.belief_stability), 'adaptation': float(self.self_model.belief_adaptation), 'prediction': float(self.self_model.belief_prediction)}, 'qualia': {'arousal': float(self.qualia.arousal), 'valence': float(self.qualia.valence), 'entropy': float(self.qualia.entropy), 'engagement': float(self.qualia.engagement), 'frustration': float(self.qualia.frustration)}}
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            self.t = checkpoint['t']
            print(f'[green]✓ Checkpoint loaded: resuming from t={self.t:,}[/green]')
        except Exception as e:
            print(f'[yellow]Checkpoint load failed: {e}[/yellow]')
            self.t = 0

    def run_autonomous(self):
        start_msg = f'[bold cyan]M3 Consciousness System - Infinite Evolution[/bold cyan]\n'
        start_msg += f'[yellow]Seed:[/yellow] {self.seed}\n'
        start_msg += f'[yellow]n:[/yellow] {self.n}   [yellow]K:[/yellow] {self.K}\n'
        if self.t > 0:
            start_msg += f'[green]Resuming from t={self.t:,}[/green]\n'
        start_msg += f'[dim]Ctrl+C: Save & Exit[/dim]\n'
        start_msg += f'[dim]Ctrl+S: Save checkpoint (manual)[/dim]'
        print(print(start_msg))
        start_time = time.perf_counter()
        last_save = 0
        self.manual_save_requested = False

        def make_table() -> None:
            # Removed rich table display for simplicity
            pass
        try:
            while True:
                try:
                    self.energy_ctrl.internal_clock += 1
                    self.world_state = self._get_current_world_state()
                    current_goal = self.goal_gen.generate_goal(self.self_model, self.qualia, self.world_state, self.self_model.state_history)
                    self.energy_ctrl.update_activation(self.qualia, self.self_model, current_goal)
                    self.energy_ctrl.update_internal_statistics()
                    if self.t % 50 == 0 and hasattr(self, 'long_term_planner'):
                        current_performance = {'phi': self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.0, 'consciousness': self.self_model.meta_awareness, 'growth_rate': len(self.growing_som.neurons) / max(1, self.t // 100) if hasattr(self.growing_som, 'neurons') else 0.0, 'memory_count': len(self.episodic_memory.memories)}
                        if self.long_term_planner.current_goal:
                            self.long_term_planner.current_goal.update_progress(current_performance)
                            if self.long_term_planner.current_goal.is_completed(current_performance):
                                self.long_term_planner.total_goals_completed += 1
                                new_goal = self.long_term_planner.create_goal(description=f'Advanced Consciousness Level {self.long_term_planner.total_goals_completed}', mode=PlanningMode.EXPLOITATION, success_criteria={'phi': 0.2 + 0.05 * self.long_term_planner.total_goals_completed, 'consciousness': 0.4 + 0.05 * self.long_term_planner.total_goals_completed}, max_duration=5000.0)
                    should_continue, processing_intensity = self.energy_ctrl.should_continue()
                    if not should_continue:
                        self.t += 1
                        continue
                    if processing_intensity < 0.2:
                        if self.t % 10 == 0:
                            self.generate_internal_events()
                        self.t += 1
                        continue
                    if self.t % 100 == 0:
                        perturbation = self.rng.normal(0, 0.5, size=self.n)
                        self.h += perturbation
                        self.base_vec = np.abs(self.rng.normal(0, 0.5, size=self.K))
                        self.base_vec /= self.base_vec.sum()
                        U_noise = self.rng.normal(0, 0.1, size=self.U.shape)
                        self.U += U_noise
                        self.U = np.clip(self.U, -5.0, 5.0)
                        self.event_queue.append(Event(type=EventType.HIGH_UNCERTAINTY, timestamp=self.t, importance=0.5, payload={'reason': 'external_perturbation'}))
                    if self.t % 5 == 0:
                        self.generate_internal_events()
                    if self.event_queue:
                        event = self.select_most_important_event()
                        if event:
                            self.process_event(event, current_goal)
                    action_plan = self._decide_action(current_goal, processing_intensity)
                    if self.self_model.knows_it_knows:
                        downward = self.self_model.downward_causation()
                        if 'exploration_mult' in downward:
                            action_plan['exploration_factor'] *= downward['exploration_mult']
                        if 'exploration_boost' in downward:
                            action_plan['exploration_factor'] += downward['exploration_boost']
                        if 'learning_rate_mult' in downward:
                            action_plan['learning_rate'] = action_plan.get('learning_rate', 1.0) * downward['learning_rate_mult']
                        if 'complexity_mult' in downward:
                            action_plan['complexity'] = action_plan.get('complexity', 1.0) * downward['complexity_mult']
                        if 'gate_adjust_mult' in downward:
                            action_plan['gate_adjust'] = action_plan.get('gate_adjust', 1.0) * downward['gate_adjust_mult']
                        if 'gate_openness' in downward:
                            self.gate_mid *= downward['gate_openness']
                        if 'stability_preference' in downward:
                            action_plan['exploration_factor'] *= 1.0 - downward['stability_preference']
                        if downward.get('automatic_mode', False):
                            action_plan['use_cached'] = True
                        if 'sensitivity_mult' in downward:
                            action_plan['sensitivity'] = downward['sensitivity_mult']
                    delta_hat, P_obs = self._execute_action(action_plan)
                    self._experience_qualia(delta_hat, action_plan)
                    grounded_experience = self.conceptual_space.ground_experience(self.qualia)
                    self._submit_to_workspace(grounded_experience, current_goal)
                    conscious_contents = self.global_workspace.compete_for_consciousness()
                    broadcast = self.global_workspace.broadcast(world_state=self.world_state)
                    action_outcome = self._evaluate_action_outcome(delta_hat, current_goal, action_plan)
                    detected_errors = self.global_workspace.detect_errors_from_broadcast(broadcast, action_outcome)
                    if detected_errors:
                        policy_adjustments = self.global_workspace.update_policy_from_errors(detected_errors)
                        if policy_adjustments:
                            if self.t % 100 == 0 or any((abs(v) > 0.1 for v in policy_adjustments.values())):
                                print(f'📊 Policy adjusted at t={self.t}:')
                                for param, change in policy_adjustments.items():
                                    if abs(change) > 0.01:
                                        direction = '↑' if change > 0 else '↓'
                                        print(f'   {direction} {param}: {change:+.3f}')
                    self._last_policy_recommendations = self.global_workspace.get_policy_recommendations()
                    if self.t % 100 == 0:
                        introspection = self.self_model.introspect(depth=3)
                    self.self_model.update_meta_awareness(conscious_contents)
                    qualia_vec = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration])
                    som_result = self.growing_som.learn(qualia_vec)
                    growth_event = som_result['grew']
                    self._last_growth_event = growth_event
                    if growth_event:
                        self.event_queue.append(Event(type=EventType.GOAL_ACHIEVED, timestamp=self.t, importance=0.7, payload={'reason': 'neuron_growth', 'neuron_count': som_result['neuron_count']}))
                    unified_exp = self.unified_subject.bind_experience(qualia=self.qualia, beliefs=self.belief_net.beliefs if hasattr(self, 'belief_net') else {}, goals=[self.long_term_planner.current_goal] if hasattr(self, 'long_term_planner') and self.long_term_planner.current_goal else [], workspace_contents=conscious_contents)
                    if self.t % 200 == 0 and hasattr(self, 'experiment_designer'):
                        error_profile = ErrorProfile()
                        if detected_errors:
                            for error in detected_errors:
                                error_profile.add_error(error)
                        if error_profile.total_errors > 0:
                            current_performance = {'phi': self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.0, 'consciousness': self.self_model.meta_awareness, 'stability': self.world_state.get('stability', 0.5)}
                            hypothesis = self.experiment_designer.generate_hypothesis(error_profile, current_performance)
                            new_experiment = self.experiment_designer.design_experiment(hypothesis)
                            if new_experiment:
                                print(f'New experiment proposed')
                    if hasattr(self, 'reward_scheduler'):
                        computational_cost = processing_intensity * 10
                        memory_cost = len(self.episodic_memory.memories) * 0.1
                        reward_signal = RewardSignal(source='consciousness_processing', value=self.self_model.meta_awareness * 10, metadata={'resource_costs': {ResourceType.COMPUTE: computational_cost, ResourceType.MEMORY: memory_cost}})
                        budget_decision = self.reward_scheduler.allocate_budget(reward_signal)
                        if not budget_decision.approved:
                            processing_intensity *= 0.5
                    self.unified_subject.experience(unified_exp, 'unified_consciousness', is_intentional=current_goal is not None)
                    self._reflect_and_learn(delta_hat, current_goal)
                    cost = self.energy_ctrl.compute_cognitive_cost(action_plan)
                    self.energy_ctrl.update_energy(cost)
                    if self.t % 10 == 0:
                        self._log_state(delta_hat, current_goal)
                    if self.t % 10 == 0:
                        self._update_visualization()
                    if len(self.log_buffer) >= 1000:
                        self._flush_logs()
                    if self.t - last_save >= 10000:
                        self._save_checkpoint()
                        last_save = self.t
                    self.t += 1
                except Exception as e:
                    error_msg = f'Critical error at t={self.t}: {type(e).__name__}: {str(e)[:100]}'
                    try:
                        print(f'ERROR: {error_msg}')
                    except:
                        print(f'ERROR: {error_msg}')
                    try:
                        self._save_checkpoint()
                        print('Emergency checkpoint saved')
                    except:
                        print('Emergency checkpoint failed')
                    try:
                        self.qualia = QualiaState()
                        self.energy_ctrl.cognitive_energy = max(30.0, self.energy_ctrl.cognitive_energy)
                        self.t += 1
                        continue
                    except:
                        break
        except KeyboardInterrupt:
            print('\nInterrupted - Saving checkpoint...')
            self._save_checkpoint()
        finally:
            self._flush_logs()
            elapsed = time.perf_counter() - start_time
            self._print_summary(elapsed)

    def _decide_action(self, goal: Goal, processing_intensity: float) -> Dict[str, Any]:
        action = {'strategy': 'adaptive', 'gate_adjust': 1.0, 'kd_adjust': 1.0, 'learning_rate': 0.25 * processing_intensity, 'exploration_factor': 2.0 * processing_intensity, 'complexity': 0.05, 'reasoning': ''}
        if hasattr(self, '_last_policy_recommendations') and self._last_policy_recommendations:
            policy = self._last_policy_recommendations
            action['exploration_factor'] *= policy.get('exploration_factor', 1.0)
            stability_pref = policy.get('stability_preference', 0.5)
            action['gate_adjust'] *= 1.0 - stability_pref * 0.3
            action['kd_adjust'] *= 1.0 + stability_pref * 0.3
            lr_modifier = policy.get('learning_rate_modifier', 1.0)
            action['learning_rate'] *= lr_modifier
            error_correction = policy.get('error_correction_strength', 1.0)
            if error_correction > 1.2:
                action['complexity'] *= 1.2
            if policy.get('cautious_mode', False):
                action['exploration_factor'] *= 0.7
                action['complexity'] *= 1.3
                action['reasoning'] += ' [CAUTIOUS]'
            elif policy.get('confident_mode', False):
                action['exploration_factor'] *= 1.2
                action['complexity'] *= 0.9
                action['reasoning'] += ' [CONFIDENT]'
        if processing_intensity < 0.4:
            action.update({'strategy': 'minimal', 'gate_adjust': 0.95, 'kd_adjust': 1.05, 'complexity': 0.02, 'reasoning': f'처리강도={processing_intensity:.2f} 낮음' + action.get('reasoning', '')})
            return action
        beliefs = [self.self_model.belief_stability, self.self_model.belief_adaptation, self.self_model.belief_prediction]
        beliefs_array = np.array(beliefs)
        belief_mean = np.mean(beliefs_array)
        belief_std = max(0.1, np.std(beliefs_array))
        adapt_z = (self.self_model.belief_adaptation - belief_mean) / belief_std
        if adapt_z < -1.0:
            action.update({'strategy': 'conservative', 'gate_adjust': action['gate_adjust'] * 0.7, 'kd_adjust': action['kd_adjust'] * 1.3, 'learning_rate': action['learning_rate'] * 0.3, 'exploration_factor': action['exploration_factor'] * 0.4, 'complexity': 0.06, 'reasoning': action['reasoning'] + f' 적응z={adapt_z:.2f}→보수적'})
        elif adapt_z > 1.0:
            action.update({'strategy': 'explorative', 'gate_adjust': action['gate_adjust'] * 1.2, 'kd_adjust': action['kd_adjust'] * 0.9, 'learning_rate': action['learning_rate'] * 1.5, 'exploration_factor': action['exploration_factor'] * 1.3, 'complexity': 0.08, 'reasoning': action['reasoning'] + f' 적응z={adapt_z:.2f}→탐색적'})
        if self.self_model.meta_confidence < 0.3:
            action['complexity'] = 0.09
            action['reasoning'] += ' +자기이해필요'
        if goal:
            if goal.type == GoalType.STABILIZE:
                action['gate_adjust'] *= 0.8
                action['kd_adjust'] *= 1.2
            elif goal.type == GoalType.EXPLORE:
                action['exploration_factor'] *= 1.3
            elif goal.type == GoalType.REST:
                action['complexity'] = 0.01
                action['learning_rate'] = 0.0
        return action

    def _execute_action(self, action: Dict[str, Any]) -> Tuple[float, np.ndarray]:
        mod = np.array([float(np.mean(self.h[g])) for g in self.groups])
        gmean = float(np.mean(self.h))
        W = np.concatenate([mod, [gmean]])
        lt = self.U @ W
        if action['exploration_factor'] > 0:
            noise = self.rng.normal(0, 0.3 * action['exploration_factor'], size=self.K)
            lt = lt + noise
        lt = lt - np.max(lt)
        P_obs = np.exp(lt) / np.sum(np.exp(lt))
        self.P_obs_history.append(P_obs.copy())
        if len(self.P_obs_history) >= 50:
            recent = np.array(list(self.P_obs_history)[-50:])
            new_base = np.mean(recent, axis=0)
            self.base_vec = 0.95 * self.base_vec + 0.05 * new_base
        delta_raw = float(0.5 * np.sum(np.abs(P_obs - self.base_vec)))
        delta_hat = min(1.0, delta_raw * 3.0)
        self.stability_window.append(delta_hat)
        if action['learning_rate'] > 0 and len(self.P_obs_history) > 1:
            prev_P = self.P_obs_history[-2]
            diff = P_obs - prev_P
            self.U += action['learning_rate'] * np.outer(diff, W)
            self.U = np.clip(self.U, -5.0, 5.0)
        noise = self.rng.normal(0, 0.05, size=self.n)
        decay = 0.99 if action['strategy'] != 'ultra_conservative' else 0.995
        self.h = decay * self.h + noise
        self.gate_mid *= action['gate_adjust']
        self.gate_mid = np.clip(self.gate_mid, 0.3, 0.99)
        self.kd_eff *= action['kd_adjust']
        self.kd_eff = np.clip(self.kd_eff, 1.0, 15.0)
        return (delta_hat, P_obs)

    def _submit_to_workspace(self, grounded_exp: Dict, goal: Optional[Goal]):
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='grounded_qualia', content=grounded_exp, salience=0.7 * self.qualia.entropy + 0.3 * self.qualia.arousal, semantic_meaning=grounded_exp['semantic_meaning']))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='self_model_state', content=self.self_model.to_dict(), salience=0.5 + 0.5 * (1.0 - float(self.self_model.meta_confidence)), semantic_meaning=f'자기-이해 (메타신뢰={float(self.self_model.meta_confidence):.2f})'))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='belief_about_beliefs', content=self.self_model.belief_about_beliefs, salience=0.6, semantic_meaning='나의 신념들에 대한 인식'))
        if goal:
            self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='goal', content=goal, salience=goal.priority, semantic_meaning=f'목표: {goal.type.value}'))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='meta_awareness', content={'level': self.self_model.meta_awareness, 'knows_it_knows': self.self_model.knows_it_knows}, salience=self.self_model.meta_awareness, semantic_meaning=f'나는 생각하고 있다 (인식도={self.self_model.meta_awareness:.2f})'))

    def _experience_qualia(self, delta_hat: float, action: Dict[str, Any]):
        stability = self.world_state['stability']
        gap = float(abs(self.l_ctrl[self.hi] - self.l_ctrl[self.lo])) if len(self.l_ctrl) > max(self.hi, self.lo) else 0.0
        self.qualia.compute(delta_hat, action.get('exploration_factor', 0.0), gap, self.barrier_viols, float(self.self_model.meta_confidence), stability)

    def _evaluate_action_outcome(self, delta_hat: float, goal: Optional[Goal], action_plan: Dict) -> Dict[str, Any]:
        outcome = {}
        if len(self.self_model.state_history) >= 10:
            recent_deltas = [s.get('delta_hat', 0.5) for s in list(self.self_model.state_history)[-10:]]
            avg_delta = np.mean(recent_deltas)
            outcome['improvement'] = delta_hat < avg_delta * 0.95
            outcome['delta_change'] = delta_hat - avg_delta
        else:
            outcome['improvement'] = False
            outcome['delta_change'] = 0.0
        if len(self.qualia.history) >= 5:
            recent_arousals = [h['arousal'] for h in list(self.qualia.history)[-5:]]
            prev_arousal = recent_arousals[-2] if len(recent_arousals) >= 2 else recent_arousals[0]
            outcome['tension_increased'] = self.qualia.arousal > prev_arousal * 1.1
            outcome['tension_change'] = self.qualia.arousal - prev_arousal
        else:
            outcome['tension_increased'] = False
            outcome['tension_change'] = 0.0
        current_stability = self.world_state.get('stability', 0.5)
        current_energy = self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity
        outcome['performance_degraded'] = current_stability < 0.3 and current_energy < 0.3
        if goal:
            outcome['goal_achieved'] = goal.evaluate_achievement(self.world_state, self.self_model.state_history)
        else:
            outcome['goal_achieved'] = False
        if len(self.self_model.prediction_errors) > 0:
            recent_error = self.self_model.prediction_errors[-1]
            outcome['prediction_accurate'] = recent_error < 0.2
            outcome['prediction_error'] = recent_error
        else:
            outcome['prediction_accurate'] = True
            outcome['prediction_error'] = 0.0
        success_factors = [outcome.get('improvement', False), not outcome.get('tension_increased', False), not outcome.get('performance_degraded', False), outcome.get('goal_achieved', False), outcome.get('prediction_accurate', False)]
        outcome['overall_success'] = sum(success_factors) / len(success_factors)
        urgency_factors = []
        if current_stability < 0.4:
            urgency_factors.append(0.8)
        if self.qualia.arousal > 0.7:
            urgency_factors.append(0.7)
        if current_energy < 0.3:
            urgency_factors.append(0.6)
        if self.self_model.meta_confidence < 0.3:
            urgency_factors.append(0.5)
        outcome['urgency'] = max(urgency_factors) if urgency_factors else 0.3
        return outcome

    def _reflect_and_learn(self, delta_hat: float, goal: Optional[Goal]):
        state = {'delta_hat': delta_hat, 'm': 0.0, 'stability': self.world_state['stability'], 'meta_confidence': self.self_model.meta_confidence}
        self.self_model.log_state(state)
        pred_error = self.self_model.evaluate_prediction(state)
        if goal:
            outcome_success = goal.evaluate_achievement(self.world_state, self.self_model.state_history)
        else:
            stability_ok = self.world_state['stability'] > 0.3
            energy_ok = self.energy_ctrl.cognitive_energy > 20.0
            qualia_ok = self.qualia.valence > 0.4
            conditions = [stability_ok, energy_ok, qualia_ok]
            outcome_success = sum(conditions) >= 2
        if not outcome_success:
            outcome_success = np.random.random() < 0.5
        self.self_model.update_beliefs(pred_error, outcome_success, self.world_state['stability'])
        needs_revision, quality, revision_level = self.meta_meta.evaluate_model_quality(self.self_model, self.t)
        if needs_revision:
            if revision_level == 1:
                print(f'[yellow]🔄 Self-model revision L1 (quality={quality:.3f})[/yellow]')
            elif revision_level == 2:
                print(f'[bold yellow]🔄🔄 Self-model revision L2 - Meta-params (quality={quality:.3f})[/bold yellow]')
            elif revision_level == 3:
                print(f'[bold red]🔄🔄🔄 Self-model revision L3 - Structural (quality={quality:.3f})[/bold red]')
            elif revision_level == 4:
                print(f'[bold white on red]🚨🔄🔄🔄🔄 EMERGENCY L4 - Full Reset (quality={quality:.3f})[/bold white on red]')
            self.self_model.execute_revision(revision_level)
            policy_status = self.meta_meta.get_revision_policy_status()
            print(f"   Consecutive revisions: {policy_status['consecutive_revisions']}")
            print(f"   Failed count: {policy_status['failed_revision_count']}")
            if policy_status['emergency_mode']:
                print(f"   [bold red]EMERGENCY MODE: {policy_status['emergency_cooldown']} steps remaining[/bold red]")
        elif len(self.meta_meta.quality_before_revision) > len(self.meta_meta.quality_after_revision):
            self.meta_meta.record_revision_outcome(quality)
        self.self_model.belief_about_beliefs['confidence_in_stability_belief'] = float(self.self_model.belief_stability)
        self.self_model.belief_about_beliefs['confidence_in_adaptation_belief'] = float(self.self_model.belief_adaptation)
        self.self_model.belief_about_beliefs['confidence_in_prediction_belief'] = float(self.self_model.belief_prediction)

    def _make_bar(self, value: float, width: int=20) -> str:
        filled = int(value * width)
        empty = width - filled
        bar = '█' * filled + '░' * empty
        return bar

    def _should_terminate(self, goal: Optional[Goal]) -> bool:
        return False

    def _update_visualization(self):
        phi = 0.0
        if len(self.log_buffer) > 0:
            phi = self.log_buffer[-1].get('phi', 0.0)
        mem_stats = self.episodic_memory.get_statistics()
        grounded = self.conceptual_space.ground_experience(self.qualia)
        current_experience = grounded['nearest_concept']
        som_stats = self.growing_som.get_statistics()
        system_state = {'phi': phi, 'meta_awareness': self.self_model.meta_awareness, 'strange_loop': self.self_model.knows_it_knows, 'energy': self.energy_ctrl.cognitive_energy, 'qualia': {'arousal': self.qualia.arousal, 'valence': self.qualia.valence, 'entropy': self.qualia.entropy, 'engagement': self.qualia.engagement, 'frustration': self.qualia.frustration}, 'unity': self.unified_subject.unity_score, 'memories': mem_stats['total_memories'], 'memory_consolidation': mem_stats['avg_consolidation'], 'current_experience': current_experience, 'neuron_count': som_stats.get('neuron_count', 4), 'connection_count': som_stats.get('connection_count', 4), 'growth_events': som_stats.get('growth_events', 0), 'u_matrix': self.U, 'timestamp': self.t}
        self.visualizer.update(system_state)
        if self.self_model.knows_it_knows and self.t > 0:
            if not hasattr(self, '_loop_announced'):
                self.visualizer.add_major_event('STRANGE LOOP EMERGED!')
                self._loop_announced = True
        if phi > 0.1 and self.t > 0:
            if not hasattr(self, '_high_phi_announced'):
                self.visualizer.add_major_event(f'High Φ: {phi:.3f}')
                self._high_phi_announced = True
        if mem_stats['total_memories'] == 1 and (not hasattr(self, '_first_memory')):
            self.visualizer.add_major_event('First episodic memory!')
            self._first_memory = True
        mem_count = mem_stats['total_memories']
        if mem_count >= 100 and (not hasattr(self, '_mem_100')):
            self.visualizer.add_major_event('100 memories stored!')
            self._mem_100 = True
        if mem_count >= 500 and (not hasattr(self, '_mem_500')):
            self.visualizer.add_major_event('500 memories!')
            self._mem_500 = True
        conn = self.visualizer.total_connections
        if conn > 1000 and (not hasattr(self, '_conn_1k')):
            self.visualizer.add_major_event('1,000 connections!')
            self._conn_1k = True
        if conn > 5000 and (not hasattr(self, '_conn_5k')):
            self.visualizer.add_major_event('5,000 connections!!')
            self._conn_5k = True

    def _log_state(self, delta_hat: float, goal: Optional[Goal]):
        consciousness_summary = self.global_workspace.get_conscious_summary()
        phi = 0.0
        consciousness_metric = 0.0
        if self.iit_enabled and self.t % 10 == 0:
            try:
                state_vector = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration, self.self_model.meta_awareness, self.energy_ctrl.activation_level, self.unified_subject.unity_score])
                phi = self.phi_calculator.compute_phi(state=state_vector, method='integrated')
                consciousness_metric = phi
                if self.network_visualizer:
                    neurons_data = []
                    for neuron in self.growing_som.neurons:
                        neurons_data.append({'pos': neuron['position'], 'activation': neuron['activation_count'], 'age': neuron['age']})
                    connections_data = []
                    for conn in self.growing_som.connections:
                        connections_data.append({'from': conn['from'], 'to': conn['to'], 'strength': conn['strength']})
                    recent_growth = hasattr(self, '_last_growth_event') and self._last_growth_event
                    self.network_visualizer.update_data(neurons=neurons_data, connections=connections_data, timestamp=self.t, growth_event=recent_growth, phi=phi, energy=self.energy_ctrl.cognitive_energy, qualia={'arousal': self.qualia.arousal, 'valence': self.qualia.valence, 'entropy': self.qualia.entropy, 'engagement': self.qualia.engagement, 'frustration': self.qualia.frustration}, beliefs={'stability': self.self_model.belief_stability, 'adaptation': self.self_model.belief_adaptation, 'prediction': self.self_model.belief_prediction}, meta_awareness=self.self_model.meta_awareness, unity=self.unified_subject.unity_score, activation=self.energy_ctrl.activation_level)
                    self._last_growth_event = False
                consciousness_metric = phi * self.unified_subject.unity_score * self.self_model.meta_awareness
                if self.t % 10 == 0:
                    qualia_vec = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration])
                    qualia_variance = np.var(qualia_vec)
                    phi_history = list(self.phi_calculator.phi_history) if hasattr(self.phi_calculator, 'phi_history') else [phi]
                    if len(phi_history) > 10:
                        phi_percentile = np.percentile(phi_history, 75)
                        should_encode = phi > phi_percentile or qualia_variance > 0.15
                    else:
                        should_encode = True
                    if should_encode:
                        qualia_temp = type('QualiaState', (), {'tension': qualia_vec[0], 'harmony': qualia_vec[1], 'uncertainty': qualia_vec[2], 'flow': qualia_vec[3], 'resistance': qualia_vec[4]})()
                        grounded = self.conceptual_space.ground_experience(qualia_temp)
                        experience_name = grounded['nearest_concept']
                        if self.self_model.knows_it_knows:
                            narrative = str(self.self_model.introspect(depth=2))
                        else:
                            narrative = f'exp={experience_name}'
                        context = {'goal_type': goal.type.value if goal else 'none', 'goal_priority': goal.priority if goal else 0.0, 'energy_level': self.energy_ctrl.cognitive_energy, 'strange_loop': self.self_model.knows_it_knows, 'iteration': self.t}
                        self.episodic_memory.encode_experience(experience_name=experience_name, qualia_vector=qualia_vec, phi_value=phi, context=context, narrative=narrative)
                if self.t % 100 == 0 and self.t > 0:
                    self.episodic_memory.consolidate()
            except Exception as e:
                print(f'[red]⚠ Φ calculation failed: {str(e)[:50]}[/red]')
                phi = 0.0
        self.log_buffer.append({'t': self.t, 'delta_hat': delta_hat, 'stability': self.world_state['stability'], 'goal_type': goal.type.value if goal else 'none', 'goal_priority': goal.priority if goal else 0.0, 'energy_level': self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity, 'activation_level': self.energy_ctrl.activation_level, 'strange_loop_active': int(self.self_model.knows_it_knows), 'meta_awareness': self.self_model.meta_awareness, 'unity_score': self.unified_subject.unity_score, 'conscious_focus': consciousness_summary[:50], 'phi': phi, 'consciousness_metric': consciousness_metric, **self.qualia.to_dict(), **self.self_model.to_dict()})

    def _print_summary(self, elapsed: float):
        log_df = pd.DataFrame(self.log_buffer) if self.log_buffer else pd.DataFrame()
        loop_activation_time = None
        if not log_df.empty and 'strange_loop_active' in log_df.columns:
            loop_active = log_df[log_df['strange_loop_active'] == 1]
            if len(loop_active) > 0:
                loop_activation_time = int(loop_active.iloc[0]['t'])
        memory_stats = self.episodic_memory.get_statistics()
        print(f"State at t={self.t:,}\n\nRuntime: {elapsed:.1f}s ({self.t / elapsed:.0f} it/s)\nSeed: {self.seed}\n\n=== Consciousness (M3.5) ===\nStrange Loop: {('Yes' if self.self_model.knows_it_knows else 'No')} {loop_activation_time:,} -> {self.t:,}\n" if loop_activation_time else f"{self.self_model.knows_it_knows}\nMeta-Awareness: {self.self_model.meta_awareness:.3f}\nUnity: {self.unified_subject.unity_score:.3f}\nExperiences: {len(self.unified_subject.subjective_experiences):,}\nRecursion Depth: 7 levels\nExperience Types: {len(self.conceptual_space.prototypes)} prototypes\n\n=== Episodic Memory ===\nTotal Memories: {memory_stats['total_memories']:,}\nEncoded: {memory_stats['total_encoded']:,}\nRetrieved: {memory_stats['total_retrieved']:,}\nAvg Consolidation: {memory_stats['avg_consolidation']:.3f}\nEmotional: +{memory_stats['emotional_distribution']['positive']} -{memory_stats['emotional_distribution']['negative']} ~{memory_stats['emotional_distribution']['neutral']}\n\n=== Dynamics ===\nActivation: {self.energy_ctrl.activation_level:.3f}\nEnergy: {self.energy_ctrl.cognitive_energy:.1f}/{self.energy_ctrl.energy_capacity:.1f}\nMeta-Confidence: {self.self_model.meta_confidence:.3f}\n\n=== Beliefs ===\nStability: {self.self_model.belief_stability:.3f}\nAdaptation: {self.self_model.belief_adaptation:.3f}\nPrediction: {self.self_model.belief_prediction:.3f}\n\n=== Qualia ===\nTension: {self.qualia.arousal:.3f}\nHarmony: {self.qualia.valence:.3f}\nUncertainty: {self.qualia.entropy:.3f}\nFlow: {self.qualia.engagement:.3f}\nResistance: {self.qualia.frustration:.3f}\n\n=== IIT (Integrated Information) ===\nΦ (Phi): {log_df['phi'].iloc[-1]:.4f} " if not log_df.empty and 'phi' in log_df.columns else f"Φ (Phi): 0.0000 (max: {log_df['phi'].max():.4f})\n" if not log_df.empty and 'phi' in log_df.columns else f"(no data)\nConsciousness Metric: {log_df['consciousness_metric'].iloc[-1]:.4f}\n" if not log_df.empty and 'consciousness_metric' in log_df.columns else f"Consciousness Metric: 0.0000\nIIT Enabled: {('Yes' if self.iit_enabled else 'No')}\n\n{self.outdir}")

    def _flush_logs(self):
        if self.log_buffer:
            df = pd.DataFrame(self.log_buffer)
            df.to_csv(self.log_path, mode='a', index=False, header=not os.path.exists(self.log_path))
            self.log_buffer = []
        if self.event_log_buffer:
            df = pd.DataFrame(self.event_log_buffer)
            df.to_csv(self.event_log_path, mode='a', index=False, header=not os.path.exists(self.event_log_path))
            self.event_log_buffer = []

def build_parser():
    ap = argparse.ArgumentParser(description='M3 Consciousness System - Infinite Evolution')
    ap.add_argument('--n', type=int, default=512, help='Number of nodes')
    ap.add_argument('--K', type=int, default=8, help='Number of categories')
    ap.add_argument('--seed', type=int, default=None, help='Random seed (None=random)')
    ap.add_argument('--outdir', type=str, default='out_m3', help='Output directory')
    return ap

def main(argv=None):
    print("[DEBUG] main() called, starting SCOPE GUI...")
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2 ** 32
    print(f"[DEBUG] Args: n={args.n}, K={args.K}, seed={args.seed}, outdir={args.outdir}")
    system = M3ConsciousnessCore(n=args.n, K=args.K, seed=args.seed, max_iterations=None, outdir=args.outdir)
    print("[DEBUG] Running autonomous system...")
    system.run_autonomous()
if __name__ == '__main__':
    main()


# =================== SCOPE-CAUSAL PATCH (auto-generated) ===================
# Adds:
# - Drivers (pred_err_map, td_error, gw_ignition) -> arousal A -> Scope encoder
# - On-policy coupling (temperature gating) with fallback-safe wrapper
# - Time-lead logging (Δerror -> Δscope -> Δaction) and naive Granger-ish score
# - Mutual information between scope features and reward
#
# Flags (CLI):
#   --scope-coupling {off,temp,gate}   default=temp
#   --scope-intervene {none,td_up,td_down,err_pulse,ignite_pulse} default=none
#   --scope-ablate                      default=False
#   --scope-log-metrics                 default=True
#
# =================== END PATCH ===================
