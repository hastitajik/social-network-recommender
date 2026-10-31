"""
Implementation of:
  "A recommender system for social networks using link prediction,
   clustering and genetic algorithm"
  Lu et al., Egyptian Informatics Journal 32 (2025) 100784

Single-file, self-contained. Requires Python 3.11+.
Run:  python recommender.py
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILE_ATTRS = ["age", "gender", "language", "occupation", "education"]

EDUCATION_MAP: dict[str, float] = {
    "none": 0.0,
    "high_school": 0.2,
    "associate": 0.4,
    "bachelor": 0.6,
    "master": 0.8,
    "phd": 1.0,
}

GENDER_MAP: dict[str, int] = {
    "unspecified": 0,
    "male": 1,
    "female": 2,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProfileData:
    """Preprocessed user profile matrix aligned with a graph node list."""
    node_ids: np.ndarray        # shape (N,)  int
    vectors: np.ndarray         # shape (N, 5) float64
    id_to_idx: dict[int, int]   # node_id -> row index


@dataclass
class GAParams:
    """Hyperparameters for the genetic algorithm (Section 3.2.4)."""
    population_size: int = 150
    max_generations: int = 300
    crossover_rate: float = 0.7      # fraction of elite carried forward (Ps)
    mutation_rate: float = 0.05      # per-chromosome mutation probability (Pm)
    fitness_threshold: float = 1.0   # early-stop threshold (T)
    tournament_k: int = 3            # tournament selection pool size
    ga_sample_size: int = 50         # users sampled for fitness evaluation


@dataclass
class ModelParams:
    """Full model hyperparameters."""
    alpha: float = 0.5       # profile vs. connection similarity balance
    n_clusters: int = 9      # optimal value reported in paper
    path_length: int = 3     # optimal value reported in paper
    top_k: int = 10          # number of recommendations per user


@dataclass
class FoldResult:
    """Evaluation metrics for one cross-validation fold."""
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Stage 1 — Profile preprocessing
# ---------------------------------------------------------------------------

def _build_categorical_map(series: pd.Series) -> dict[str, int]:
    """
    Map each distinct non-null string value in series to a 1-based integer.
    Missing values map to 0.  The paper assigns the positional index in a
    sorted distinct list (Section 3.2.1, language/occupation encoding).
    """
    distinct = sorted(
        v for v in series.dropna().unique() if isinstance(v, str) and v.strip()
    )
    return {val: idx + 1 for idx, val in enumerate(distinct)}


def encode_profile_attributes(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame with columns [age, gender, language, occupation,
    education] into a float64 matrix of shape (N, 5).

    Encoding rules (Section 3.2.1):
      age        : raw integer, normalised separately via normalize_age()
      gender     : 0=unspecified, 1=male, 2=female
      language   : 1-based positional index in sorted distinct list; missing->0
      occupation : same as language
      education  : ordinal {none:0, hs:0.2, assoc:0.4, bsc:0.6, msc:0.8, phd:1.0}
    """
    n = len(df)
    vectors = np.zeros((n, 5), dtype=np.float64)

    # age (column 0) — raw; normalised in a separate pass
    vectors[:, 0] = (
        pd.to_numeric(df["age"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )

    # gender (column 1)
    vectors[:, 1] = (
        df["gender"]
        .fillna("unspecified")
        .str.lower()
        .str.strip()
        .map(lambda v: GENDER_MAP.get(v, 0))
        .to_numpy(dtype=float)
    )

    # language (column 2)
    lang_map = _build_categorical_map(df["language"])
    vectors[:, 2] = (
        df["language"]
        .fillna("")
        .map(lambda v: lang_map.get(str(v).strip(), 0))
        .to_numpy(dtype=float)
    )

    # occupation (column 3)
    occ_map = _build_categorical_map(df["occupation"])
    vectors[:, 3] = (
        df["occupation"]
        .fillna("")
        .map(lambda v: occ_map.get(str(v).strip(), 0))
        .to_numpy(dtype=float)
    )

    # education (column 4)
    vectors[:, 4] = (
        df["education"]
        .fillna("none")
        .str.lower()
        .str.strip()
        .map(lambda v: EDUCATION_MAP.get(v, 0.0))
        .to_numpy(dtype=float)
    )

    return vectors


def normalize_age(vectors: np.ndarray) -> np.ndarray:
    """
    Min-max normalise the age column (index 0) to [0, 1] in-place.
    Equation (1): N_age = (age - age_min) / (age_max - age_min).
    Returns the modified array.
    """
    age = vectors[:, 0]
    age_min = float(age.min())
    age_max = float(age.max())
    denom = age_max - age_min
    vectors[:, 0] = (age - age_min) / denom if denom > 0.0 else 0.0
    return vectors


def build_profile_data(df: pd.DataFrame, node_order: list[int]) -> ProfileData:
    """
    Full preprocessing pipeline for user profiles.

    Parameters
    ----------
    df:
        DataFrame indexed by node_id with columns matching PROFILE_ATTRS.
    node_order:
        Ordered list of node ids aligned with the graph's node iteration.

    Returns
    -------
    ProfileData with normalised vectors aligned to node_order.
    """
    df_aligned = df.reindex(node_order)
    vectors = encode_profile_attributes(df_aligned)
    normalize_age(vectors)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_order)}
    return ProfileData(
        node_ids=np.array(node_order, dtype=np.int64),
        vectors=vectors,
        id_to_idx=id_to_idx,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Edge weighting
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity S(x, y) = (x · y) / (||x|| ||y||).  Equation (4).
    Returns 0.0 when either vector has zero norm.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _connection_vector(
    G: nx.Graph,
    node: int,
    node_index: dict[int, int],
) -> np.ndarray:
    """
    Build connection vector C_i for a node: a dense array of length
    |node_index| where entry j=1 if (node, nodes[j]) is an edge.

    Scoped to node_index (graph or cluster) to keep memory bounded.
    """
    vec = np.zeros(len(node_index), dtype=np.float64)
    for nbr in G.neighbors(node):
        if nbr in node_index:
            vec[node_index[nbr]] = 1.0
    return vec


def build_weighted_graph(
    G: nx.Graph,
    profile: ProfileData,
    alpha: float = 0.5,
) -> nx.Graph:
    """
    Assign a weight to every edge in G using Equation (3):

        W(i,j) = 1 / [alpha * S(P_i,P_j) + (1-alpha) * S(C_i,C_j)]

    Lower weight = higher combined similarity = stronger relationship.
    The minimum spanning tree built downstream captures the strongest
    connections.

    Complexity: O(E * deg_avg) for connection vectors + O(E) for profiles.
    """
    node_list = list(G.nodes())
    local_idx: dict[int, int] = {n: i for i, n in enumerate(node_list)}

    # Cache connection vectors to avoid recomputing shared endpoints
    conn_cache: dict[int, np.ndarray] = {}

    def get_conn(node: int) -> np.ndarray:
        if node not in conn_cache:
            conn_cache[node] = _connection_vector(G, node, local_idx)
        return conn_cache[node]

    W = nx.Graph()
    W.add_nodes_from(G.nodes())

    for u, v in G.edges():
        p_u = profile.vectors[profile.id_to_idx[u]]
        p_v = profile.vectors[profile.id_to_idx[v]]
        p_sim = cosine_similarity(p_u, p_v)
        c_sim = cosine_similarity(get_conn(u), get_conn(v))

        combined = alpha * p_sim + (1.0 - alpha) * c_sim
        # A combined similarity of 0 means no meaningful relationship;
        # assign a large weight so this edge is excluded from the MST.
        weight = 1.0 / combined if combined > 1e-9 else 1e9
        W.add_edge(u, v, weight=weight)

    return W


# ---------------------------------------------------------------------------
# Stage 3 — Community detection (MST + modularity merging)
# ---------------------------------------------------------------------------

def build_mst(weighted_G: nx.Graph) -> nx.Graph:
    """
    Compute the Minimum Spanning Tree via Prim's algorithm.  O(E log N).
    Minimum weight = maximum similarity, so the MST keeps the strongest
    connectivity backbone.
    """
    return nx.minimum_spanning_tree(weighted_G, weight="weight", algorithm="prim")


def prune_mst_to_local_communities(mst: nx.Graph) -> list[set[int]]:
    """
    Remove the floor((N-1)/2) highest-weight edges from the MST.
    The resulting connected components become local communities.

    Section 3.2.2: removing k edges from a tree with N nodes produces
    k+1 components, so we get ceil((N+1)/2) local communities.
    """
    n = mst.number_of_nodes()
    n_remove = math.floor((n - 1) / 2)

    edges_sorted = sorted(
        mst.edges(data="weight"), key=lambda e: e[2], reverse=True
    )

    pruned = mst.copy()
    for u, v, _ in edges_sorted[:n_remove]:
        pruned.remove_edge(u, v)

    return [set(comp) for comp in nx.connected_components(pruned)]


def _modularity_delta(
    G: nx.Graph,
    comm_a: set[int],
    comm_b: set[int],
    m_total: float,
) -> float:
    """
    Incremental modularity gain when merging comm_a and comm_b.
    Uses the Newman-Girvan formula applied to Equation (5):

        delta_Q = e_ab/m - (k_a * k_b) / (2m)^2

    where e_ab is the total cross-community edge weight and k_a, k_b are
    the weighted degree sums of each community.
    """
    if m_total <= 0.0:
        return 0.0

    e_ab = sum(
        G[u][v].get("weight", 1.0)
        for u in comm_a
        for v in G.neighbors(u)
        if v in comm_b
    )

    k_a = sum(d for _, d in G.degree(comm_a, weight="weight"))
    k_b = sum(d for _, d in G.degree(comm_b, weight="weight"))

    return e_ab / m_total - (k_a * k_b) / ((2.0 * m_total) ** 2)


def merge_communities(
    G: nx.Graph,
    local_communities: list[set[int]],
) -> dict[int, list[int]]:
    """
    Greedy agglomerative modularity merging (Section 3.2.2, steps 1-6).

    At each iteration:
      - Find the pair of adjacent communities with the largest positive
        modularity gain delta_Q.
      - Merge that pair.
      - Repeat until no merge improves modularity.

    Only adjacent communities (sharing at least one cross-edge) are
    considered at each step, keeping the inner loop tractable.

    Returns
    -------
    dict mapping cluster_id (int) -> list of node ids
    """
    m_total = sum(d for _, d in G.degree(weight="weight")) / 2.0

    comms: list[set[int]] = [set(c) for c in local_communities]

    # node -> index into comms list
    node_to_comm: dict[int, int] = {}
    for idx, comm in enumerate(comms):
        for node in comm:
            node_to_comm[node] = idx

    improved = True
    while improved and len([c for c in comms if c]) > 1:
        improved = False
        best_delta = 0.0
        best_i = -1
        best_j = -1

        checked: set[tuple[int, int]] = set()

        for i, comm_i in enumerate(comms):
            if not comm_i:
                continue
            # Collect indices of neighbouring communities
            nbr_comms: set[int] = set()
            for node in comm_i:
                for nbr in G.neighbors(node):
                    j = node_to_comm.get(nbr, -1)
                    if j != -1 and j != i and comms[j]:
                        nbr_comms.add(j)

            for j in nbr_comms:
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)

                delta = _modularity_delta(G, comm_i, comms[j], m_total)
                if delta > best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_i != -1:
            for node in comms[best_j]:
                node_to_comm[node] = best_i
            comms[best_i] |= comms[best_j]
            comms[best_j] = set()
            improved = True

    # Compact to sequential cluster ids
    cluster_map: dict[int, list[int]] = {}
    cid = 0
    for comm in comms:
        if comm:
            cluster_map[cid] = list(comm)
            cid += 1

    return cluster_map


def detect_communities(
    weighted_G: nx.Graph,
    n_clusters_hint: int = 9,
) -> dict[int, list[int]]:
    """
    Full community detection pipeline (Section 3.2.2):
      1. Build Minimum Spanning Tree.
      2. Prune highest-weight edges to create local communities.
      3. Greedily merge local communities to maximise modularity.

    The n_clusters_hint is not a hard constraint; the merging phase
    determines the final count organically via modularity.

    Returns
    -------
    dict: cluster_id -> list[node_id]
    """
    mst = build_mst(weighted_G)
    local_comms = prune_mst_to_local_communities(mst)
    return merge_communities(weighted_G, local_comms)


def compute_cluster_centers(
    clusters: dict[int, list[int]],
    profile: ProfileData,
) -> dict[int, np.ndarray]:
    """
    Compute the mean profile vector for each cluster.
    Used to assign new/unseen users to the nearest cluster.
    """
    centers: dict[int, np.ndarray] = {}
    for cid, nodes in clusters.items():
        idxs = [profile.id_to_idx[n] for n in nodes if n in profile.id_to_idx]
        centers[cid] = (
            profile.vectors[idxs].mean(axis=0) if idxs
            else np.zeros(5, dtype=np.float64)
        )
    return centers


def assign_new_user(
    user_vector: np.ndarray,
    cluster_centers: dict[int, np.ndarray],
) -> int:
    """
    Assign a new user to the nearest cluster by Euclidean distance.
    Section 3.2: "the new user is assigned to the cluster with the
    nearest center."
    """
    return min(
        cluster_centers,
        key=lambda cid: float(np.linalg.norm(user_vector - cluster_centers[cid])),
    )


# ---------------------------------------------------------------------------
# Stage 4 — Link prediction
# ---------------------------------------------------------------------------

def count_loopfree_paths(
    G: nx.Graph,
    src: int,
    dst: int,
    max_len: int,
) -> dict[int, int]:
    """
    Count loop-free (simple) paths from src to dst with lengths in
    [2, max_len] using iterative DFS with a visited frozenset.

    Returns {length: count}.

    Complexity: O(k^max_len) per pair within the cluster, where k is the
    cluster's average degree.  The clustering step keeps this tractable by
    ensuring cluster sizes << N.

    Note: frozenset is used for visited to allow stack-based iteration
    without backtracking bookkeeping.
    """
    counts: dict[int, int] = defaultdict(int)
    if max_len < 2:
        return counts

    # Stack entries: (current_node, current_depth, visited_frozenset)
    stack: list[tuple[int, int, frozenset[int]]] = [
        (src, 0, frozenset({src}))
    ]

    while stack:
        cur, depth, visited = stack.pop()
        if depth >= max_len:
            continue
        for nbr in G.neighbors(cur):
            if nbr in visited:
                continue
            new_depth = depth + 1
            if nbr == dst and new_depth >= 2:
                counts[new_depth] += 1
            if new_depth < max_len and nbr != dst:
                stack.append((nbr, new_depth, visited | {nbr}))

    return counts


def compute_pair_similarity(
    G: nx.Graph,
    u: int,
    v: int,
    max_path_len: int,
    n_nodes: int,
) -> float:
    """
    Link-prediction similarity between users u and v.  Equation (8):

        sim(vx, vy) = sum_{i=2}^{l} [1/(i-1)] * |paths_i| / prod_{j=2}^{i}(n-j)

    Paths of length i are weighted by attenuation 1/(i-1):
      length 2 -> weight 1.0
      length 3 -> weight 0.5
      length 4 -> weight 0.333  ...

    The denominator normalises by the total number of possible loop-free
    paths of that length in the cluster.

    Parameters
    ----------
    G:
        Subgraph (cluster) in which both u and v reside.
    u, v:
        Target node pair.
    max_path_len:
        Maximum path length l.
    n_nodes:
        Cluster size (used as n in the normalisation denominator).
    """
    path_counts = count_loopfree_paths(G, u, v, max_path_len)
    if not path_counts:
        return 0.0

    sim = 0.0
    for length, cnt in path_counts.items():
        attenuation = 1.0 / (length - 1)
        # prod_{j=2}^{length} (n - j)
        denom = 1.0
        valid = True
        for j in range(2, length + 1):
            factor = n_nodes - j
            if factor <= 0:
                valid = False
                break
            denom *= factor
        if not valid or denom == 0.0:
            continue
        sim += attenuation * cnt / denom

    return sim


def build_similarity_matrix(
    G: nx.Graph,
    nodes: list[int],
    max_path_len: int,
) -> dict[tuple[int, int], float]:
    """
    Compute upper-triangle pairwise similarity for all non-connected node
    pairs in the cluster.  Only non-edges are scored (the purpose is link
    prediction, not scoring existing edges).

    Returns
    -------
    dict[(u, v)] -> similarity  where u < v (upper triangle only)
    """
    n = len(nodes)
    sim_matrix: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            if G.has_edge(u, v):
                continue   # already connected — skip
            s = compute_pair_similarity(G, u, v, max_path_len, n)
            if s > 0.0:
                sim_matrix[(u, v)] = s

    return sim_matrix


def recommend_for_user(
    sim_matrix: dict[tuple[int, int], float],
    target: int,
    existing_neighbors: set[int],
    top_k: int = 10,
) -> list[int]:
    """
    Extract the top-k candidate friends for a target user from the
    precomputed similarity matrix.

    Only users not already connected to target are considered.
    Results are sorted by similarity score (descending).
    """
    scores: list[tuple[float, int]] = []
    for (u, v), s in sim_matrix.items():
        candidate = v if u == target else (u if v == target else None)
        if candidate is not None and candidate not in existing_neighbors:
            scores.append((s, candidate))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in scores[:top_k]]


# ---------------------------------------------------------------------------
# Stage 5 — Genetic algorithm
# ---------------------------------------------------------------------------

def decode_chromosome(chrom: np.ndarray) -> tuple[int, int]:
    """
    Decode an 8-bit chromosome into (n_clusters, path_length).

    Bits 0-3 encode cluster count C in [1, 15].
    Bits 4-7 encode path length L in [2, 15].

    Values of 0 are clamped to their minimum valid values.
    """
    n_clusters = int("".join(str(b) for b in chrom[:4].astype(int)), 2)
    path_len = int("".join(str(b) for b in chrom[4:].astype(int)), 2)
    return max(1, n_clusters), max(2, path_len)


def encode_chromosome(n_clusters: int, path_len: int) -> np.ndarray:
    """Encode (n_clusters, path_len) into an 8-bit chromosome array."""
    c_bin = format(min(15, max(1, n_clusters)), "04b")
    l_bin = format(min(15, max(2, path_len)), "04b")
    return np.array(list(c_bin + l_bin), dtype=np.uint8)


def evaluate_fitness(
    G: nx.Graph,
    profile: ProfileData,
    n_clusters: int,
    path_len: int,
    val_edges: list[tuple[int, int]],
    alpha: float,
    sample_size: int = 50,
) -> float:
    """
    Evaluate a (n_clusters, path_len) candidate using mean(precision, recall)
    on a small validation sample.  Section 3.2.4: fitness = average of the
    two metrics.

    A training graph is built by removing val_edges, then clustering and
    link prediction are run for a sampled subset of users whose connections
    appear in val_edges.
    """
    if not val_edges:
        return 0.0

    G_train = G.copy()
    G_train.remove_edges_from(val_edges)

    try:
        weighted = build_weighted_graph(G_train, profile, alpha)
        clusters = detect_communities(weighted, n_clusters_hint=n_clusters)
    except Exception:
        return 0.0

    node_to_cluster: dict[int, int] = {
        node: cid for cid, nodes in clusters.items() for node in nodes
    }

    # Ground-truth: held-out connections per user
    gt: dict[int, set[int]] = defaultdict(set)
    for u, v in val_edges:
        gt[u].add(v)
        gt[v].add(u)

    val_users = list(gt.keys())
    random.shuffle(val_users)
    sample_users = val_users[:sample_size]

    total_prec = 0.0
    total_rec = 0.0
    evaluated = 0

    for target in sample_users:
        cid = node_to_cluster.get(target)
        if cid is None:
            continue
        cluster_nodes = clusters[cid]
        subgraph = G_train.subgraph(cluster_nodes).copy()
        sim_matrix = build_similarity_matrix(subgraph, cluster_nodes, path_len)
        existing = set(G_train.neighbors(target))
        recs = recommend_for_user(sim_matrix, target, existing, top_k=10)

        actual = gt.get(target, set())
        if not actual:
            continue

        hits = len(set(recs) & actual)
        total_prec += hits / len(recs) if recs else 0.0
        total_rec += hits / len(actual)
        evaluated += 1

    if evaluated == 0:
        return 0.0

    avg_prec = total_prec / evaluated
    avg_rec = total_rec / evaluated
    return (avg_prec + avg_rec) / 2.0


def _tournament_select(
    population: list[np.ndarray],
    fitnesses: list[float],
    k: int = 3,
) -> np.ndarray:
    """
    Tournament selection: draw k candidates at random, return the fittest.
    """
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def _two_point_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover on length-8 chromosomes.
    Two distinct cut-points are chosen; the middle segment is swapped.
    """
    pts = sorted(random.sample(range(1, len(p1)), 2))
    c1 = np.concatenate([p1[: pts[0]], p2[pts[0]: pts[1]], p1[pts[1]:]])
    c2 = np.concatenate([p2[: pts[0]], p1[pts[0]: pts[1]], p2[pts[1]:]])
    return c1, c2


def _mutate(chrom: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Flip a randomly selected bit with probability mutation_rate.
    Operates on a copy of the chromosome.
    """
    result = chrom.copy()
    if random.random() < mutation_rate:
        pos = random.randint(0, len(result) - 1)
        result[pos] = 1 - result[pos]
    return result


def run_genetic_algorithm(
    G: nx.Graph,
    profile: ProfileData,
    val_edges: list[tuple[int, int]],
    alpha: float,
    ga_params: GAParams,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Run the genetic algorithm to find optimal (n_clusters, path_length).

    Algorithm (Section 3.2.4):
      - Population: Np=150 chromosomes, each 8 bits.
      - Elitism: top Ps=70% passed directly to next generation.
      - Crossover: two-point on remaining slots; parents by tournament.
      - Mutation: each chromosome flips one bit with probability Pm=0.05.
      - Stop after G=300 generations or when best fitness >= T=1.0.

    Returns
    -------
    (n_clusters, path_length) of the best chromosome found.
    """
    Np = ga_params.population_size
    rng = np.random.default_rng(42)

    population: list[np.ndarray] = [
        rng.integers(0, 2, size=8, dtype=np.uint8) for _ in range(Np)
    ]

    best_fitness = 0.0
    best_chrom = population[0].copy()

    for gen in range(ga_params.max_generations):
        fitnesses: list[float] = []
        for chrom in population:
            nc, pl = decode_chromosome(chrom)
            f = evaluate_fitness(
                G, profile, nc, pl, val_edges, alpha, ga_params.ga_sample_size
            )
            fitnesses.append(f)

        gen_best = int(np.argmax(fitnesses))
        if fitnesses[gen_best] > best_fitness:
            best_fitness = fitnesses[gen_best]
            best_chrom = population[gen_best].copy()

        if verbose:
            nc, pl = decode_chromosome(best_chrom)
            print(
                f"  GA gen {gen+1:3d}/{ga_params.max_generations}  "
                f"best_fitness={best_fitness:.4f}  "
                f"n_clusters={nc}  path_len={pl}"
            )

        if best_fitness >= ga_params.fitness_threshold:
            break

        # Build next generation
        sorted_pairs = sorted(
            zip(fitnesses, population), key=lambda x: x[0], reverse=True
        )
        n_elite = max(1, int(ga_params.crossover_rate * Np))
        elite = [chrom.copy() for _, chrom in sorted_pairs[:n_elite]]
        elite_fitnesses = [f for f, _ in sorted_pairs[:n_elite]]

        offspring: list[np.ndarray] = []
        n_needed = Np - n_elite
        while len(offspring) < n_needed:
            p1 = _tournament_select(elite, elite_fitnesses, ga_params.tournament_k)
            p2 = _tournament_select(elite, elite_fitnesses, ga_params.tournament_k)
            c1, c2 = _two_point_crossover(p1, p2)
            offspring.append(_mutate(c1, ga_params.mutation_rate))
            if len(offspring) < n_needed:
                offspring.append(_mutate(c2, ga_params.mutation_rate))

        population = elite + offspring[:n_needed]

    return decode_chromosome(best_chrom)


# ---------------------------------------------------------------------------
# Stage 6 — Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    recommended: list[int],
    actual: set[int],
    total_negatives: int,
) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and F1 for one user.

    TP: recommended and actually connected
    FP: recommended but not connected
    FN: connected but not recommended
    TN: not recommended and not connected (estimated from total_negatives)

    Equations (9)-(12) from the paper.
    """
    recs = set(recommended)
    tp = len(recs & actual)
    fp = len(recs - actual)
    fn = len(actual - recs)
    tn = max(0, total_negatives - fp)

    denom_acc = tp + fp + tn + fn
    accuracy = (tp + tn) / denom_acc if denom_acc > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_single_fold(
    G_full: nx.Graph,
    profile: ProfileData,
    train_edges: list[tuple[int, int]],
    test_edges: list[tuple[int, int]],
    params: ModelParams,
    fold_idx: int,
) -> FoldResult:
    """
    Execute one cross-validation fold:
      1. Build training graph (test edges removed).
      2. Weight graph and detect communities.
      3. Build the similarity matrix ONCE PER CLUSTER (key optimisation:
         all users in the same cluster share the same matrix).
      4. For each test user, look up recommendations from the cached matrix.
      5. Aggregate accuracy/precision/recall/F1 over all test users.

    The per-cluster caching avoids recomputing the O(n_c^2 * k_c^L)
    similarity matrix for every individual test user.
    """
    t0 = time.time()

    G_train = G_full.copy()
    G_train.remove_edges_from(test_edges)

    weighted = build_weighted_graph(G_train, profile, params.alpha)
    clusters = detect_communities(weighted, n_clusters_hint=params.n_clusters)

    node_to_cluster: dict[int, int] = {
        node: cid for cid, nodes in clusters.items() for node in nodes
    }

    # Build per-user ground truth from test edges
    gt: dict[int, set[int]] = defaultdict(set)
    for u, v in test_edges:
        gt[u].add(v)
        gt[v].add(u)

    # Pre-build one similarity matrix per cluster that contains at least
    # one test user — this is the core efficiency improvement.
    clusters_with_test_users: set[int] = set()
    for target in gt:
        cid = node_to_cluster.get(target)
        if cid is not None:
            clusters_with_test_users.add(cid)

    cluster_sim_matrices: dict[int, dict[tuple[int, int], float]] = {}
    cluster_subgraphs: dict[int, nx.Graph] = {}
    for cid in clusters_with_test_users:
        cluster_nodes = clusters[cid]
        sg = G_train.subgraph(cluster_nodes).copy()
        cluster_subgraphs[cid] = sg
        cluster_sim_matrices[cid] = build_similarity_matrix(
            sg, cluster_nodes, params.path_length
        )

    n_total = G_full.number_of_nodes()
    all_metrics: list[dict[str, float]] = []

    for target, actual in gt.items():
        cid = node_to_cluster.get(target)
        if cid is None or cid not in cluster_sim_matrices:
            all_metrics.append(
                {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            )
            continue

        sim_matrix = cluster_sim_matrices[cid]
        existing = set(G_train.neighbors(target))
        recs = recommend_for_user(sim_matrix, target, existing, params.top_k)

        total_neg = max(0, n_total - 1 - len(existing))
        m = compute_metrics(recs, actual, total_neg)
        all_metrics.append(m)

    if not all_metrics:
        return FoldResult(fold_idx, 0.0, 0.0, 0.0, 0.0, time.time() - t0)

    avg = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in ("accuracy", "precision", "recall", "f1")
    }
    return FoldResult(
        fold=fold_idx,
        accuracy=avg["accuracy"],
        precision=avg["precision"],
        recall=avg["recall"],
        f1=avg["f1"],
        elapsed_sec=time.time() - t0,
    )


def k_fold_cross_validation(
    G: nx.Graph,
    profile: ProfileData,
    params: ModelParams,
    k: int = 10,
    seed: int = 42,
) -> list[FoldResult]:
    """
    10-fold cross-validation (Section 3.1).

    Edges are shuffled and partitioned into k equal folds.
    Each fold is held out as the test set once; the remaining k-1 folds
    form the training graph.

    The Pokec dataset lacks timestamps, making this the standard evaluation
    protocol used by the paper.
    """
    rng = random.Random(seed)
    edges = list(G.edges())
    rng.shuffle(edges)

    fold_size = len(edges) // k
    results: list[FoldResult] = []

    for fold in range(k):
        test_edges = edges[fold * fold_size: (fold + 1) * fold_size]
        train_edges = edges[: fold * fold_size] + edges[(fold + 1) * fold_size:]

        print(
            f"\n[Fold {fold+1}/{k}]  "
            f"train={len(train_edges)}  test={len(test_edges)}"
        )
        result = run_single_fold(G, profile, train_edges, test_edges, params, fold)
        results.append(result)
        print(
            f"  acc={result.accuracy:.4f}  prec={result.precision:.4f}  "
            f"rec={result.recall:.4f}  f1={result.f1:.4f}  "
            f"({result.elapsed_sec:.1f}s)"
        )

    return results


def print_cv_summary(results: list[FoldResult]) -> None:
    """Print aggregate statistics across all folds."""
    line = "-" * 62
    print(f"\n{line}")
    print("Cross-Validation Summary")
    print(line)
    for metric in ("accuracy", "precision", "recall", "f1"):
        vals = [getattr(r, metric) for r in results]
        print(
            f"  {metric:<12}  mean={np.mean(vals):.4f}  "
            f"std={np.std(vals):.4f}  "
            f"min={np.min(vals):.4f}  max={np.max(vals):.4f}"
        )
    total_time = sum(r.elapsed_sec for r in results)
    print(f"\n  Total wall time : {total_time:.1f}s")
    print(f"  Folds evaluated : {len(results)}")
    print(line)


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_users: int = 400,
    avg_degree: int = 10,
    n_communities: int = 6,
    seed: int = 42,
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Generate a synthetic social network for pipeline testing.

    Uses a stochastic block model (SBM) to embed community structure:
    intra-block edge probability is high; inter-block probability is low.
    User profiles are generated with community-correlated attributes so
    that the profile-similarity weighting provides signal.

    Parameters
    ----------
    n_users:
        Total node count.
    avg_degree:
        Target average degree (used to set edge probabilities).
    n_communities:
        Number of ground-truth communities (SBM blocks).
    seed:
        Reproducibility seed.

    Returns
    -------
    (G, profile_df)
        G           : undirected nx.Graph
        profile_df  : DataFrame indexed by node_id with PROFILE_ATTRS columns
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    block_size = n_users // n_communities
    sizes = [block_size] * n_communities
    sizes[-1] += n_users - sum(sizes)

    # Probability calibration: p_intra dominates, p_inter is small
    p_intra = min(0.8, avg_degree / block_size)
    p_inter = max(0.002, (avg_degree - p_intra * block_size) / max(1, n_users - block_size))

    p_matrix = [
        [p_intra if i == j else p_inter for j in range(n_communities)]
        for i in range(n_communities)
    ]

    G_sbm = nx.stochastic_block_model(sizes, p_matrix, seed=int(seed))
    G = nx.Graph(G_sbm)
    G.remove_edges_from(nx.selfloop_edges(G))

    node_ids = list(G.nodes())

    languages = ["slovak", "czech", "english", "hungarian", "german"]
    occupations = [
        "student", "engineer", "teacher", "doctor",
        "manager", "artist", "scientist", "other",
    ]
    educations = list(EDUCATION_MAP.keys())

    # Assign community membership to each node
    comm_assignment: list[int] = []
    for cid, sz in enumerate(sizes):
        comm_assignment.extend([cid] * sz)

    profiles: list[dict] = []
    for i, nid in enumerate(node_ids):
        comm = comm_assignment[i]

        # Age: community biases the mean age
        age = int(rng.normal(20.0 + comm * 5.0, 5.0))
        age = max(10, min(70, age))

        gender = rng.choice(["male", "female", "unspecified"])
        language = languages[comm % len(languages)]   # community-correlated
        occupation = str(rng.choice(occupations))
        education = str(rng.choice(educations))

        profiles.append({
            "node_id": int(nid),
            "age": age,
            "gender": str(gender),
            "language": language,
            "occupation": occupation,
            "education": education,
        })

    df = pd.DataFrame(profiles).set_index("node_id")
    return G, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end pipeline demonstration on a synthetic dataset.

    Steps
    -----
    1. Generate a synthetic social network with embedded community structure.
    2. Preprocess user profiles.
    3. Optionally run the genetic algorithm to find optimal parameters.
    4. Run 10-fold cross-validation with the chosen parameters.
    5. Print a summary and demonstrate a single-user recommendation.
    """
    separator = "=" * 62
    print(separator)
    print("Social Network Recommender System")
    print("Lu et al., Egyptian Informatics Journal 32 (2025) 100784")
    print(separator)

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------
    print("\n[1] Generating synthetic dataset...")
    G, profile_df = generate_synthetic_dataset(
        n_users=400,
        avg_degree=10,
        n_communities=6,
        seed=42,
    )
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(
        f"    Nodes     : {n_nodes}\n"
        f"    Edges     : {n_edges}\n"
        f"    Avg degree: {2 * n_edges / n_nodes:.2f}"
    )

    # ------------------------------------------------------------------
    # 2. Profile preprocessing
    # ------------------------------------------------------------------
    print("\n[2] Preprocessing user profiles...")
    node_order = list(G.nodes())
    profile = build_profile_data(profile_df, node_order)
    print(
        f"    Profile matrix : {profile.vectors.shape}\n"
        f"    Age range [0,1]: "
        f"[{profile.vectors[:,0].min():.3f}, {profile.vectors[:,0].max():.3f}]"
    )

    # ------------------------------------------------------------------
    # 3. Parameter optimisation (genetic algorithm)
    #    Set run_ga=True to search; False uses the paper's best values.
    # ------------------------------------------------------------------
    run_ga = False

    if run_ga:
        print("\n[3] Running genetic algorithm for parameter optimisation...")
        edges = list(G.edges())
        random.shuffle(edges)
        val_edges = edges[: max(10, len(edges) // 10)]

        ga_params = GAParams(
            population_size=20,    # reduced for demo speed
            max_generations=10,
            ga_sample_size=20,
        )
        best_nc, best_pl = run_genetic_algorithm(
            G, profile, val_edges,
            alpha=0.5,
            ga_params=ga_params,
            verbose=True,
        )
        print(f"\n    GA result: n_clusters={best_nc}, path_length={best_pl}")
    else:
        best_nc, best_pl = 9, 3
        print(
            f"\n[3] Using paper's optimal parameters: "
            f"n_clusters={best_nc}, path_length={best_pl}"
        )

    # ------------------------------------------------------------------
    # 4. Cross-validation
    # ------------------------------------------------------------------
    print("\n[4] Running 10-fold cross-validation...")
    params = ModelParams(
        alpha=0.5,
        n_clusters=best_nc,
        path_length=best_pl,
        top_k=10,
    )
    results = k_fold_cross_validation(G, profile, params, k=10, seed=42)
    print_cv_summary(results)

    # ------------------------------------------------------------------
    # 5. Single-user recommendation demo
    # ------------------------------------------------------------------
    print("\n[5] Single-user recommendation demo...")
    weighted_G = build_weighted_graph(G, profile, params.alpha)
    clusters = detect_communities(weighted_G, n_clusters_hint=params.n_clusters)

    node_to_cluster = {
        node: cid for cid, nodes in clusters.items() for node in nodes
    }

    target = node_order[0]
    cid = node_to_cluster.get(target, 0)
    cluster_nodes = clusters[cid]
    subgraph = G.subgraph(cluster_nodes).copy()

    sim_matrix = build_similarity_matrix(subgraph, cluster_nodes, params.path_length)
    existing = set(G.neighbors(target))
    recs = recommend_for_user(sim_matrix, target, existing, top_k=5)

    print(f"    Target user            : {target}")
    print(f"    Cluster id             : {cid}")
    print(f"    Cluster size           : {len(cluster_nodes)}")
    print(f"    Existing connections   : {len(existing)}")
    print(f"    Top-5 recommendations  : {recs}")
    print(f"    Total communities found: {len(clusters)}")
    print()


if __name__ == "__main__":
    main()
