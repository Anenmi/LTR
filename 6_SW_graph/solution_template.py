from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # допишите ваш код здесь 
    return np.linalg.norm(pointA-documents, axis=1)[:, np.newaxis]


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    # допишите ваш код здесь 
    result = defaultdict()
    for i, point in enumerate(data): 
        distances = dist_f(point, data)
        candidates = np.argsort(distances, axis=0)

        candidates_long = candidates[-num_candidates_for_choice_long:]
        edges_long = candidates_long[np.random.choice(num_candidates_for_choice_long, num_edges_long, replace=False)]
        
        candidates_short = candidates[candidates != i][:num_candidates_for_choice_short+1]
        candidates_short = candidates_short[candidates_short != i][:num_candidates_for_choice_short]
        edges_short = candidates_short[np.random.choice(num_candidates_for_choice_short, num_edges_short, replace=False)]

        candidates = np.unique(np.concatenate([edges_long.flatten(), edges_short.flatten()])).tolist()

        result[i] = candidates
    return result

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # допишите ваш код здесь 
    numb_per_point = search_k // num_start_points + 1
    k_neighbors = defaultdict()
    entry_points = np.random.choice(all_documents.shape[0], num_start_points, replace=False)
    for i in range(num_start_points):
        entry_point = entry_points[i]
        new_point = entry_point
        prev_point=-1
        candidates = {}
        while prev_point!=new_point:
            point_edges = graph_edges[new_point]
            distances = dist_f(query_point, all_documents[point_edges + [new_point]])
            prev_point = new_point
            new_point = (point_edges + [new_point])[np.argmin(distances)]
            new_dist = np.min(distances)
            k_neighbors[new_point] = new_dist
    return list(dict(sorted(k_neighbors.items(), key=lambda x: x[1])[:search_k]).keys())