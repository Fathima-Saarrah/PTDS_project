import math
from typing import Dict, Tuple, List

def dijkstra(graph: Dict[str, Dict[str, float]], start: str, goal: str) -> Tuple[List[str], float]:
    unvisited = set(graph.keys())
    dist = {node: math.inf for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0.0
    while unvisited:
        cur = min((n for n in unvisited), key=lambda n: dist[n])
        if dist[cur] == math.inf:
            break
        unvisited.remove(cur)
        if cur == goal:
            break
        for nbr, w in graph[cur].items():
            alt = dist[cur] + w
            if alt < dist[nbr]:
                dist[nbr] = alt
                prev[nbr] = cur
    # Reconstruct path
    path = []
    node = goal
    if dist[goal] == math.inf:
        return [], math.inf
    while node:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[goal]

def run_phase1(prediction_index: float = 0.1717):
    # Static map (minutes)
    static = {
        "S": {"M": 5.0, "H": 8.0},
        "M": {"Home": 15.0},
        "H": {"Home": 1.0},
        "Home": {}
    }

    # Decide congestion factor from prediction index (simple threshold mapping)
    # Here: index > 0.15 => mark H as heavily congested (2.5x)
    congestion_factors = {
        ("S", "H"): 2.5 if prediction_index > 0.15 else 1.0,
        ("S", "M"): 1.0,
        ("M", "Home"): 1.0,
        ("H", "Home"): 1.0
    }

    # Build dynamic-weight graph
    dynamic = {}
    for u, neighbors in static.items():
        dynamic[u] = {}
        for v, base_w in neighbors.items():
            factor = congestion_factors.get((u, v), 1.0)
            dynamic[u][v] = base_w * factor

    # Compute shortest path
    path, cost = dijkstra(dynamic, "S", "Home")

    # Print execution log matching requested format
    print(f"Prediction Index: {prediction_index:.4f}")
    print(f"Applied congestion factor on S->H: {congestion_factors[('S','H')]:.1f}x")
    print()
    if path:
        nice_path = " -> ".join(path)
        print(f"Optimal Path found with predicted congestion: {nice_path}")
        print(f"Predicted Travel Cost (Weight): {cost:.2f} min")
    else:
        print("No path found from S to Home.")

if __name__ == "__main__":
    run_phase1()