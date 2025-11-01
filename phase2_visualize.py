import math
import matplotlib.pyplot as plt
import networkx as nx


def dijkstra(graph, start, goal):
    nodes = set(graph.keys())
    dist = {n: math.inf for n in nodes}
    prev = {n: None for n in nodes}
    dist[start] = 0.0
    while nodes:
        cur = min(nodes, key=lambda n: dist[n])
        if dist[cur] == math.inf:
            break
        nodes.remove(cur)
        if cur == goal:
            break
        for nbr, w in graph[cur].items():
            alt = dist[cur] + w
            if alt < dist[nbr]:
                dist[nbr] = alt
                prev[nbr] = cur
    if dist[goal] == math.inf:
        return [], math.inf
    path = []
    n = goal
    while n:
        path.append(n)
        n = prev[n]
    path.reverse()
    return path, dist[goal]


def visualize_phase1(prediction_index: float = 0.1717, show_plot: bool = True):
    # Step 1: Static map (minutes) - same as run_phase1
    static = {
        "S": {"M": 5.0, "H": 8.0},
        "M": {"Home": 15.0},
        "H": {"Home": 1.0},
        "Home": {}
    }

    # Step 1: Build dynamic using same threshold rule used in run_phase1
    congestion_factors = {
        ("S", "H"): 2.5 if prediction_index > 0.15 else 1.0,
        ("S", "M"): 1.0,
        ("M", "Home"): 1.0,
        ("H", "Home"): 1.0
    }

    dynamic = {}
    for u, neigh in static.items():
        dynamic[u] = {}
        for v, base_w in neigh.items():
            factor = congestion_factors.get((u, v), 1.0)
            dynamic[u][v] = base_w * factor

    # Step 2: Compute optimal route (same dijkstra)
    path, cost = dijkstra(dynamic, "S", "Home")

    # Step 3: Build NetworkX directed graph for plotting
    G = nx.DiGraph()
    for u, neigh in dynamic.items():
        for v, w in neigh.items():
            G.add_edge(u, v, weight=w)

    # Layout: place nodes for clear map
    pos = {
        "S": (0, 0),
        "M": (1, 1),
        "H": (1, -1),
        "Home": (2, 0)
    }

    # Determine edge colors: default green, heavy edge S->H red
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) == ("S", "H") and congestion_factors.get(("S", "H"), 1.0) > 1.0:
            edge_colors.append("red")   # heavy traffic
            edge_widths.append(3.0)
        else:
            edge_colors.append("green")
            edge_widths.append(1.5)

    # Highlight the optimal path edges (blue, thicker) if path exists
    path_edges = set()
    for i in range(len(path) - 1):
        path_edges.add((path[i], path[i+1]))
    # update colors/widths for path edges
    new_colors = []
    new_widths = []
    for idx, (u, v) in enumerate(G.edges()):
        if (u, v) in path_edges:
            new_colors.append("blue")
            new_widths.append(4.0)
        else:
            new_colors.append(edge_colors[idx])
            new_widths.append(edge_widths[idx])

    # Draw nodes and labels
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color="#f0f0f0", edgecolors="k")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw edges with colors and widths
    nx.draw_networkx_edges(
        G, pos,
        edge_color=new_colors,
        width=new_widths,
        arrowsize=18,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw edge labels with dynamic weights
    edge_labels = {(u, v): f"{d['weight']:.1f}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray", label_pos=0.6, font_size=9)

    # Legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='blue', label='Optimal path (selected)')
    red_patch = mpatches.Patch(color='red', label='Heavy traffic (S->H)')
    green_patch = mpatches.Patch(color='green', label='Normal flow')
    plt.legend(handles=[blue_patch, red_patch, green_patch], loc='lower left')

    title = f"Phase1 Traffic Map — Prediction Index: {prediction_index:.4f}\nOptimal Path: {' -> '.join(path) if path else 'None'}  |  Cost: {cost:.2f} min"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if show_plot:
        plt.show()

    # Also return structured data for programmatic use
    return {
        "static": static,
        "dynamic": dynamic,
        "prediction_index": prediction_index,
        "optimal_path": path,
        "optimal_cost": cost,
        "congestion_factors": congestion_factors
    }


if __name__ == "__main__":
    # Quick run to reproduce the exact requested output and visualization
    result = visualize_phase1()
    print(f"Prediction Index: {result['prediction_index']:.4f}")
    s_h_factor = result['congestion_factors'][('S','H')]
    print(f"Applied congestion factor on S->H: {s_h_factor:.1f}x\n")
    if result['optimal_path']:
        print(f"Optimal Path found with predicted congestion: {' -> '.join(result['optimal_path'])}")
        print(f"Predicted Travel Cost (Weight): {result['optimal_cost']:.2f} min")
    else:
        print("No path found from S to Home.")