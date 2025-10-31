# graph/road_graph.py - Data Structures and Mathematics Core
import networkx as nx
from typing import Dict, Any

class RoadGraph:
    """
    Models the road network as a graph (BCS304 - Data Structures). 
    Enables dynamic weight updates based on AI predictions (BCS301 - Mathematics). 
    """
    def __init__(self, junctions: list, road_segments: list):
        # Create a Directed Graph for road travel
        self.graph = nx.DiGraph() 
        self.graph.add_nodes_from(junctions)
        for u, v, default_weight in road_segments:
            # Edges are road segments, default weight is static travel time
            self.graph.add_edge(u, v, weight=default_weight, congestion_factor=1.0) 

    def update_weights(self, predicted_density: Dict[str, float]):
        """
        Dynamically adjusts road costs using the AI's prediction.
        """
        print("Applying AI congestion predictions to graph weights...")
        for u, v, data in self.graph.edges(data=True):
            # Assume congestion at the 'to' node (v) impacts the travel time
            junction_id = v 
            
            # Normalize the prediction (0 to 1) into a congestion factor (e.g., 1.0 to 3.0)
            raw_density = predicted_density.get(junction_id, 0.0)
            # HIGH density (1.0) leads to a factor of 3.0 (tripling travel time)
            congestion_factor = 1.0 + (raw_density * 2.0) 
            
            # Dynamic Weight = Static Base Weight * Congestion Factor
            new_weight = data['weight'] * congestion_factor
            
            self.graph.edges[u, v]['dynamic_weight'] = new_weight
            
        print("Weights successfully updated for dynamic routing.")

    def find_shortest_path(self, start_node: str, end_node: str, weight_attribute: str = 'dynamic_weight') -> list:
        """
        Calculates the optimal route based on the predicted travel time/cost.
        This function satisfies the Mathematics for CS optimization requirement. 
        """
        try:
            # We use Dijkstra's algorithm (built into NetworkX) on the dynamically updated weights.
            path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight=weight_attribute)
            path_length = nx.shortest_path_length(self.graph, source=start_node, target=end_node, weight=weight_attribute)
            print(f"Optimal Path found with predicted congestion: {path}")
            print(f"Predicted Travel Cost (Weight): {path_length:.2f}")
            return path
        except nx.NetworkXNoPath:
            return f"No path found between {start_node} and {end_node}."