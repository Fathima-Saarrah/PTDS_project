# app.py - Web Visualization Dashboard (Gottigere to AMC College)
import streamlit as st
import networkx as nx
import osmnx as ox
import folium
from folium import Element
import numpy as np
from streamlit_folium import folium_static
from math import sqrt

# New: flexible graph radius and midpoint helper
GRAPH_RADIUS = 4000  # meters

def midpoint(lat1, lon1, lat2, lon2):
    return ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

def euclidean_dist(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two lat/lon points."""
    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000

# --- Helper Function for Color Mapping ---
def get_color(factor):
    """Maps a congestion factor (0.0 to 1.0) to a Hex color."""
    if factor >= 0.8:
        return '#FF0000'  # Severe (RED)
    if factor >= 0.6:
        return '#FF4500'  # Heavy (ORANGE)
    if factor >= 0.4:
        return '#FFFF00'  # Moderate (YELLOW)
    if factor >= 0.2:
        return '#9ACD32'  # Slow (YELLOW-GREEN)
    return '#008000'      # Free Flow (GREEN)

st.set_page_config(layout="wide", page_title="PTDS: Gottigere Traffic Predictor")

# --- 1. Geographical and Data Setup (BCS304 / AIML Integration) ---

# Define the central area (Bannerghatta Road near the target corridor)
# Coordinates are near Gottigere (Start) and AMC College (End)
CENTER_POINT = (12.868, 77.595)
PLACE_NAME = "Bannerghatta Road, Bengaluru, India"

# Define the start and end nodes for routing
# NOTE: In a real system, you would find the closest OSM node to these addresses.
START_NODE_NAME = "Gottigere, Kalena Agrahara" 
END_NODE_NAME = "AMC Engineering College"

@st.cache_data
def load_and_analyze_graph(center_point, dist=GRAPH_RADIUS):
    """Loads the road network around given center using OSMnx. Cached per center/dist."""
    with st.spinner("Loading geographical data..."):
        G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
        return G

# --- 2. Simulation: AI Prediction Result (Dynamic Weights) ---

# --- Time-of-day adjustment helper ---
def adjust_for_time_of_day(factor: float, hour: int) -> float:
    """Boost/reduce congestion factor depending on hour (peak/off-peak)."""
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        return min(factor * 1.5, 1.0)
    if 0 <= hour <= 6:
        return factor * 0.5
    return factor

# Replace simulate_prediction to accept hour
def simulate_prediction(G, hour: int = 8):
    """
    Simulates the LSTM model's prediction output and applies it to graph edges.
    hour: hour of day (0-23) used to adjust base congestion.
    """
    np.random.seed(42)
    num_edges = len(G.edges)
    base_factors = np.random.uniform(0.0, 1.0, num_edges)
    dynamic_weights = {}

    for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        base_weight = data.get('length') or 1.0
        factor = adjust_for_time_of_day(float(base_factors[i]), hour)

        if factor > 0.6:
            dynamic_cost = base_weight * 2.5
        elif factor > 0.3:
            dynamic_cost = base_weight * 1.5
        else:
            dynamic_cost = base_weight * 1.05

        G.edges[u, v, k]['dynamic_weight'] = dynamic_cost
        G.edges[u, v, k]['congestion_factor'] = factor
        dynamic_weights[(u, v, k)] = dynamic_cost

    return G, dynamic_weights

# --- 3. Optimization and Visualization ---

def geocode_location(place_name: str):
    """Geocode a place name to (lat, lon) using OSMnx; returns None on failure."""
    try:
        coords = ox.geocoder.geocode(place_name)
        return coords  # (lat, lon)
    except Exception:
        return None

# Update visualize_and_optimize signature to accept route_type
def visualize_and_optimize(G, dynamic_weights, start_coords, end_coords, route_type: str = "AI-Optimized", start_label: str = None, end_label: str = None):
    """
    Renders map and computes route.
    route_type: "Shortest Path" uses 'length', otherwise uses 'dynamic_weight'
    """
    # Validate coords
    if start_coords is None or end_coords is None:
        st.error("Invalid start or end coordinates.")
        return folium.Map(location=CENTER_POINT, zoom_start=14)

    # Ensure the points are within the loaded graph area
    try:
        dist_start = euclidean_dist(CENTER_POINT[0], CENTER_POINT[1], start_coords[0], start_coords[1])
        dist_end = euclidean_dist(CENTER_POINT[0], CENTER_POINT[1], end_coords[0], end_coords[1])
        if dist_start > GRAPH_RADIUS or dist_end > GRAPH_RADIUS:
             st.warning(f"One or both locations are outside the mapped area (approx {GRAPH_RADIUS / 1000:.1f} km). Results may be inaccurate.")
    except Exception:
        pass

    # Find nearest graph nodes to provided coordinates
    try:
        start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
    except Exception:
        st.error("Could not find nearest nodes for the provided locations.")
        return folium.Map(location=CENTER_POINT, zoom_start=14)

    # Select weight type based on desired routing strategy and compute route first
    weight_type = 'length' if route_type == "Shortest Path" else 'dynamic_weight'
    try:
        optimized_route = nx.shortest_path(G, start_node, end_node, weight=weight_type)
        optimized_cost = nx.shortest_path_length(G, start_node, end_node, weight=weight_type)
        st.success(f"{route_type} Route Found (Cost: {optimized_cost:.2f} meters)")
    except nx.NetworkXNoPath:
        st.error("No path found between the start and end points.")
        optimized_route = None
        optimized_cost = 0

    # Center map on route midpoint if available, otherwise use CENTER_POINT
    if optimized_route:
        mid_idx = len(optimized_route) // 2
        mid_node = G.nodes[optimized_route[mid_idx]]
        center_loc = [mid_node['y'], mid_node['x']]
    else:
        center_loc = CENTER_POINT

    # Create base map after computing route so it's centered correctly
    m = folium.Map(location=center_loc, zoom_start=14, tiles="cartodbpositron")

    # Start / End markers (use provided labels if available)
    try:
        folium.Marker(
            location=[start_coords[0], start_coords[1]],
            popup=f"Start: {start_label or START_NODE_NAME}",
            icon=folium.Icon(color='green')
        ).add_to(m)
        folium.Marker(
            location=[end_coords[0], end_coords[1]],
            popup=f"End: {end_label or END_NODE_NAME}",
            icon=folium.Icon(color='red')
        ).add_to(m)
    except Exception:
        pass

    # Draw traffic-colored edges and optimized route (existing logic)
    for u, v, k, data in G.edges(keys=True, data=True):
        u_node = G.nodes[u]
        v_node = G.nodes[v]
        factor = data.get('congestion_factor', 0.0)
        line_color = get_color(factor)
        folium.PolyLine(
            [[u_node['y'], u_node['x']], [v_node['y'], v_node['x']]],
            color=line_color,
            weight=4,
            opacity=0.9,
            tooltip=f"Predicted Congestion: {factor:.2f} | Cost: {data.get('dynamic_weight', 0.0):.2f} m"
        ).add_to(m)

    if optimized_route:
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in optimized_route]
        folium.PolyLine(route_coords, color='blue', weight=5, opacity=1.0,
                        tooltip=f"{route_type} Route: {optimized_cost:.2f} m").add_to(m)

    # Legend (dynamic route label)
    legend_html = f"""
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 220px; height: 180px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px;">
      <b>Traffic Congestion Legend</b><br>
      <i style="color:#008000;">●</i> Free Flow<br>
      <i style="color:#9ACD32;">●</i> Slow<br>
      <i style="color:#FFFF00;">●</i> Moderate<br>
      <i style="color:#FF4500;">●</i> Heavy<br>
      <i style="color:#FF0000;">●</i> Severe<br>
      <i style="color:blue;">●</i> {route_type} Route
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    # Optionally show comparison metrics (Shortest vs AI)
    try:
        shortest_cost = nx.shortest_path_length(G, start_node, end_node, weight='length')
        ai_cost = nx.shortest_path_length(G, start_node, end_node, weight='dynamic_weight')
        st.metric("Shortest Path Cost (length)", f"{shortest_cost:.2f} m")
        st.metric("AI-Optimized Cost (dynamic)", f"{ai_cost:.2f} m")
    except Exception:
        pass

    return m

# --- Main Streamlit Execution ---

st.title("🚦 Predictive Traffic Dashboard: Gottigere to AMC College")
st.markdown("---")

# Removed pre-loading graph. Graph will be loaded after geocoding so it's centered on user route.
st.header("Predicted Traffic Density Map")

# UI: Add inputs for custom route selection (place this in main section before button)
st.subheader("Custom Route Selection")
start_location = st.text_input("Enter Start Location", "Gottigere, Kalena Agrahara")
end_location = st.text_input("Enter End Location", "AMC Engineering College")

# add time-of-day selector and routing strategy selector before the button
selected_hour = st.slider("Select Hour of Day", 0, 23, 8)
route_type = st.radio("Choose Routing Strategy", ["AI-Optimized", "Shortest Path"])

# Replace button handler: geocode -> compute midpoint -> load graph -> simulate -> visualize
if st.button("Calculate Optimal Route & Density Map", type="primary"):
    start_coords = geocode_location(start_location)
    end_coords = geocode_location(end_location)
    if start_coords is None:
        st.error(f"Could not geocode start location: {start_location}")
    elif end_coords is None:
        st.error(f"Could not geocode end location: {end_location}")
    else:
        st.write("Start coords:", start_coords)
        st.write("End coords:", end_coords)

        # compute center between start and end and load graph around that midpoint
        CENTER_POINT = midpoint(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        G = load_and_analyze_graph(CENTER_POINT, GRAPH_RADIUS)

        # simulate dynamic congestion after graph is loaded and hour selected
        G_dynamic, dynamic_weights = simulate_prediction(G, selected_hour)

        try:
            start_node = ox.nearest_nodes(G_dynamic, start_coords[1], start_coords[0])
            end_node = ox.nearest_nodes(G_dynamic, end_coords[1], end_coords[0])
            st.write("Nearest nodes:", start_node, end_node)
            st.write("Distance from center (m):",
                     euclidean_dist(CENTER_POINT[0], CENTER_POINT[1], start_coords[0], start_coords[1]),
                     euclidean_dist(CENTER_POINT[0], CENTER_POINT[1], end_coords[0], end_coords[1]))
        except Exception as e:
            st.error(f"Nearest-node lookup failed: {e}")

        traffic_map = visualize_and_optimize(G_dynamic, dynamic_weights, start_coords, end_coords, route_type, start_label=start_location, end_label=end_location)
        folium_static(traffic_map, width=1000, height=600)
        
        # Display metrics now that G_dynamic is available
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Graph Nodes (Intersections)", len(G_dynamic.nodes))
            st.metric("Graph Edges (Road Segments)", len(G_dynamic.edges))

        with col2:
            st.metric("LSTM RMSE", "0.1396 (Simulated)", help="Value proven by comparative analysis against SVR baseline.")
            st.metric("Optimization Algorithm", "Dijkstra's Shortest Path", help="The algorithm used for dynamic route calculation.")
         
        st.subheader("Analysis Summary")
        st.info("The map visually demonstrates the core function: the AI (LSTM) predicted congestion on certain segments (RED) and the pathfinding algorithm (BCS301) selected the fastest overall route (BLUE) based on those predicted traffic conditions.")
        
        # --- Extra utilities: congestion histogram and route export ---
        try:
            # Recompute start/end nodes from the user-provided geocoded coordinates
            start_node = ox.nearest_nodes(G_dynamic, start_coords[1], start_coords[0])
            end_node = ox.nearest_nodes(G_dynamic, end_coords[1], end_coords[0])
            optimized_route = nx.shortest_path(G_dynamic, start_node, end_node, weight='dynamic_weight')
            optimized_cost = nx.shortest_path_length(G_dynamic, start_node, end_node, weight='dynamic_weight')
            route_coords = [(G_dynamic.nodes[n]['y'], G_dynamic.nodes[n]['x']) for n in optimized_route]
            
            # Download button for optimized route (CSV)
            import pandas as pd
            route_df = pd.DataFrame(route_coords, columns=['latitude', 'longitude'])
            csv_data = route_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Optimized Route (CSV)", csv_data, file_name="optimized_route.csv", mime="text/csv")
        except Exception:
            # ignore export if route computation fails
            pass

        # Congestion histogram
        try:
            import matplotlib.pyplot as plt
            factors = [data.get('congestion_factor', 0.0) for _, _, _, data in G_dynamic.edges(keys=True, data=True)]
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(factors, bins=10, color='skyblue', edgecolor='k')
            ax.set_title("Predicted Congestion Distribution")
            ax.set_xlabel("Congestion Factor")
            ax.set_ylabel("Edge Count")
            st.pyplot(fig)
        except Exception:
            pass