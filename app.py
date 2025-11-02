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

# --- Session state defaults for validation ---
if 'start_valid' not in st.session_state:
    st.session_state['start_valid'] = False
if 'end_valid' not in st.session_state:
    st.session_state['end_valid'] = False
if 'start_coords' not in st.session_state:
    st.session_state['start_coords'] = None
if 'end_coords' not in st.session_state:
    st.session_state['end_coords'] = None

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
    """Loads and projects the road network around given center using OSMnx. Cached per center/dist."""
    with st.spinner("Loading geographical data..."):
        G = ox.graph_from_point(center_point, dist=dist, network_type="drive")
        G = ox.project_graph(G)  # Project to metric coordinates
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


def validate_location(place_name: str, G=None, center_point: tuple = None):
    """Validate a human-readable location:

    - Geocodes the place name.
    - Optionally computes distance to center_point and checks against GRAPH_RADIUS.
    - Optionally finds nearest node in G (if provided).

    Returns a dict: {success: bool, coords: (lat,lon)|None, message: str, within: bool|None, node: id|None}
    """
    coords = geocode_location(place_name)
    if coords is None:
        return {'success': False, 'coords': None, 'message': f"Could not geocode '{place_name}'", 'within': None, 'node': None}

    lat, lon = coords
    msg = f"Geocoded to ({lat:.6f}, {lon:.6f})"
    within = None

    if center_point is not None:
        try:
            dist = euclidean_dist(center_point[0], center_point[1], lat, lon)
            within = dist <= GRAPH_RADIUS
            msg += f"; Distance from center: {dist:.0f} m " + ("(within area)" if within else "(outside area)")
        except Exception:
            # ignore distance failures
            pass

    nearest_node = None
    if G is not None:
        try:
            nearest_node = ox.nearest_nodes(G, lon, lat)
            msg += f"; Nearest node id: {nearest_node}"
        except Exception:
            # nearest node failed; ignore
            nearest_node = None

    return {'success': True, 'coords': coords, 'message': msg, 'within': within, 'node': nearest_node}

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

    # Mark edges belonging to the optimized path so we can draw them on top (robust for MultiDiGraph)
    optimized_edges = set()
    if optimized_route:
        for i in range(len(optimized_route) - 1):
            u = optimized_route[i]
            v = optimized_route[i + 1]
            # get_edge_data may return None; handle safely for MultiDiGraph
            edge_dict = G.get_edge_data(u, v) or {}
            if not isinstance(edge_dict, dict):
                continue
            for k in edge_dict.keys():
                optimized_edges.add((u, v, k))

    # Center map on route midpoint if available, otherwise use CENTER_POINT
    if optimized_route:
        mid_idx = len(optimized_route) // 2
        mid_node = G.nodes[optimized_route[mid_idx]]
        center_loc = [mid_node['y'], mid_node['x']]
    else:
        center_loc = CENTER_POINT

    # Create base map after computing route so it's centered correctly
    m = folium.Map(location=center_loc, zoom_start=14, tiles="cartodbpositron", control_scale=True)
    
    # Add distance scale and fullscreen control
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)

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
        # skip edges that are part of the optimized route; we'll draw them thicker and blue later
        if (u, v, k) in optimized_edges:
            continue
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

        # Draw a simple blue route line now; detailed popup and km-markers
        # will be added after metrics are computed to avoid referencing
        # variables that are initialized later (ai_distance, segment_details, etc.).
        folium.PolyLine(route_coords, color='blue', weight=5, opacity=1.0,
                        tooltip=f"{route_type} Route").add_to(m)

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

    # Calculate detailed route metrics and travel time
    try:
        # Get actual route distances
        shortest_distance = nx.shortest_path_length(G, start_node, end_node, weight='length')
        ai_distance = sum(G.edges[u, v, 0]['length'] for u, v in zip(optimized_route[:-1], optimized_route[1:]))
        aerial_distance = euclidean_dist(start_coords[0], start_coords[1], end_coords[0], end_coords[1])

        # Calculate segment-wise details
        segment_details = []
        total_time_mins = 0
        road_type_distances = {}
        cumulative_distance = 0
        
        for i in range(len(optimized_route)-1):
            u, v = optimized_route[i], optimized_route[i+1]
            edge_data = G.edges[u, v, 0]
            distance = edge_data['length']
            road_type = edge_data.get('highway', 'unknown')
            congestion = edge_data.get('congestion_factor', 0.0)
            
            # Estimate speed based on road type and congestion
            base_speed = {
                'motorway': 80, 'trunk': 60, 'primary': 50,
                'secondary': 40, 'tertiary': 30, 'residential': 25
            }.get(road_type, 35)  # km/h
            
            # Adjust speed based on congestion
            if congestion >= 0.8:
                speed = base_speed * 0.3  # Severe congestion
            elif congestion >= 0.6:
                speed = base_speed * 0.5  # Heavy congestion
            elif congestion >= 0.4:
                speed = base_speed * 0.7  # Moderate congestion
            else:
                speed = base_speed * 0.9  # Light/no congestion
            
            # Calculate time for segment
            time_hours = (distance / 1000) / speed
            time_mins = time_hours * 60
            total_time_mins += time_mins
            
            # Track distance by road type
            road_type_distances[road_type] = road_type_distances.get(road_type, 0) + distance
            
            cumulative_distance += distance
            segment_details.append({
                'distance': distance,
                'road_type': road_type,
                'congestion': congestion,
                'time_mins': time_mins,
                'cumulative_distance': cumulative_distance
            })
        
        # After computing segments, add detailed route popup and km markers to the map
        try:
            # Build HTML summary for the route popup
            route_html = f"""
            <div style='font-size:12px;width:220px'>
              <h4>Route Summary</h4>
              <b>Total Distance:</b> {ai_distance/1000:.2f} km<br>
              <b>Est. Time:</b> {int(total_time_mins//60)}h {int(total_time_mins%60)}m<br>
              <b>Avg Speed:</b> {(ai_distance/1000)/(total_time_mins/60):.1f} km/h<br>
              <b>Route Type:</b> {route_type}<br>
              <hr>
              <b>Traffic Conditions:</b><br>
              🔴 Heavy: {sum(1 for s in segment_details if s['congestion'] >= 0.6)}<br>
              🟡 Medium: {sum(1 for s in segment_details if 0.3 <= s['congestion'] < 0.6)}<br>
              🟢 Low: {sum(1 for s in segment_details if s['congestion'] < 0.3)}
            </div>
            """

            # Attach popup to the blue route (add a new polyline with popup)
            folium.PolyLine(
                route_coords,
                color='blue',
                weight=5,
                opacity=1.0,
                popup=folium.Popup(route_html, max_width=300),
                tooltip=f"Click for route details"
            ).add_to(m)

            # Add kilometer markers along the route using cumulative_distance
            total_km = int(ai_distance // 1000)
            if total_km >= 1:
                for km_mark in range(1, total_km + 1):
                    # find the first segment where cumulative_distance >= km_mark*1000
                    marker_latlon = None
                    for seg in segment_details:
                        if seg['cumulative_distance'] >= km_mark * 1000:
                            # approximate marker position using proportional index on route_coords
                            frac = seg['cumulative_distance'] / ai_distance if ai_distance > 0 else 0
                            idx = min(int(len(route_coords) * frac), len(route_coords) - 1)
                            marker_latlon = route_coords[idx]
                            break
                    if marker_latlon:
                        folium.CircleMarker(
                            location=marker_latlon,
                            radius=4,
                            color="black",
                            fill=True,
                            fill_color="white",
                            popup=f"{km_mark} km",
                            tooltip=f"{km_mark} km",
                        ).add_to(m)
        except Exception:
            # If adding popup/markers fails, ignore and continue
            pass
        
        # Show primary metrics in columns
        dist_col1, dist_col2, dist_col3 = st.columns(3)
        with dist_col1:
            st.metric("Total Distance", f"{ai_distance/1000:.2f} km", 
                     delta=f"{((ai_distance-shortest_distance)/shortest_distance)*100:.1f}% vs shortest",
                     help="Actual distance along the selected route")
        with dist_col2:
            st.metric("Estimated Travel Time", 
                     f"{int(total_time_mins//60)}h {int(total_time_mins%60)}min",
                     help="Based on road types and current congestion")
        with dist_col3:
            avg_speed = (ai_distance/1000)/(total_time_mins/60)
            st.metric("Average Speed", f"{avg_speed:.1f} km/h",
                     help="Average travel speed considering traffic")
        
        # Show detailed breakdown with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Route Analysis", "Speed Profile", "Alternative Routes", "Time Estimates"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Distance by Road Type:**")
                for road_type, distance in road_type_distances.items():
                    st.write(f"- {road_type.title()}: {distance/1000:.2f} km ({(distance/ai_distance)*100:.1f}%)")
            
            with col2:
                st.write("**Journey Analysis:**")
                st.write(f"- Direct (aerial) distance: {aerial_distance/1000:.2f} km")
                st.write(f"- Route efficiency: {(aerial_distance/ai_distance)*100:.1f}%")
                st.write(f"- Number of segments: {len(segment_details)}")
                
                # Calculate congestion distribution
                high_cong = sum(1 for s in segment_details if s['congestion'] >= 0.6)
                med_cong = sum(1 for s in segment_details if 0.3 <= s['congestion'] < 0.6)
                low_cong = sum(1 for s in segment_details if s['congestion'] < 0.3)
                st.write(f"- High congestion segments: {high_cong}")
                st.write(f"- Medium congestion segments: {med_cong}")
                st.write(f"- Low congestion segments: {low_cong}")
        
        with tab2:
            # Create speed and congestion profile charts
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Prepare data for the profile charts
            distances = [0]
            speeds = []
            congestions = []
            cum_dist = 0
            
            for seg in segment_details:
                cum_dist += seg['distance']
                distances.append(cum_dist/1000)  # Convert to km
                
                # Calculate speed for this segment
                base_speed = {
                    'motorway': 80, 'trunk': 60, 'primary': 50,
                    'secondary': 40, 'tertiary': 30, 'residential': 25
                }.get(seg['road_type'], 35)
                
                # Apply congestion factor
                if seg['congestion'] >= 0.8:
                    speed = base_speed * 0.3
                elif seg['congestion'] >= 0.6:
                    speed = base_speed * 0.5
                elif seg['congestion'] >= 0.4:
                    speed = base_speed * 0.7
                else:
                    speed = base_speed * 0.9
                    
                speeds.append(speed)
                congestions.append(seg['congestion'])
            
            # Create subplot figure
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Speed Profile', 'Congestion Level'))
            
            # Add speed profile
            fig.add_trace(
                go.Scatter(x=distances, y=speeds, mode='lines', name='Speed (km/h)',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add congestion profile
            fig.add_trace(
                go.Scatter(x=distances, y=congestions, mode='lines', name='Congestion',
                          line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(height=500, title_text="Route Profiles")
            fig.update_xaxes(title_text="Distance (km)")
            fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
            fig.update_yaxes(title_text="Congestion Factor", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Calculate and show alternative routes
            st.write("**Alternative Routes Comparison:**")
            
            # Get three different routes: shortest, fastest, and balanced
            routes = {
                "Current Route": optimized_route,
                "Shortest Distance": nx.shortest_path(G, start_node, end_node, weight='length'),
                "Least Congestion": nx.shortest_path(G, start_node, end_node, 
                                                   weight=lambda u, v, d: d[0].get('length', 1) * (1 + d[0].get('congestion_factor', 0)))
            }
            
            route_metrics = []
            for route_name, route in routes.items():
                distance = sum(G.edges[u, v, 0]['length'] for u, v in zip(route[:-1], route[1:]))
                congestion = np.mean([G.edges[u, v, 0].get('congestion_factor', 0) for u, v in zip(route[:-1], route[1:])])
                time = sum((G.edges[u, v, 0]['length']/1000) / 
                         (40 * (1 - 0.7*G.edges[u, v, 0].get('congestion_factor', 0))) 
                         for u, v in zip(route[:-1], route[1:]))
                
                route_metrics.append({
                    "Route": route_name,
                    "Distance (km)": f"{distance/1000:.2f}",
                    "Avg Congestion": f"{congestion:.2f}",
                    "Est. Time": f"{int(time*60//60)}h {int(time*60%60)}m"
                })
            
            import pandas as pd
            st.table(pd.DataFrame(route_metrics))
        
        with tab4:
            # Show time estimates for different hours
            st.write("**Estimated Travel Times by Hour:**")
            
            hours = list(range(24))
            times = []
            
            for hour in hours:
                # Adjust congestion based on time of day
                if 8 <= hour <= 10 or 17 <= hour <= 19:  # Peak hours
                    factor = 1.5
                elif 0 <= hour <= 5:  # Night hours
                    factor = 0.5
                else:  # Normal hours
                    factor = 1.0
                
                # Calculate estimated time for this hour
                time = total_time_mins * factor
                times.append(time)
            
            # Create time estimation chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=times,
                mode='lines+markers',
                name='Travel Time',
                hovertemplate='Hour: %{x}:00<br>Time: %{text}<extra></extra>',
                text=[f"{int(t//60)}h {int(t%60)}m" for t in times]
            ))
            
            fig.update_layout(
                title="Travel Time by Hour of Day",
                xaxis_title="Hour",
                yaxis_title="Minutes",
                hovermode='x'
            )
            fig.update_xaxes(ticktext=[f"{h:02d}:00" for h in hours], tickvals=hours)
            
            st.plotly_chart(fig, use_container_width=True)
            
        
        
        
        # Add route distance popup to the blue route line
        if optimized_route:
            route_popup = f"""
            <div style='font-size:12px'>
            <b>Route Details:</b><br>
            Distance: {ai_distance/1000:.2f} km<br>
            Type: {route_type}<br>
            </div>
            """
            folium.PolyLine(
                route_coords,
                color='blue',
                weight=5,
                opacity=1.0,
                popup=route_popup,
                tooltip=f"{route_type} Route: {ai_distance/1000:.2f} km"
            ).add_to(m)
            
    except Exception as e:
        st.warning(f"Could not calculate some distance metrics: {str(e)}")

    return m

# --- Main Streamlit Execution ---

st.title("🚦 Predictive Traffic Dashboard: Gottigere to AMC College")
st.markdown("---")

# Removed pre-loading graph. Graph will be loaded after geocoding so it's centered on user route.
st.header("Predicted Traffic Density Map")

# UI: Add inputs for custom route selection (place this in main section before button)
st.subheader("Custom Route Selection")

# Predefined locations along Bannerghatta Road corridor
start_locations = [
    "Gottigere, Kalena Agrahara",
    "Jayadeva Hospital, Bannerghatta Road",
    "Jambu Savari Dinne, Bannerghatta Road",
    "IIM Bangalore",
    "Bilekahalli, Bannerghatta Road",
    "Hulimavu Gate",
    "Meenakshi Temple, Bannerghatta Road",
    "Bannerghatta Circle",
    "Royal Meenakshi Mall",
    "Dairy Circle, Bannerghatta Road",
    "BTM Layout 2nd Stage",
    "JP Nagar 6th Phase",
    "Arekere Gate",
    "DeviKrupa Temple",
    "Apollo Hospital Bannerghatta",
    "Nice Road Junction",
    "Hulimavu Police Station",
    "DLF Apartments"
]

# Educational institutions and landmarks for end points
end_locations = [
    "AMC Engineering College",
    "Reva University",
    "Christ University Bannerghatta Road Campus",
    "New Horizon College of Engineering",
    "IIMB Management Institute",
    "Alliance University",
    "BMS College of Engineering",
    "Oxford College of Engineering",
    "Dayananda Sagar College",
    "Sir M Visvesvaraya Institute of Technology"
]

# Custom location input with dropdown and text input for start location
col_start1, col_start2 = st.columns([1, 1])
with col_start1:
    start_location = st.selectbox("Choose Start Location", start_locations, index=0)
with col_start2:
    start_location = st.text_input("Or Enter Custom Start Location", value="", help="Type a custom location if not in the dropdown list")
    if not start_location:  # If text input is empty, use dropdown value
        start_location = start_locations[0]

# Custom location input with dropdown and text input for end location
col_end1, col_end2 = st.columns([1, 1])
with col_end1:
    end_location = st.selectbox("Choose Destination", end_locations, index=0)
with col_end2:
    end_location = st.text_input("Or Enter Custom Destination", value="", help="Type a custom destination if not in the dropdown list")
    if not end_location:  # If text input is empty, use dropdown value
        end_location = end_locations[0]

# add time-of-day selector and routing strategy selector before the button
selected_hour = st.slider("Select Hour of Day", 0, 23, 8)
route_type = st.radio("Choose Routing Strategy", ["AI-Optimized", "Shortest Path"])

# --- Validation UI ---
st.markdown("### Validate Locations")
col_validate1, col_validate2, col_validate3 = st.columns([1,1,2])
with col_validate1:
    if st.button("Validate Start Location"):
        res = validate_location(start_location, None, CENTER_POINT)
        if res['success']:
            st.success(res['message'])
            st.session_state['start_valid'] = True
            st.session_state['start_coords'] = res['coords']
        else:
            st.error(res['message'])
            st.session_state['start_valid'] = False
with col_validate2:
    if st.button("Validate End Location"):
        res = validate_location(end_location, None, CENTER_POINT)
        if res['success']:
            st.success(res['message'])
            st.session_state['end_valid'] = True
            st.session_state['end_coords'] = res['coords']
        else:
            st.error(res['message'])
            st.session_state['end_valid'] = False
with col_validate3:
    if st.button("Validate Both"):
        res_start = validate_location(start_location, None, CENTER_POINT)
        res_end = validate_location(end_location, None, CENTER_POINT)
        if res_start['success'] and res_end['success']:
            st.success("Both locations validated")
            st.write("Start:", res_start['message'])
            st.write("End:", res_end['message'])
            st.session_state['start_valid'] = True
            st.session_state['end_valid'] = True
            st.session_state['start_coords'] = res_start['coords']
            st.session_state['end_coords'] = res_end['coords']
        else:
            if not res_start['success']:
                st.error("Start: " + res_start['message'])
                st.session_state['start_valid'] = False
            if not res_end['success']:
                st.error("End: " + res_end['message'])
                st.session_state['end_valid'] = False

# Replace button handler: geocode -> compute midpoint -> load graph -> simulate -> visualize
if st.button("Calculate Optimal Route & Density Map", type="primary"):
    # Prefer validated coordinates stored in session_state (if user validated previously)
    start_coords = st.session_state.get('start_coords') or geocode_location(start_location)
    end_coords = st.session_state.get('end_coords') or geocode_location(end_location)
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
            
            # Add nearby landmarks
            landmarks = {
                "Hospitals": ["Jayadeva Hospital", "Apollo Hospital", "Fortis Hospital"],
                "Shopping": ["Royal Meenakshi Mall", "Gopalan Mall", "Bannerghatta Mall"],
                "Education": ["IIM Bangalore", "Christ University", "AMC College"],
                "Transport": ["Metro Station", "Bus Terminal", "Nice Road Junction"]
            }
            
            st.subheader("Nearby Landmarks")
            landmark_cols = st.columns(len(landmarks))
            
            for col, (category, places) in zip(landmark_cols, landmarks.items()):
                with col:
                    st.write(f"**{category}**")
                    for place in places:
                        # Get the coordinates for the landmark (if available)
                        try:
                            coords = geocode_location(f"{place}, Bannerghatta Road, Bangalore")
                            if coords:
                                dist = euclidean_dist(start_coords[0], start_coords[1], coords[0], coords[1])
                                st.write(f"- {place} ({dist/1000:.1f} km)")
                                
                                # Indicate if landmark is within the displayed map area
                                if dist <= GRAPH_RADIUS * 1.5:  # 1.5x the graph radius
                                    st.write("  — within map area")
                        except Exception:
                            st.write(f"- {place} (distance unknown)")
        except Exception:
            pass