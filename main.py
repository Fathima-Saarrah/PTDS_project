# main.py - Orchestrates the entire PTDS system
import numpy as np
import random
from phase1_route import run_phase1
from phase2_visualize import visualize_phase1
from data.data_ingestion import load_and_preprocess
from models.baseline_model import SVR_Model
from models.lstm_model import LSTM_Model
from graph.road_graph import RoadGraph 

# --- A: Initialization and Data ---
DATA_FILE = 'data/traffic_data.csv'
TIME_STEPS = 24 # Use 24 historical hours to predict the next hour

# 1. Download data and place it in the 'data' folder
# 2. Run the full script!

if __name__ == "__main__":
    
    print("--- PHASE 1: Data Preparation ---")
    try:
        # Load and preprocess data (Fixes BCSL305 errors)
        X_train, y_train, X_test, y_test, scaler = load_and_preprocess(DATA_FILE, n_steps_in=TIME_STEPS)
        
        # Reshape data for LSTM [samples, time steps, features]
        # Reshape data for LSTM [samples, time steps, features]
        # CRITICAL FIX: Explicitly use X_train.shape (number of samples/rows)
        X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print("Data loaded and prepared successfully.")
        
        print("Data loaded and prepared successfully.")

    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_FILE}. Please ensure 'traffic_data.csv' is in the 'data' folder.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during Phase 1: {e}. Cannot proceed.")
        exit()


    # --- B: Model Training and Prediction (AIML Core) ---
    print("\n--- PHASE 2: LSTM Training and Prediction ---")
    
    # Train the model 
    lstm_model = LSTM_Model(n_features=1, n_steps=TIME_STEPS)
    # Using 30 epochs for speed in the crash plan
    lstm_model.train(X_train_3d, y_train, epochs=30) 

    # Generate the prediction: traffic density index (0 to 1) for the next hour
    # We use the last sequence in the test set to predict the next single hour
    input_sequence = X_test[-1].reshape(1, TIME_STEPS, 1)
    raw_prediction = lstm_model.predict(input_sequence)
    
    # Extract the single normalized prediction value
    normalized_prediction_value = raw_prediction[0][0]
    
    # Map prediction to junctions. This simulates how the model output is distributed spatially.
    PREDICTED_DENSITY = {
        'J1': float(normalized_prediction_value * 0.4),   # Low congestion factor
        'J2': float(normalized_prediction_value * 1.0),   # Moderate congestion factor
        'J3': float(normalized_prediction_value * 1.5),   # HIGH congestion factor - Model is predicting heavy traffic here
        'J4': float(normalized_prediction_value * 0.7)    # Medium congestion factor
    }

    print(f"Prediction Index (0-1) Generated: {normalized_prediction_value:.4f}")

    # Run Phase 1 route check using the same prediction index
    try:
        print("\n--- PHASE 1 ROUTE CHECK (Local Map) ---")
        run_phase1(prediction_index=float(normalized_prediction_value))
    except Exception as e:
        print(f"Warning: Phase 1 route check failed: {e}")

    # Visualize Phase 1 results (will show NetworkX/Matplotlib plot)
    try:
        print("\n--- PHASE 2 (VISUALIZATION): Drawing traffic map ---")
        visualize_phase1(prediction_index=float(normalized_prediction_value), show_plot=True)
    except ImportError:
        print("Visualization dependencies missing. Install with: pip install networkx matplotlib")
    except Exception as e:
        print(f"Warning: Phase 2 visualization failed: {e}")

    # --- PHASE 4: Model Comparison (BCS302 Metrics) ---
    print("\n--- PHASE 4: Model Comparison (BCS302 Metrics) ---")

    # 1. Run LSTM Evaluation
    # Note: X_test_3d is already ready for LSTM
    y_pred_lstm = lstm_model.predict(X_test_3d) 
    rmse_lstm, mae_lstm = lstm_model.evaluate(y_test, y_pred_lstm)
    print(f"LSTM Results:\nRMSE: {rmse_lstm:.4f}\nMAE: {mae_lstm:.4f}")

    # 2. Run Baseline (SVR) Evaluation
    svr_model = SVR_Model()
    # Train SVR using the same data, but it requires input flattened to 2D
    svr_model.train(X_train_3d, y_train) 
    y_pred_svr = svr_model.predict(X_test_3d)
    rmse_svr, mae_svr = svr_model.evaluate(y_test, y_pred_svr)
    print(f"SVR Baseline Results:\nRMSE: {rmse_svr:.4f}\nMAE: {mae_svr:.4f}")
    
    # Update the phase label for the final integration block:
    # --- E: Final Integration Logic (Now Phase 5) ---
    print("\n--- PHASE 5: Dynamic Graph Optimization ---")
    # --- C: Integration Logic (Maths/DS) ---
    print("\n--- PHASE 3: Dynamic Graph Optimization ---")

    JUNCTIONS = ['J1', 'J2', 'J3', 'J4']
    # Edges: (Source, Destination, Static_Weight) 
    ROAD_SEGMENTS = [
        ('J1', 'J2', 5), ('J1', 'J3', 8), 
        ('J2', 'J4', 10), ('J3', 'J4', 3), # Note: J3->J4 is usually faster (weight 3)
        ('J4', 'J1', 12), ('J3', 'J2', 4)
    ]

    # Instantiate the Graph (BCS304)
    road_network = RoadGraph(JUNCTIONS, ROAD_SEGMENTS)

    # CRITICAL INTEGRATION: Update weights using the AI prediction (BCS301)
    # This is where the AI result drives the optimization.
    road_network.update_weights(PREDICTED_DENSITY)

    # Calculate the shortest path using the dynamically updated weights
    start_node = 'J1'
    end_node = 'J4'
    path = road_network.find_shortest_path(start_node, end_node, weight_attribute='dynamic_weight')

    print(f"\n--- SUCCESS: Core System Functional ---")
    print(f"If J3 is congested by the AI prediction, the path avoids the J3->J4 segment if possible.")


