import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_and_preprocess(file_path: str, n_steps_in: int = 24) -> Tuple:
    """Loads traffic data, performs feature engineering, and prepares sequences for LSTM."""
    
    # CRITICAL FIX: Explicitly parse the 'DateTime' column while reading the CSV.
    # This list format is the correct way to specify the column and fix the syntax error.
    df = pd.read_csv(file_path, parse_dates=['DateTime'], dayfirst=True)
    
    # **BCSL305 Requirement: Data Structures Lab (Sorting)**
    # Sorting must happen after parsing to ensure chronological order.
    df = df.sort_values(by='DateTime')

    # Focus on the 'Vehicles' column for time series prediction
    data = df['Vehicles'].values.reshape(-1, 1)
    
    #... rest of the code remains the same...

    # Normalization (Scaling data between 0 and 1, required for stable Deep Learning training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create LSTM Sequences (e.g., 24 historical hours to predict the 25th)
    X, y = list(), list()
    for i in range(len(scaled_data)):
        end_ix = i + n_steps_in
        if end_ix > len(scaled_data) - 1:
            break
        
        # Seq_x is the 24-hour input sequence; Seq_y is the target (the 25th hour)
        seq_x, seq_y = scaled_data[i:end_ix, 0], scaled_data[end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)

    # Split data for training/testing
    X_train = np.array(X[:-1000])
    y_train = np.array(y[:-1000])
    X_test = np.array(X[-1000:])
    y_test = np.array(y[-1000:])

    return X_train, y_train, X_test, y_test, scaler