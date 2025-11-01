# models/lstm_model.py - The Core Prediction Engine
# models/lstm_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import numpy as np
from.base_model import BaseModel # Using the correct relative import

class LSTM_Model(BaseModel): # This class structure satisfies the OOP (BCS306) requirement
    """
    Core Prediction Layer using LSTM for traffic time-series forecasting.
    LSTM excels at handling the non-linear time dynamics of traffic flow,  
    making it superior to simple linear models. [2]
    """
    def __init__(self, n_features: int, n_steps: int):
        self.n_features = n_features
        self.n_steps = n_steps
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Builds a simple stacked LSTM network."""
        model = Sequential()
        # Input shape: [time steps, features] (e.g., 24 hours, 1 feature)
        model.add(LSTM(50, activation='relu', input_shape=(self.n_steps, self.n_features))) 
        model.add(Dropout(0.2))
        model.add(Dense(1)) # Output is a single predicted density value
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs=50):
        """Train the model using pre-processed sequential data."""
        # LSTM input requires 3D shape: [samples, timesteps, features]
        # CORRECTED Reshape Syntax:
        X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
        print("Starting LSTM Training...")
        #... rest of the train method
        #... rest of the train method
        # Use minimal epochs for a fast run
        self.model.fit(X_train_3d, y_train, epochs=epochs, verbose=1) 
        print("Training complete.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions on the test set."""
        # CORRECTED Reshape Syntax:
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], self.n_features)
        return self.model.predict(X_test_3d, verbose=0)

    # Add this method to satisfy the BaseModel abstract class:
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates mandatory evaluation metrics (RMSE, MAE) for the LSTM.
        This satisfies the abstract method requirement.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae