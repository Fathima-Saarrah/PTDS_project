# models/baseline_model.py - Baseline model for comparison (BCS302 deliverable)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from.base_model import BaseModel # Using the corrected relative import

class SVR_Model(BaseModel):
    """Simple Support Vector Regression Model for comparative analysis."""
    
    def __init__(self):
        # We use a linear SVR kernel for simplicity
        self.model = SVR(kernel='linear') 

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # SVR expects a 2D input [samples, features]. We flatten the time steps.
        # X_train is (samples, timesteps, 1). We reshape it to (samples, timesteps).
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        # y_train might also need to be flattened if it's 2D
        self.model.fit(X_train_flat, y_train.ravel()) #.ravel() ensures y is 1D

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_flat)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        # You will use these metrics for your BCS302/AIML report comparison
        return rmse, mae