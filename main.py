import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import deque
from sklearn.ensemble import IsolationForest
import ruptures as rpt

class AnomalyDetector:
    def __init__(self, window_size=100, contamination=0.05, n_estimators=100):
        """
        Initializes the anomaly detector with key parameters:
        - `window_size`: The number of most recent data points to consider.
        - `contamination`: The proportion of outliers in the data used by Isolation Forest.
        - `n_estimators`: Number of trees in the Isolation Forest.
        """
        self.window_size = window_size
        self.contamination = contamination
        self.window = deque(maxlen=window_size)  # Stores the latest data points in a sliding window.
        self.iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.ewm_mean = None  # Exponentially weighted moving mean.
        self.ewm_std = None   # Exponentially weighted moving standard deviation.
        self.alpha = 0.1      # Smoothing factor for the EWM calculation.
        self.change_point_model = rpt.Pelt(model="rbf").fit  # For detecting sudden changes in trends using RBF kernel.

    def update_ewm(self, value):
        """
        Updates exponentially weighted moving statistics for mean and standard deviation.
        This helps in calculating anomalies based on trends and local deviations.
        """
        if self.ewm_mean is None:
            self.ewm_mean = value
            self.ewm_std = 0
        else:
            diff = value - self.ewm_mean
            incr = self.alpha * diff
            self.ewm_mean += incr
            self.ewm_std = (1 - self.alpha) * (self.ewm_std**2 + self.alpha * diff**2)**0.5

    def detect(self, value):
        """
        Detects anomalies using a combination of EWM Z-score and Isolation Forest.
        - Updates the window with the current value.
        - Checks if the data in the sliding window indicates an anomaly.
        """
        self.window.append(value)
        self.update_ewm(value)
        
        if len(self.window) < self.window_size:
            return False
        
        # Calculate Z-score using the exponentially weighted mean and standard deviation.
        if self.ewm_std > 0:
            z_score = (value - self.ewm_mean) / self.ewm_std
        else:
            z_score = 0

        # Fit the Isolation Forest on the current sliding window and compute anomaly scores.
        X = np.array(list(self.window)).reshape(-1, 1)
        self.iforest.fit(X)
        anomaly_scores = self.iforest.decision_function(X)
        is_anomaly = self.iforest.predict(X[-1].reshape(1, -1)) == -1

        # Combine the Z-score and Isolation Forest results to produce a final anomaly score.
        combined_score = (abs(z_score) + (1 - anomaly_scores[-1])) / 2
        threshold = 2  # Adjust this threshold for different sensitivity levels.

        return combined_score > threshold

    def detect_change_points(self, data_stream):
        """
        Detects change points in a given data stream using the PELT algorithm.
        A change point indicates a sudden shift in the data's behavior.
        """
        change_points = self.change_point_model(data_stream).predict(pen=10)
        return change_points

def generate_data_stream(n_samples=1000, anomaly_ratio=0.05):
    """
    Generates a synthetic data stream for testing the anomaly detection system.
    The data stream includes:
    - Noise: Normally distributed random noise.
    - Trend: A linear increasing trend.
    - Seasonality: A sinusoidal wave simulating periodic behavior.
    - Anomalies: Random extreme values added to simulate outliers.
    - Shift: A sudden change in the data's mean value.
    """
    data = np.random.randn(n_samples) * 0.5  # Random noise
    
    # Add linear trend
    data += np.linspace(0, 2, n_samples)
    
    # Add sinusoidal seasonality
    data += np.sin(np.linspace(0, 8*np.pi, n_samples))
    
    # Inject random anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * anomaly_ratio), replace=False)
    data[anomaly_indices] += np.random.uniform(3, 5, size=len(anomaly_indices)) * np.random.choice([-1, 1], size=len(anomaly_indices))
    
    # Introduce a sudden shift in data at the midpoint
    shift_point = n_samples // 2
    data[shift_point:] += 2
    
    return data

def main():
    """
    Main function that orchestrates the anomaly detection and visualization process:
    """
    data_stream = generate_data_stream()
    detector = AnomalyDetector(window_size=50, contamination=0.05)
    
    anomalies = []
    for i, value in enumerate(data_stream):
        if detector.detect(value):
            anomalies.append((i, value))

    # Detect change points
    change_points = detector.detect_change_points(data_stream)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label='Data Stream', color='blue', alpha=0.7)
    
    # Plot detected anomalies
    if anomalies:
        anomaly_indices, anomaly_values = zip(*anomalies)
        plt.scatter(anomaly_indices, anomaly_values, color='red', label='Anomalies', zorder=5)
    
    plt.title('Data Stream with Anomalies and Change Points Detected')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
