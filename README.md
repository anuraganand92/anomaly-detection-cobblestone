# Anomaly Detection Code Explanation

## Overview

This Python code implements an **anomaly detection system** that monitors a data stream in real-time and identifies any unusual patterns, such as anomalies and change points. The system is built using a combination of **statistical techniques** (e.g., exponentially weighted moving average) and **machine learning** (e.g., Isolation Forest), along with **change-point detection** via the `ruptures` library.

The key components include:
- **Data Stream Simulation**: The synthetic data stream has noise, trends, seasonality, anomalies, and sudden shifts to mimic real-world behavior.
- **Anomaly Detection**: Using a combination of statistical anomaly detection (Z-scores) and machine learning (Isolation Forest).
- **Change-Point Detection**: Identifies shifts in the underlying behavior of the data.
- **Visualization**: Results are plotted to visualize the data stream, anomalies, and detected change points.

---

## Core Logic Explanation

### 1. `AnomalyDetector` Class

This class is designed to detect anomalies in a data stream in real-time. It also has functionality to detect sudden change points.

#### 1.1 `__init__(self, window_size=100, contamination=0.05, n_estimators=100)`

- **Parameters**:
  - `window_size`: The size of the sliding window that holds the most recent data points. This helps in analyzing local trends.
  - `contamination`: The expected fraction of anomalies in the data, used by the Isolation Forest. A higher value makes the model more sensitive to outliers.
  - `n_estimators`: Number of trees in the Isolation Forest ensemble model. More trees typically increase accuracy at the cost of computational resources.

This method initializes key components:
- **Sliding Window** (`self.window`): A `deque` structure to store recent data points.
- **Isolation Forest** (`self.iforest`): A machine learning model that detects anomalies by measuring how easily a point can be isolated in a tree-based structure.
- **Exponentially Weighted Mean/Standard Deviation** (`self.ewm_mean`, `self.ewm_std`): These statistics are used to measure deviations from a local trend. They are updated incrementally with each new data point.
- **Change Point Model** (`self.change_point_model`): Uses the PELT algorithm with an RBF kernel to detect points where the data distribution changes abruptly.

#### 1.2 `update_ewm(self, value)`

This function **updates the exponentially weighted moving average (EWMA)** and **standard deviation** for the incoming value. This approach is useful for smoothing out noise and reacting to local trends in the data stream.

- **EWMA Formula**:
  - EWMA update: `ewm_mean = alpha * new_value + (1 - alpha) * old_mean`
  - EWM standard deviation (`ewm_std`) is similarly updated but with more complexity to account for deviations.
  - `alpha` is the smoothing factor (default: 0.1), where smaller values make the average less reactive to recent data.

#### 1.3 `detect(self, value)`

This method processes the incoming `value` and checks whether it is anomalous using two approaches:
1. **Z-Score Calculation**:
   - **Z-score** measures how far the value deviates from the exponentially weighted mean in terms of standard deviations. 
   - Formula: `Z = (value - ewm_mean) / ewm_std`
   - If the Z-score is high, it indicates that the value is unusual compared to the local trend.

2. **Isolation Forest**:
   - The Isolation Forest model is trained on the current sliding window of data points (`self.window`).
   - It assigns an **anomaly score** to each point, measuring how easily it can be isolated from the others in the data.

3. **Combined Scoring**:
   - The Z-score and Isolation Forest scores are combined into a **final anomaly score**.
   - A **threshold** (default: 2) is used to determine if the combined score indicates an anomaly. If the score is greater than the threshold, the value is flagged as an anomaly.

#### 1.4 `detect_change_points(self, data_stream)`

This function detects **change points** in the data using the **PELT (Pruned Exact Linear Time)** algorithm from the `ruptures` library.

- **Change Point Detection**:
  - A **change point** occurs when there is a sudden shift in the data's statistical properties (e.g., mean or variance).
  - The PELT algorithm minimizes the cost function of modeling the data as segments with different statistical properties. The `pen=10` parameter adjusts the sensitivity of the detection.
  
---

### 2. `generate_data_stream(n_samples=1000, anomaly_ratio=0.05)`

This function generates a synthetic data stream with the following characteristics:
- **Noise**: Gaussian noise is added to mimic random fluctuations in real-world data.
- **Trend**: A linear upward trend is introduced over the entire data stream.
- **Seasonality**: A sinusoidal wave is added to simulate periodic behaviors.
- **Anomalies**: Random anomalies (extreme values) are injected into the data at random positions.
- **Sudden Shift**: At the halfway point of the data, a sudden jump (shift) in the mean is introduced to simulate a regime change.

This data stream mimics real-world metrics that experience variations due to noise, periodic behavior, and rare but significant anomalies.

---

### 3. `main()`

This function ties everything together:
- Generates the synthetic data stream.
- Initializes the `AnomalyDetector` object with a sliding window and anomaly detection model.
- Iterates over the data stream, detecting anomalies in real-time.
- Detects change points in the complete data stream after all values have been processed.
- Visualizes the results by plotting the data stream, detected anomalies, and change points.

---

## Mathematical Concepts

1. **Exponentially Weighted Moving Average (EWMA)**:
   - EWMA is used to smooth the time series by giving more weight to recent values and progressively less weight to older values.
   - It helps detect deviations from recent trends and reduces the influence of historical values.

2. **Z-Score**:
   - The Z-score standardizes the data and quantifies how many standard deviations a value is from the mean.
   - High Z-scores indicate values that are significantly different from the local trend, making them potential anomalies.

3. **Isolation Forest**:
   - Isolation Forest is a machine learning algorithm that isolates anomalies by randomly partitioning the data.
   - It is based on the idea that anomalies are easier to isolate than normal points.
   - The model constructs multiple trees and assigns an anomaly score based on how many splits are needed to isolate a point.

4. **PELT Algorithm**:
   - PELT is a fast, exact algorithm used for detecting change points in time series.
   - It optimizes the cost function, balancing between the goodness of fit within segments and the penalty for adding change points.

---

## Parameters

1. **window_size**: Defines how much history is considered when calculating trends and feeding the Isolation Forest model.
2. **contamination**: The fraction of data that is expected to be anomalous, used in the Isolation Forest to adjust its sensitivity.
3. **n_estimators**: Number of trees in the Isolation Forest, controlling its robustness.
4. **alpha**: Smoothing factor for the EWMA, controlling how quickly the average adapts to recent data.
5. **threshold**: A combined anomaly score above this threshold is flagged as anomalous.

---

## Conclusion

This anomaly detection system combines statistical methods and machine learning for real-time detection of anomalies in data streams. It is designed to adapt to trends, handle noise, and detect sudden changes or shifts in data behavior. The synthetic data stream generation helps to simulate various patterns commonly seen in real-world data, making the system robust and generalizable.
