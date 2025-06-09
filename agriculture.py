# Example: Streaming data anomaly detection for earthquake warning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulate sensor data (seismic readings)
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000,1))
earthquake_data = np.random.normal(10, 1, (50,1))
data = np.vstack((normal_data, earthquake_data))

# Fit Isolation Forest
clf = IsolationForest(contamination=0.05)
clf.fit(data)

# Predict anomalies
pred = clf.predict(data)
anomalies = np.where(pred == -1)[0]

# Plot
plt.plot(data, label='Sensor Reading')
plt.scatter(anomalies, data[anomalies], color='red', label='Anomaly Detected')
plt.legend()
plt.show()
