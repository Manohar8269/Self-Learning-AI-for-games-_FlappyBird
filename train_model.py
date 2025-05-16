import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

data = pd.read_csv("flappy_dataset.csv")

X = data[["bird_y", "dist_top", "dist_bottom", "velocity", "dist_to_pipe"]]
y = data["action"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    max_iter=10000,
    learning_rate_init=0.001,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.6f}")

model_filename = "trained_flappy_model.pkl"
previous_mse = None

if os.path.exists(model_filename):
    old_model = joblib.load(model_filename)
    old_pred = old_model.predict(X_test)
    previous_mse = mean_squared_error(y_test, old_pred)

if previous_mse is None or mse < previous_mse:
    joblib.dump(model, model_filename)
    joblib.dump(scaler, "flappy_scaler.pkl")
    print("✅ New model saved (improved MSE).")
else:
    print("⚠️ MSE did not improve. Model not overwritten.")

# plt.figure(figsize=(10, 5))
# plt.plot(y_test.values, label="Actual", alpha=0.7)
# plt.plot(y_pred, label="Predicted", alpha=0.7)
# plt.title("Actual vs Predicted Actions")
# plt.xlabel("Sample Index")
# plt.ylabel("Jump Probability")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
