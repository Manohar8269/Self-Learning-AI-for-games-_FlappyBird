import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv("flappy_dataset.csv")

X = data[["bird_y", "dist_top", "dist_bottom", "velocity", "dist_to_pipe"]]
y = data["action"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(16, 16), activation='relu', max_iter=5000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.6f}")

plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6)
plt.title("Actual vs Predicted Output")
plt.xlabel("Sample Index")
plt.ylabel("Jump Output")
plt.legend()
plt.tight_layout()
plt.show()

joblib.dump(model, "trained_flappy_model.pkl")
print("Model saved as 'trained_flappy_model.pkl'")
