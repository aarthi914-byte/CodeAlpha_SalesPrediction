import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the dataset
df = pd.read_csv("advertising.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 2. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Separate features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared (R2):", round(r2, 4))

# 8. Display model coefficients
print("\n--- Advertising Impact ---")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, "coefficient:", round(coefficient, 4))

print("\nIntercept:", round(model.intercept_, 4))

# 9. Predict sales for a sample advertising budget
sample = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

sample_prediction = model.predict(sample)

print("\nPredicted sales for TV=150, Radio=30, Newspaper=20:",
      round(sample_prediction[0], 2))

# 10. Actual vs Predicted Sales graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.savefig("actual_vs_predicted_sales.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Advertising impact graph
plt.figure(figsize=(8, 5))
plt.bar(X.columns, model.coef_)
plt.xlabel("Advertising Platform")
plt.ylabel("Coefficient")
plt.title("Advertising Impact on Sales")
plt.grid(axis="y")
plt.savefig("advertising_impact.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nTask 4 completed successfully!")