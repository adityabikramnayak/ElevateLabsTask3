# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------
# Create Save Directory
# --------------------------
save_dir = "C:\\Users\\KIIT\\OneDrive\\Desktop\\ELAB\\Task-3"
os.makedirs(save_dir, exist_ok=True)

# File path
path = os.path.join(save_dir, "housing.csv")

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv(path)
df.columns = df.columns.str.strip()  # Clean column names

print("📝 Available Columns:")
print(df.columns.tolist())
print("\n🔍 Sample Data:")
print(df.head())

df = df.dropna()

# --------------------------
# SIMPLE LINEAR REGRESSION
# --------------------------
if 'area' not in df.columns or 'price' not in df.columns:
    raise ValueError("Expected columns like 'area' and 'price' not found. Please check dataset.")

X_simple = df[['area']]
y_simple = df['price']

X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=0)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)

print("\n📊 SIMPLE LINEAR REGRESSION:")
print("Coefficient (slope):", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R² Score:", r2_score(y_test, y_pred_simple))

# Save Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_simple, color='red', label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "simple_regression_plot.png"))
plt.show()

# --------------------------
# MULTIPLE LINEAR REGRESSION
# --------------------------
# Using only numerical features for simplicity
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for col in features:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

X_multi = df[features]
y_multi = df['price']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=0)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_multi = model_multi.predict(X_test_m)

print("\n📊 MULTIPLE LINEAR REGRESSION:")
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)
print("MAE:", mean_absolute_error(y_test_m, y_pred_multi))
print("MSE:", mean_squared_error(y_test_m, y_pred_multi))
print("R² Score:", r2_score(y_test_m, y_pred_multi))

# Save CSV
df_result = pd.DataFrame({
    'Actual': y_test_m,
    'Predicted': y_pred_multi
})
csv_path = os.path.join(save_dir, "multiple_regression_predictions.csv")
df_result.to_csv(csv_path, index=False)
print(f"\n✅ Predictions saved to: {csv_path}")

# Sample Predictions
print("\n📌 Sample Predictions:")
print(df_result.head())

# Optional: Open folder automatically
os.startfile(save_dir)
