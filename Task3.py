# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File path
path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\ELAB\\Task-3\\housing.csv"

# Loading dataset
df = pd.read_csv(path)
df.columns = df.columns.str.strip()  # Clean any extra spaces

# Show real column names
print("📝 Columns available in dataset:")
print(df.columns.tolist())

# Show top 5 rows
print("\n🔍 Sample data:")
print(df.head())

# Drop missing values
df = df.dropna()

# -------------------------
# Simple Linear Regression
# -------------------------
# Use 'median_income' if present (else tell me column list)
if 'median_income' not in df.columns or 'median_house_value' not in df.columns:
    raise ValueError("Expected columns not found. Please share column names shown above.")

X_simple = df[['median_income']]
y_simple = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=0)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)

# Coefficients and intercept
print("\n📊 SIMPLE LINEAR REGRESSION:")
print("Coefficient (slope):", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R² Score:", r2_score(y_test, y_pred_simple))

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred_simple, color='red', label='Predicted')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig("C:\\Users\\KIIT\\OneDrive\\Desktop\\ELAB\\Task-3\\simple_regression_plot.png")
plt.show()

# -------------------------
# Multiple Linear Regression
# -------------------------
features = ['median_income', 'total_rooms', 'housing_median_age']
for f in features:
    if f not in df.columns:
        raise ValueError(f"Column '{f}' is missing — please check dataset.")

X_multi = df[features]
y_multi = df['median_house_value']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=0)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_multi = model_multi.predict(X_test_m)

# Coefficients
print("\n📊 MULTIPLE LINEAR REGRESSION:")
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)

# Evaluation
print("MAE:", mean_absolute_error(y_test_m, y_pred_multi))
print("MSE:", mean_squared_error(y_test_m, y_pred_multi))
print("R² Score:", r2_score(y_test_m, y_pred_multi))

# Saving predictions to CSV
df_result = pd.DataFrame({
    'Actual': y_test_m,
    'Predicted': y_pred_multi
})
output_path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\ELAB\\Task-3\\multiple_regression_predictions.csv"
df_result.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to: {output_path}")

# Print few predicted values
print("\n📌 Sample Predictions (Multiple Regression):")
print(df_result.head())
