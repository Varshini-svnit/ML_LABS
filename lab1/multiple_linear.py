# multiple_linear.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Housing.csv")

# Display first few rows
print("Dataset Sample:\n", data.head())

# Choose relevant columns
features = ['area', 'bedrooms', 'bathrooms', 'parking']
target = 'price'

X = data[features]
y = data[target]

# Check correlations
sns.heatmap(data[features + [target]].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict example
example = pd.DataFrame({
    'area': [2500],
    'bedrooms': [4],
    'bathrooms': [3],
    'parking': [2]
})
predicted_price = model.predict(example)[0]
print("Predicted Price for example house:", predicted_price)
