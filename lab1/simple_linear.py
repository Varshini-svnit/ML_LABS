# simple_linear.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("advertising.csv")

# Use 'TV' to predict 'Sales'
X = data[['TV']]
y = data['Sales']

# Visualize data
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Budget (in $1000s)")
plt.ylabel("Sales (in $1000s)")
plt.grid()
plt.show()

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluation
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))

# Plot regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title("Regression Line: TV vs Sales")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.grid()
plt.show()
