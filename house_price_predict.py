import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

hdf=pd.read_csv(r"C:\Users\dines\OneDrive\Documents\Datascience\HousingData.csv", sep=',')
hdf.head(5)
hdf.dropna(inplace=True)  # Remove rows with missing values

features=['size','uds']
x=hdf[features]
y=hdf['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)
print(x_train.shape)
print(x_test.shape)

model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))

# Scatter plot of actual vs predicted values
sns.scatterplot(x=y_test, y=pred, color='blue', alpha=0.6)

# Plot a reference line (y = x) to show perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

residuals = y_test - pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')  # Reference line at zero
plt.xlabel("Actual Prices")
plt.ylabel("Residuals (Error)")
plt.title("Residual Plot")
plt.show()

