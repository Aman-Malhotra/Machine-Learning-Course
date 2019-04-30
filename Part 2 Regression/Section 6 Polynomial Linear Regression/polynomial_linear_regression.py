#Polynomial Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No train and test set beacuse of small  set of data that we have

# We will make both linear and polynomial regression models so that we can compare it later

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#changing the degree to 4 now 
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing linear regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truth OR Bluff (Linear Regression)")
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial Regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("Truth OR Bluff (Polynomial Linear Regression)")
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


# making a new matrix X_grid so that we can make the graph with more curves and not straight lines 
# by taking more number of values b/w 1-10 rather than just taking only the integers we could 
# take values with hight resolution like 0.1, 0.2, ......
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape([len(X_grid), 1])
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth OR Bluff (Polynomial Linear Regression)")
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])


# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

































