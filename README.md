# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Initialize dataset, weights, and bias.


2.Define the sigmoid function.


3.Compute predicted output using current weights.


4.Update weights and bias using gradient descent.


5.Repeat until convergence and display the result.



## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIDHYA SHREE K
RegisterNumber:  212225230296

import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1])

# Initialize parameters
w = 0
b = 0
learning_rate = 0.1
epochs = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(epochs):
    z = w * X + b
    y_pred = sigmoid(z)
    
    # Derivatives
    dw = (1/len(X)) * np.sum((y_pred - y) * X)
    db = (1/len(X)) * np.sum(y_pred - y)
    
    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Output
print("Weight:", w)
print("Bias:", b)

# Plot
plt.scatter(X, y, color='red')
x_line = np.linspace(0, 6, 100)
y_line = sigmoid(w * x_line + b)
plt.plot(x_line, y_line)
plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/75ec2da1-a4af-448e-a3ea-88307c951abe" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

