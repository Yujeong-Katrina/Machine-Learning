import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


f, axes = plt.subplots(1,3)
axes[0].scatter(X_train[:50, 0], y_train[:50])
axes[1].scatter(X_train[:50, 1], y_train[:50])
axes[2].scatter(X_train[:50, 2], y_train[:50]) #I will use only 100 regions and three features
axes[0].set_xlabel('Median Income')
axes[1].set_xlabel('House Age')
axes[2].set_xlabel('Average Rooms per Household')
axes[0].set_ylabel('Price')
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
for ax in axes:
    ax.grid()
f.tight_layout()
f.set_size_inches(9, 3)
plt.show()

def offset_data(X: np.ndarray) -> np.ndarray: #Add a column with ones to the begining og X
    #In Korean, it enables y축의 절댓값: w0
    ones_column = np.ones((X.shape[0], 1))
    X_offset = np.concatenate((ones_column, X), axis = 1)
    return X_offset

def regressor(X: np.ndarray, y: np.ndarray) -> np.ndarray: #w is weights that refers the minimum of error function
    X_offset = offset_data(X)
    weights = np.linalg.solve(X_offset.T @ X_offset, X_offset.T @y) #We can get such expression by setting the derivative of the error function to zero
    return weights

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray: #This function makes predictions
    X_offset = offset_data(X)
    predictions = X_offset @ w
    return predictions

weights = regressor(X_train, y_train)
train_predictions = predict(X_train, weights)
test_predictions = predict(X_test, weights)
print("Predicted prices for new houses: ")
num_house = 0 #Since there are so many test case, we only check num_house test cases.
for i, (features, true_price, test_preds) in enumerate(zip(X_test, y_test, test_predictions)):
    num_house += 1
    print(f'House {i} with {features[0]} Median Income, {features[1]} House ages and {features[2]} Average rooms. Sold for {true_price* 100000: 6.0f}$. Predicted Price: {test_preds * 100000: 6.0f}$')
    if num_house==4: break

def compute_r2_score(preds: np.ndarray, y: np.ndarray) -> np.ndarray: #This function enabels to access how the predicted model fits to the data
    Residual_sum_of_squares = np.sum((y - preds) ** 2)
    Averaged_y = np.mean(y)
    TSS = np.sum((y - Averaged_y)**2)
    R2_score = 1 - (Residual_sum_of_squares/TSS)
    return R2_score #If this score is zero, the model does not better than using the mean to do predictions

r2_score = compute_r2_score(train_predictions, y_train)
print(f"The computes multivariate linear regression has a R^2 value of {r2_score:0.3f}!")



#Now I will visualize the linear model.only using the first feature(;Median Income)
x_simple = X_train[:50, 0].reshape(-1, 1)
y_simple = y_train[:50]
weights = regressor(x_simple, y_simple)
predictions = predict(x_simple, weights)
f, ax = plt.subplots()
ax.scatter(x_simple, y_simple)
ax.plot(x_simple, predictions, label = "Linear Fit")
ax.set_xlabel("Median Income")
ax.set_ylabel("Price")
ax.legend()
plt.show()