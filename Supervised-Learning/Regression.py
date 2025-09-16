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
