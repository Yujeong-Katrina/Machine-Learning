import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import pandas as pd

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape, y.shape)
#We have 1797 images, wach with 64 pixels of the 8x8 image

def visualization_digit(index): #From this function we can check the image of a digit!
    some_digit = np.array([np.array(X[i]).reshape(8, 8) for i in range(1797)])
    plt.imshow(some_digit[index], cmap='gray')
    plt. title("Digit: " + str(y[index]))
    plt.show()

def get_features(image, grid = 4): #to extract 
    image = image.reshape(8,8)
    pixels_per_grid = int(np.ceil(8/grid)) #2
    features = np.zeros((grid, grid))
    
    for i in range(0, 8):
        for j in range(0, 8):
            features[i//pixels_per_grid][j//pixels_per_grid] += image[i][j]

    return features.reshape(-1)


def plot_some_digit_features(N): #to extract two features from N digit images
    features = np.zeros((N, 16))
    targets = y[:N]
    for i in range(N):
        features[i] = get_features(X[i], 4)

    pca = PCA(n_components = 2) #We use PCA Principal Componant Analysis to reduce the 16-dimensionality to 2-dimensionality .
    #First I didn't use the pca-function, but just 2-grids features from get_features-function. However the features in Plot weren't represented clearly from that way. 
    #That's why I used PCA from sklearn
    features_in_two = pca.fit_transform(features)

    for i in range(10):
        idx = np.where(targets == i) #indices which are i are stored in idx as tuple
        plt.scatter(features_in_two[idx, 0], features_in_two[idx, 1], label=f"Digit {i}") 
    return features_in_two
        
    
def make_classification_for_0_myself(): #I made my own classifier that classify the digit zero
    left = -5
    right = 20
    above = -3
    below = 30
    rect = Rectangle((left,below), right - left, above - below, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    ax = plt.gca()
    ax.add_patch(rect)

    return left, right, above, below



def predict_my_classifier(features, left, right, above, below):
    left, right, above, below = make_classification_for_0_myself()
    if left <= features[0] <= right and below <= features[1] < above:
        prediction = 0
    else: 
        prediction = 1

    return prediction

def test_my_classifier(N):
    features = plot_some_digit_features(N)
    left, right, above, below = make_classification_for_0_myself()
    plt.legend() #label in plot
    plt.show()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(N):
        prediction = predict_my_classifier(features[i], left, right, above, below)
        if y[i] == 0: #actual digit is 0 
            if prediction == 0: #predicted digit is 0
                tp+=1
            else:
                fn+=1
        else:
            if prediction ==0:
                fp+=1
            else:
                tn+=1
    
    accuracy = (tp+tn)/N
    print("Accuracy:", accuracy)
    
N = 1797 #the number of images we will use
test_my_classifier(N)

