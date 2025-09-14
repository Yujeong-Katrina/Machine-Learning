import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from time import time
from copy import deepcopy

W = np.random.randn(100, 100)
x = np.random.randn(1000, 100)
time0 = time()
z_sequential = np.zeros((1000,100))

for i in range(1000):
    z = np.dot(W, x[i]) #100dim

time1 = time()
z_parallel = np.matmul(x, W.T) #vectorized calculation
time2 = time()

print(f"Time for sequential matrix multiplication: {time1 - time0}")
print(f"Time for parallel matrix multiplication: {time2 - time1}")
#Sedond one is faster!

#Data Generation
n_datapoints = 300
np.random.seed(0) # fixed initial values
X = np.random.rand(n_datapoints, 1) #0~1 300values

def f(input):
    return np.sin(2 * np.pi * input[: , 0]) + np.cos(4 * np.pi * input[: , 0])
#input의 첫번째 열 * 파이 *2 -:> 주기가 1인 사인, ... -:>주기가 0.5인 코사인
Y = f(X)
Y = Y[:, None]

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.plot(X[:, 0], Y[:, 0], ".") #x-axis and y-axis with dot graph, in other words mixing graph
ax.set_title("Full toy dataset")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()

print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)

#Split data into train and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=1) #total 300 datasets: 1: 0.1 ratio, fixed seed(random_state = 1)
print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)

activate_function_map = {
    "iden" : (lambda x: x),
    "relu" : (lambda x: np.maximum(0.0, x)),
    "tanh" : (lambda x: np.tanh(x))
}

activate_function_derivative_map = {
    "iden" : (lambda x: 1),
    "relu" : (lambda x: np.sign(np.maximum(0.0, x))),
    "tanh" : (lambda x: (1 - np.tanh(x)**2))
}

class Layer:
    def __init__(self, num_output: int, dim_input: int, act_fun_name: str):
        self.num_output = num_output
        self.act_fun = activate_function_map[act_fun_name]
        self.act_fun_deriv = activate_function_derivative_map[act_fun_name]

        if act_fun_name in ["iden", "tanh"]:
            weight_variance = 2.0 / (num_output + dim_input) #Xavier initialization
        elif act_fun_name == "relu":
            weight_variance = 2.0 / dim_input #He initialization

        self.W = np.random.randn(num_output, dim_input) * np.sqrt(weight_variance) #the weight that connects a layer to another one
        self.b = np.zeros(num_output) #bias
        self.x = None #input data
        self.z = None #준output data
        self.a = None #ouput data, after activation function

        self.gradient_W = None
        self.gradient_b = None

class MuliLayerPerceptron:
    def __init__(self, input_dim: int, layer_parameter_list: List[Tuple], seed: int = 2):
        self.layer_list = []
        np.random.seed(seed)
        dim_input = input_dim
        for(num_output, act_fun_name) in layer_parameter_list:
            layer = Layer(num_output, dim_input, act_fun_name)
            self.layer_list.append(layer)
            dim_input = num_output

    def forward(self, x: np.ndarray) -> np.ndarray: #forward pass
        out = x
        for layer in self.layer_list:
            layer.x = out
            layer.z = layer.W @ out.T + layer.b[:, np.newaxis]
            layer.z = layer.z.T
            layer.a = layer.act_fun(layer.z)
            out = layer.a
        return out
    
    def backward(self, delta: np.ndarray): #backward pass
        for layer in reversed(self.layer_list):
            delta = delta * layer.act_fun_deriv(layer.z)
            layer.grad_W = delta.T @ layer.x
            layer.grad_b = np.sum(delta, axis=0)
            delta = delta @ layer.W

    def train_with_square_loss(self, X : np.ndarray, y : np.ndarray, eta = 0.001): 
        prediction = self.forward(X)
        delta = (prediction - y) / X.shape[0]
        self.backward(delta)

        for layer in self.layer_list:
            layer.W -= eta * layer.grad_W
            layer.b -= eta * layer.grad_b
        
        return np.mean((prediction - y)**2)
    
    def error(self, X, y):
        prediction = self.forward(X)
        return 0.5 * np.mean((prediction - y)**2)
    
#Set up an MLP with 2 hidden layer . And We use ReLU activation function

net = MuliLayerPerceptron(
    input_dim=X_train.shape[1],
    layer_parameter_list=[(32, "tanh"), (32, "relu"), (32, "relu"), (1, "iden")]
)

train_error_list = []
valid_error_list = []
batch_checkpoints = []

n_steps = 20000
batch_times = []
time_start = time()
for i in range(n_steps): # Using Back propagation, training error and validation error will be decreased over time
    train_error = net.train_with_square_loss(X_train, Y_train, 0.02)
    valid_error = net.error(X_valid, Y_valid)
    train_error_list.append(train_error)
    valid_error_list.append(valid_error)
    batch_times.append(time() - time_start)

    if i % 1000 == 0:
        print(f"Training step {i}; training error: {train_error}; validation error: {valid_error}")
        batch_checkpoints.append(deepcopy(net))

plt.plot(valid_error_list, linewidth = 2, label="validation")
plt.plot(train_error_list, label = "train")
plt.yscale("log")
plt.xlabel("steps")
plt.ylabel("error")
plt.legend()
plt.show()