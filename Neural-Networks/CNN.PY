import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import random

#I use Fashion MNIST as dataset-->Clothes
use_mps = False
def set_seed_(seed):
    torch.manual_seed(seed) #Fix seed about CPU of PyTorch
    np.random.seed(seed) #Fix seed for NumPy random seed
    random.seed(seed) 
    if use_mps: #For Mac MPS
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    else: #For CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) #Seed for all GPU device

seed = 42 #It means same random value per all run
set_seed_(seed)


#And ChatGPT generated this snippet of code and its corresponding comments for me.

 # Transform to normalize the data
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])
# Download and load the training data
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# Split the training set into training and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_subset, val_subset = random_split(trainset, [train_size, val_size])
# Create data loaders for the training, validation, and test sets
trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
valloader = DataLoader(val_subset, batch_size=64, shuffle=False)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
#Finish to use chat GPT

label_map = {0 : "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

#Construct the neural networks
def create_convolution_layers():
    layers = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
        #The data set is grayscale image, so input channel is 1 and we will make it 32 channels. Channel means feature(here may be the number of colors!
        nn.ReLU(), #Non-linear: Negative value to zero
        nn.MaxPool2d(kernel_size = 2), #It decreases the size of the hidden representation -> to 14*14
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2) #Again reduce the pixel to 7*7, and features: 64
    )

    return layers

def create_fc_layers():
    layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7* 7, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    return layers

class FashionMNIST_NN(nn.Module):
    def __init__(self, conv_layers, fc_layers, hook_func=None):
        super(FashionMNIST_NN, self).__init__()
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers

        if hook_func is not None:
            self.outputs = {}
            self.hooks = []
            self.hook_func = hook_func
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = torch.softmax(x, dim = 1)
        return x
    
    def extract_features(self):
        if hasattr(self, "outputs"):
            return self.outputs
        else:
            return None
    
    def _remove_hooks(self):
        if hasattr(self, "hooks"):
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        else:
            pass

    def reset_features(self):
        self.outputs = {}

class ModelTrainer:
    def __init__(self, model, trainloader, testloader, train_params, valloader = None, hook_func = None):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = train_params["lr"]

        if train_params["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        elif train_params["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate)
        else:
            raise NotImplementedError()
        
        self.use_gpu = train_params["use_gpu"]

        if use_mps:
            self.device = torch.device("mps" if (torch.backends.mps.is_available() and self.use_gpu) else "cpu")
        else:
            self.device = torch.device("cuda:0" if(torch.cuda.is_available() and self.use_gpu) else "cpu")

        print("My device is:", self.device)
        self.model.to(self.device)
        self.seed = train_params["seed"]

    def set_seed(self):
        set_seed_(seed)

    def train(self, epochs=10):
        n_epochs = epochs
        train_losses, val_losses, others = [],[],[]
        for epochs in range(n_epochs):
            self.model.train()
            if hasattr(self.model, "hook_func"):
                self.model.hook_func(conv_layers = self.model.conv_layers, outputs=self.model.outputs, hooks = self.model.hooks)
            running_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
            train_losses.append(running_loss / len(trainloader))
            self.model._remove_hooks()

            if self.valloader is not None:
                val_loss, other = self.evaluate()
                val_losses.append(val_loss)
                others.append(other)
                
            print(f'Epoch {epochs+1}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

        return train_losses, val_losses, others
        
    def predict(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')
        return accuracy

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                val_loss += loss.item()

        return val_loss / len(self.valloader), None
    
    def get_features(self):
        return self.model.extract_features()
    
model_baseline = FashionMNIST_NN(create_convolution_layers(), create_fc_layers())

train_params = {
    "lr" : 0.001,
    "optimizer": "sgd",
    "use_gpu": True,
    "seed": 42
}

trainer_baseline = ModelTrainer(model=model_baseline, trainloader=trainloader, testloader=testloader, train_params=train_params, valloader=valloader)
trainer_baseline.set_seed()
train_losses, val_losses,_ = trainer_baseline.train(epochs=15)

def plot_curves(curves, labels, xlabel = "Epochs", ylabel = "Loss", title = "Training and Validation Loss"):
    for curve, label in zip(curves, labels):
        plt.plot(curve, label = label)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plot_curves([train_losses, val_losses], ["Training Loss", "Validation Loss"])

accracy = trainer_baseline.predict()

def plot_images(images, labels, preds):
    images, labels = images.cpu(), labels.cpu()
    fig = plt.figure(figsize = (25, 4))
    for idx in range(20):
        ax = fig.add_subplot(2, int (20/2), idx + 1, xticks=[], yticks = [])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title("[] ([])".format(label_map[preds[idx].item()], label_map[labels[idx].item()]), color = ("green" if preds[idx]==labels[idx] else "red"))
    plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(trainer_baseline.device), labels.to(trainer_baseline.device)
# Predict
output = model_baseline(images)
_, preds = torch.max(output, 1)
plot_images(images, labels, preds)