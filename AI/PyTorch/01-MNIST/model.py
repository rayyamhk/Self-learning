import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms

"""
Load MNIST dataset (60k for training, 10k for testing),
and transform the images to PyTorch tensor.

Split the 10k dataset to validation set (5k) and test set (5k) randomly

Convert all dataset to DataLoader, which is a python iteratable.
You can create a iterator by itr = iter(dataLoader) and get the data by next(itr).
If shuffle=True, the dataset is reshuffled at every epoch (The reshuffle is triggered by iter(dataLoader))
"""
train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

dev_dataset, test_dataset = random_split(test_dataset, [5000, 5000])

train_dataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_dataLoader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
test_dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=True)

"""
Every model in PyTorch is extended by nn.Module (Inheritance)

model.train() sets the model to training mode, which activates some layers such as BatchNormalization and Dropout
model.eval() sets it to evaluation mode, which disable those layers

optimizer is responsible for weights updating only. So different types of optimizer updates weights in different ways

By default, Tensor.backward() accumulates gradient. Therefore, you should reset all gradients by optimizer.zero_grad().

with torch.no_grad() is a context manager that disables gradient calculation during Tensor.backward() to achieve better performance
"""
class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(8*4*4, 10)
        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = self.relu(self.conv1(x)) # 4 * 24 * 24
        x = self.pool(x) # 4 * 12 * 12
        x = self.relu(self.conv2(x)) # 8 * 8 * 8
        x = self.pool(x) # 8 * 4 * 4
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def fit(self, train_dataLoader, dev_dataLoader, epochs=1, lr=1e-3, momentum=0.9):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        loss_fn = nn.CrossEntropyLoss()
        size = len(train_dataLoader.dataset)

        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch, (X, y) in enumerate(train_dataLoader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.forward(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch % 100 == 99:
                    print("Epoch %d - training error: %.3f %d/%d" % (epoch + 1, running_loss / 100, (batch + 1) * len(X) ,size))
                    running_loss = 0

            with torch.no_grad():
                for batch, (X, y) in enumerate(dev_dataLoader):
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.forward(X)
                    loss = loss_fn(pred, y)
                    running_loss += loss
            print("Epoch %d - validation error: %.3f" % (epoch + 1, running_loss / (batch + 1)))
            running_loss = 0

    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.forward(x)
            pred = self.softmax(pred)
            return torch.argmax(pred, dim=1).item()

    def evaluate(self, test_dataLoader):
        self.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in test_dataLoader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.forward(X)
                pred = self.softmax(pred)
                pred = torch.argmax(pred, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
            print('Evaluation: %.3f%%' % (correct / total * 100))

    def save(self, path='weights.pth'):
        print("Saving trained model weights to %s..." % path)
        trained_weights = self.state_dict()
        torch.save(trained_weights, path)
        print("Completed")

    def load(self, path):
        print("Loading pre-trained model weights from %s..." % path)
        weights = torch.load(path)
        self.load_state_dict(weights)
        print("Completed")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork(device)
model.fit(train_dataLoader, dev_dataLoader, epochs=10)
model.evaluate(test_dataLoader)
model.save()
# model.load("weights.pth")
# model.evaluate(test_dataLoader)