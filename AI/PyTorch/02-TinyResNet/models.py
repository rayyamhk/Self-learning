import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils import log
from PIL import Image

train_dataset = datasets.ImageFolder(
    root="./data/seg_train/seg_train",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150, 150))
    ])
)

dev_dataset = datasets.ImageFolder(
    root="./data/seg_test/seg_test",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150, 150))
    ])
)

train_dataLoader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
)

dev_dataLoader = DataLoader(
    dataset=dev_dataset,
    batch_size=64,
    shuffle=True
)

class TinyResNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_0 = nn.Conv2d(64, 128, 1, stride=2) # Skip connection resizing
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_0 = nn.Conv2d(128, 256, 1, stride=2) # Skip connection resizing
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_0 = nn.Conv2d(256, 512, 1, stride=2) # Skip connection resizing
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.avgpool = nn.AvgPool2d(5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 6)
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        self.to(device)
        self.labels = {
            0: 'buildings',
            1: 'forest',
            2: 'glacier',
            3: 'mountain',
            4: 'sea',
            5: 'street'
        }

    def forward(self, imgs):
        bs = imgs.shape[0]
        assert imgs.shape == (bs, 3, 150, 150), "Incorrect shape for input image"

        # Layer 1
        imgs = self.conv1(imgs) # N * 64 * 150 * 150
        imgs = self.bn_64(imgs)
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 64, 75, 75), "Incorrect shape for output of layer 1"
        
        # Layer 2
        imgs = self.maxpool(imgs) # N * 64 * 37 * 37

        sc = imgs

        imgs = self.conv2(imgs)
        imgs = self.bn_64(imgs)
        imgs = self.relu(imgs)

        imgs = self.conv2(imgs)
        imgs = self.bn_64(imgs)
        imgs += sc
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 64, 37, 37), "Incorrect shape for output of layer 2"

        # Layer 3
        sc = self.conv3_0(imgs) # N * 128 * 19 * 19
        sc = self.bn_128(sc)
        
        imgs = self.conv3_1(imgs) # N * 128 * 19 * 19
        imgs = self.bn_128(imgs)
        imgs = self.relu(imgs)

        imgs = self.conv3_2(imgs)
        imgs = self.bn_128(imgs)
        imgs += sc
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 128, 19, 19), "Incorrect shape for output of layer 3"

        # Layer 4
        sc = self.conv4_0(imgs) # N * 256 * 10 * 10
        sc = self.bn_256(sc)

        imgs = self.conv4_1(imgs) # N * 256 * 10 * 10
        imgs = self.bn_256(imgs)
        imgs = self.relu(imgs)

        imgs = self.conv4_2(imgs)
        imgs = self.bn_256(imgs)
        imgs += sc
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 256, 10, 10), "Incorrect shape for output of layer 4"

        # Layer 5
        sc = self.conv5_0(imgs) # N * 512 * 5 * 5
        sc = self.bn_512(sc)

        imgs = self.conv5_1(imgs) # N * 512 * 5 * 5
        imgs = self.bn_512(imgs)
        imgs = self.relu(imgs)

        imgs = self.conv5_2(imgs)
        imgs = self.bn_512(imgs)
        imgs += sc
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 512, 5, 5), "Incorrect shape for output of layer 5"

        # Classification layer
        imgs = self.avgpool(imgs) # N * 512 * 1 * 1
        imgs = self.flatten(imgs) # N * 512
        imgs = self.fc(imgs)
        assert imgs.shape == (bs, 6), "Incorrect shape for output of classification layer"

        return imgs

    def fit(self, train_dataLoader, dev_dataLoader, batch_size, epochs=1):
        self.train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        size = len(train_dataLoader.dataset)

        log("Training has started")

        for epoch in range(epochs):
            running_loss, count = 0.0, 0
            for batch, (imgs, labels) in enumerate(train_dataLoader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.forward(imgs)
                loss = loss_fn(pred, labels)
                running_loss += loss.item() * len(labels)
                count += len(labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 50 == 49 or batch == len(train_dataLoader) - 1:
                    log("Epoch %d [%d/%d]: training error is %.3f" % (epoch + 1, (batch + 1) * batch_size if batch_size == len(labels) else batch * batch_size + len(labels), size, running_loss / count))
                    running_loss, count = 0.0, 0

            with torch.no_grad():
                for batch, (imgs, labels) in enumerate(dev_dataLoader):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    pred = self.forward(imgs)
                    running_loss += loss_fn(pred, labels).item() * len(labels)
                    count += len(labels)
                log("Epoch %d: validation error is %.3f" % (epoch + 1, running_loss / count))
                running_loss, count = 0.0, 0

        log("Training has finished")
    
    def evaluate(self, test_dataLoader):
        self.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for imgs, labels in test_dataLoader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.forward(imgs)
                pred = self.softmax(pred)
                pred = torch.argmax(pred, dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        log("Accuracy: %.3f%%" % (correct / total))

    def predict(self, imgs):
        self.eval()
        with torch.no_grad():
            imgs = imgs.to(self.device)
            pred = self.forward(imgs)
            pred = self.softmax(pred)
            pred = torch.argmax(pred, dim=1).item()
            return labels[pred]

    def save(self, path="weights.pth"):
        log("Saving trained model weights to %s..." % path)
        trained_weights = self.state_dict()
        torch.save(trained_weights, path)
        log("Completed")

    def load(self, path):
        log("Loading pre-trained model weights from %s..." % path)
        weights = torch.load(path)
        self.load_state_dict(weights)
        log("Completed") 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = TinyResNet(device)
    network.fit(train_dataLoader, dev_dataLoader, 64, epochs=10)
    network.save()
    # network.load("weights.pth")
    # img = Image.open("./data/seg_pred/seg_pred/24330.jpg")
    # pred = network.predict(transforms.ToTensor()(img).view(1, 3, 150, 150).to(device))
    # print(pred)