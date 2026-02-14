import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_viz(self, x):
        x = x.view(x.size(0), -1)
        fc1_relu = self.relu(self.fc1(x))
        fc2_relu = self.relu(self.fc2(fc1_relu))
        logits = self.fc3(fc2_relu)
        return logits, {
            "fc1_relu": fc1_relu,
            "fc2_relu": fc2_relu,
            "logits": logits,
        }

    def get_architecture(self):
        return [
            {"name": "input", "type": "input", "shape": [1, 28, 28], "weight_key": None},
            {"name": "fc1_relu", "type": "linear", "shape": [128], "weight_key": "fc1"},
            {"name": "fc2_relu", "type": "linear", "shape": [64], "weight_key": "fc2"},
            {"name": "logits", "type": "linear", "shape": [10], "weight_key": "fc3"},
        ]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_viz(self, x):
        conv1_block = self.pool(self.relu(self.conv1(x)))
        conv2_block = self.pool(self.relu(self.conv2(conv1_block)))
        flat = conv2_block.view(conv2_block.size(0), -1)
        fc1_relu = self.relu(self.fc1(flat))
        logits = self.fc2(fc1_relu)
        return logits, {
            "conv1_block": conv1_block,
            "conv2_block": conv2_block,
            "fc1_relu": fc1_relu,
            "logits": logits,
        }

    def get_architecture(self):
        return [
            {"name": "input", "type": "input", "shape": [1, 28, 28], "weight_key": None},
            {"name": "conv1_block", "type": "conv2d", "shape": [8, 14, 14], "weight_key": "conv1"},
            {"name": "conv2_block", "type": "conv2d", "shape": [16, 7, 7], "weight_key": "conv2"},
            {"name": "fc1_relu", "type": "linear", "shape": [64], "weight_key": "fc1"},
            {"name": "logits", "type": "linear", "shape": [10], "weight_key": "fc2"},
        ]


MODEL_CLASSES = {
    "mlp": MLP,
    "cnn": CNN,
}


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    batch_size = 64
    lr = 0.01

    print(f"Using device: {device}")
    print(f"Training model: {args.model}")

    train_loader, test_loader = load_data(batch_size)
    model = MODEL_CLASSES[args.model]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, epoch)
        loss, accuracy = evaluate(model, test_loader, criterion)
        print(f"  Epoch {epoch} -- Test Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        print()

    print(f"Final test accuracy: {accuracy:.2f}%")

    save_path = f"models/mnist_{args.model}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
