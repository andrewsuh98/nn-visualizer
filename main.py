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
    batch_size = 64
    lr = 0.01
    epochs = 10

    print(f"Using device: {device}")

    train_loader, test_loader = load_data(batch_size)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, epoch)
        loss, accuracy = evaluate(model, test_loader, criterion)
        print(f"  Epoch {epoch} -- Test Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        print()

    print(f"Final test accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "models/mnist_mlp.pth")
    print("Model saved to models/mnist_mlp.pth")


if __name__ == "__main__":
    main()
