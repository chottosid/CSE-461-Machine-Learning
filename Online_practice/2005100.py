import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        
        # TODO: Squeeze layer
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels,kernel_size=1),
            nn.ReLU()
        )
        # TODO: Expand layer
        self.block21 =nn.Sequential(
            nn.Conv2d(squeeze_channels,expand_channels,kernel_size=1),
            nn.ReLU()
        )
        self.block22=nn.Sequential(
            nn.Conv2d(squeeze_channels,expand_channels,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x1 = self.block21(x)
        x2 = self.block22(x)
        final = torch.cat([x1,x2],dim=1)
        return final


class SqueezeLite(nn.Module):
    def __init__(self):
        super(SqueezeLite, self).__init__()
        
        # TODO :  SqueezeNet-like architecture for MNIST
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            FireModule(in_channels=32,squeeze_channels=8,expand_channels=16),
            nn.MaxPool2d(kernel_size=4,padding=1,stride=2),
            FireModule(in_channels=32,squeeze_channels=16,expand_channels=32),
            nn.Conv2d(in_channels=64,out_channels=10,kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
       # TODO : Implement forward pass
       x = self.block(x)
       m = nn.Flatten()
       return m(x)

def train_model():

    # MNIST Transform (single channel)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=100,
        shuffle=True
    )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False
    )

    model = SqueezeLite().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    print("Starting Training...")

    model.train()
    for epoch in range(5):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}')
                running_loss = 0.0

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    train_model()