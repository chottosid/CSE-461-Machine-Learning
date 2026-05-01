import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),  # TODO
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False,
                                            transform=transforms.ToTensor())  # TODO

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# -----------------------------
# Tiny MobileNetV1-style network
# -----------------------------

class ConvBNReLU(nn.Module):
    """
    TODO:
    Implement a standard block: Conv2d -> BatchNorm2d -> ReLU

    Requirements:

    - ReLU should be inplace=True
    - Must support configurable kernel_size, stride, padding
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass


class DepthwiseSeparableConv(nn.Module):
    """
    TODO:
    Implement MobileNetV1 depthwise separable conv block:

        depthwise: 3x3 Conv2d with groups=in_ch (in_ch -> in_ch)
        pointwise: 1x1 Conv2d (in_ch -> out_ch)

    Requirements:
    - Must apply BN+ReLU after depthwise and after pointwise
    - Stride is applied to the depthwise conv (used for downsampling)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass


# Convolutional neural network (MobileNet-like)
class ConvNet(nn.Module):
    """
    You must replace the original 2-layer CNN with a compact MobileNetV1-style architecture.

    What you need to build:

    1) Stem Layer
       - Start with a standard ConvBNReLU block.
       - Input channels = 3 (CIFAR-10 images are RGB).
       - Use stride=1 in the stem to preserve spatial resolution initially.
       - This layer performs the first stage of feature extraction.

    2) Depthwise-Separable Blocks
       - Define a configuration list:
             self.cfg = [(out_channels, stride), ...]
       - For each tuple in this list:
             - Apply a DepthwiseSeparableConv block.
             - If stride=2, the spatial resolution must be reduced by half.
       - Include at least ONE block with stride=2 (to downsample).
       - Avoid too many stride=2 blocks early in the network.

    3) Global Pooling
       - After all convolutional blocks, use:
             nn.AdaptiveAvgPool2d(1)
       - This converts feature maps of shape (N, C, H, W)
         into (N, C, 1, 1), regardless of spatial size.

    4) Classifier
       - Flatten the pooled output.
       - Use a Linear layer:
             nn.Linear(last_channel_size, num_classes)
       - Final output must have shape:
             (N, num_classes)

    Architectural Constraints:
    - Total trainable parameters must be <= 250,000.
    - Do NOT hard-code spatial dimensions (e.g., avoid 8*8*channels).
    - The network must correctly accept input of shape (N, 3, 32, 32).

    Example configuration idea:
        [(32,1), (64,2), (64,1), (128,2), (128,1)]

    You should think carefully about:
    - How stride changes feature map size
    - Why depthwise separable convolutions reduce parameters
    - How channel sizes affect both model capacity and parameter count
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # TODO: implement model definition here
        pass
        
    def forward(self, x):
        # TODO: implement forward pass:
        # x -> stem -> features -> pool -> flatten -> fc
        pass



model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# TODO (optional for assignment difficulty):
# Replace Adam with SGD+momentum+weight_decay and/or add an LR scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
	# TODO 
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
