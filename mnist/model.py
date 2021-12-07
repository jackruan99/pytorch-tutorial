import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper Parameters
INPUT_SIZE = 784  # 28 x 28
HIDDEN_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.001


# MNIST, DataLoader and Transformation
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size=BATCH_SIZE, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()


# Multilayer Neural Network and Activation Function
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training Loop (Batch Training)
N_TOTAL_STEPS = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)  # (100, 1, 28, 28) -> (100, 784)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{NUM_EPOCHS}, step {i+1}/{N_TOTAL_STEPS}, loss = {loss.item():.4f}')


# Model Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)  # (100, 1, 28, 28) -> (100, 784)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()
        n_samples += labels.shape[0]
    
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}%')


# GPU Support


# Source: https://www.youtube.com/watch?v=c36lUUr864M