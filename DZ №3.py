import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(42)

writer = SummaryWriter()

batch_size = 64

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

class MNISTPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 200)
        self.linear2 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return torch.nn.functional.softmax(x, dim=1)

def loss_fn(y_pred, y_true):
    return torch.nn.functional.cross_entropy(y_pred, y_true)

model = MNISTPerceptron().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
best_accuracy = 0

for epoch in range(epochs):
    model.train()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start_time = time.time()

    error = 0.0
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        error += loss.item()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {error / len(train_dataloader):.4f}, Time: {epoch_duration:.2f}s')

    writer.add_scalar('Train loss', error / len(train_dataloader), epoch)


    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            correct += (output.argmax(dim=1) == y).sum().item()

    accuracy = correct / len(test_dataset)
    writer.add_scalar('Test accuracy', accuracy, epoch)
    print(f'Accuracy: {accuracy:.4f}')

    
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_mnist.pt')
        best_accuracy = accuracy

print('Training complete!')

model.load_state_dict(torch.load('best_mnist.pt'))
model.eval()

example = train_dataset[0][0].unsqueeze(0).to(device)
output = model(example)
print(f'Predicted label: {output.argmax(dim=1).item()}')