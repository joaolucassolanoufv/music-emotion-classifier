import torch
import torch.nn as nn

class MusicEmotionClassifierNet(torch.nn.Module):
  def __init__(self):
    super(MusicEmotionClassifierNet, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=12, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=5)

    self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(3 * 247, 1000)
    self.fc2 = nn.Linear(1000, 100)
    self.fc3 = nn.Linear(100, 5)
    
    self.dropout = nn.Dropout1d(p=0.4)
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = self.max_pooling(x)
    x = self.dropout(x)

    x = torch.relu(self.conv2(x))
    x = self.max_pooling(x)
    x = self.dropout(x)

    x = x.view(-1, 3 * 247)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

torch.manual_seed(1)

batch_size = 1
in_channels = 12
in_width = 1000

model = MusicEmotionClassifierNet()

with torch.no_grad():
  x = torch.randint(0, 10, (batch_size, in_channels, in_width))
  y_hat = model(x.to(torch.float32))

print("y_hat = ", y_hat.numpy())
