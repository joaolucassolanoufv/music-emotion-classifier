import torch
import torch.nn as nn
from inf721_model import MusicEmotionClassifierNet

# Load data loaders from the saved file
loaders_path = './dataloaders.pth'
saved_data = torch.load(loaders_path)
train_loader = saved_data['train_loader']
test_loader = saved_data['test_loader']

def calc_test_loss(model, dataloader, loss_function):
  with torch.no_grad():
    total_loss = 0.0
    for inputs, labels in dataloader:
      outputs = model(inputs)
      y_hat = model(inputs.to(torch.float32))
      total_loss += loss_function(y_hat, labels - 1).item()
    average_loss = total_loss / len(dataloader)
    return average_loss
  
torch.manual_seed(1)

model = MusicEmotionClassifierNet()
loss_function = torch.nn.CrossEntropyLoss()
average_loss = calc_test_loss(model, test_loader, loss_function)

print("Average loss: ", average_loss)

def optimize(model, train_loader, test_loader, learning_rate=0.001, num_epochs=10, outpath='musicemotionclassifiernet.pth'):
  train_losses = []
  test_losses = []

  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
    total_train_loss = 0.0
    for x, y in train_loader:
      optimizer.zero_grad()
      y_hat = model(x.to(torch.float32))
      loss = loss_function(y_hat, y - 1)
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    train_losses.append(average_train_loss)
    average_test_loss = calc_test_loss(model, test_loader, loss_function)
    test_losses.append(average_test_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Test Loss: {average_test_loss}')
  torch.save(model.state_dict(), outpath)
  return train_losses, test_losses

torch.manual_seed(1)

model = MusicEmotionClassifierNet()

num_epochs = 10
train_losses, test_losses = optimize(model, train_loader, test_loader, num_epochs=num_epochs)