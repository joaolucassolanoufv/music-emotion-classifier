import torch
from inf721_dataset import get_data_loaders
from inf721_model import MusicEmotionClassifierNet

train_loader, test_loader, val_loader = get_data_loaders()

def evaluate_model_accuracy(model, dataloader):
  with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in dataloader:
      outputs = model(inputs)
      predicted = torch.argmax(outputs, dim=1)
      correct += (predicted == labels - 1).sum().item()

      total += labels.size(0)

      return correct/total
    
model = MusicEmotionClassifierNet()
model.load_state_dict(torch.load('musicemotionclassifiernet.pth'))
model.eval()

train_accuracy = evaluate_model_accuracy(model, train_loader)
test_accuracy = evaluate_model_accuracy(model, test_loader)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)