from pathlib import Path
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from fastai.metrics import accuracy, error_rate
# from fastai.vision import models
# from fastai.vision.augment import aug_transforms, Resize
# from fastai.vision.data import ImageDataLoaders
# from fastai.vision.learner import cnn_learner
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import timm
from fastai import *
from fastai.vision.all import *
import multiprocessing


multiprocessing.set_start_method('spawn')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('resnet50.a1_in1k', pretrained=True).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(root='DataSet', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 7


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print(f'Accuracy on train set after epoch {epoch + 1}: {100 * correct / total}%')
    print(f'Accuracy on test set after epoch {epoch + 1}: {100 * correct_test / total_test}%')

print('Training completed')

torch.save(model.state_dict(), 'model.pt')