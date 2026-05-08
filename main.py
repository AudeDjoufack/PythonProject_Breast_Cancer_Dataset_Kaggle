import csv

import torch
import torchvision
import torch.optim as optim
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split

print(os.path.exists("Data/Breast_cancer_dataset.csv"))
csv_file = "Data/Breast_cancer_dataset.csv"
df = pd.read_csv(csv_file)
df = df.drop(columns=['id', 'Unnamed: 32'])
print(df.head())
print(df.shape) #(569, 31)
print(df.shape[1]) #31
print(df.shape[1] - 1) #30

#per normalizzare i dati
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

class BreastDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

        self.classes = sorted(self.data['diagnosis'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # separa feature e label
        self.X = self.data.drop(columns=['diagnosis']).values.astype('float32')
        self.y = self.data['diagnosis'].map(self.class_to_idx).values
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx])
        label = torch.tensor(self.y[idx]).long()

        return features, label

batch_size=32
classes = ('B', 'M')
dataset = BreastDataset(df)

#train_dataset = Subset(dataset, range(0, 400))
#test_dataset = Subset(dataset, range(400, 569))
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#train_df, test_df = train_test_split(df, test_size=200, random_state=42)
#train_dataset = BreastDataset(train_df)
#test_dataset = BreastDataset(test_df)

dataset_size = len(dataset)
train_size = dataset_size - 169
test_size = 169

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))
print(classes)

# get some random training data
dataiter = iter(train_loader)
features, labels = next(dataiter)
print(labels)
print(features.shape) #[32, 3] : 32 righe, 3 feature (es. età, dimensione, densità)
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#net = Net(num_features=32, num_classes=2)
net = Net(num_features=df.shape[1] - 1, num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

dati = []
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        features, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 500 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}')
            dati.append((epoch, running_loss / 4))
            running_loss = 0.0
print('Finished Training')

with open("Dati_Training.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)

PATH = './eco_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
features, labels = next(dataiter)

# print images
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net(num_features=df.shape[1] - 1, num_classes=2)
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(features)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
#with torch.no_grad():
#   for data in test_loader:
#        features, labels = data
#        # calculate outputs by running images through the network
#        outputs = net(features)
#        # the class with the highest energy is what we choose as prediction
#        _, predicted = torch.max(outputs, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()


for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(test_loader, 0):
        features, labels = data
        outputs = net(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 500 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}')
            dati.append((epoch, running_loss / 4))
            running_loss = 0.0

print(f'Accuracy of the network on the 169 test images: {100 * correct // total} %')
with open("Dati_Test.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        features, labels = data
        outputs = net(features)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print(correct_pred)
print(total_pred)
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

inputs, labels = data[0].to(device), data[1].to(device)
