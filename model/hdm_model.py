import pickle
# from fast_histogram import histogram1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/reco_alldata.csv')
df_gen = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/gen_alldata.csv')
print(df_gen.shape)
print(df.shape)

torch.set_num_threads(2)

X = df_gen[['pi1_pt', 'pi1_eta', 'pi1_phi',
        'pi2_pt', 'pi2_eta', 'pi2_phi',
        'pi3_pt', 'pi3_eta', 'pi3_phi']].values
Y = df[['neu_pt', 'neu_eta', 'neu_phi']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(X_train, Y_train)
test_dataset = CustomDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=376, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=9433, shuffle=False)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 48)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)
        # self.fc6 = nn.Linear(64, 48)
        #self.fc7 = nn.Linear(64, 48)
        self.fc8 = nn.Linear(48, 24)
        self.fc9 = nn.Linear(24, 12)
        #self.fc10 = nn.Linear(16, 12)
        self.fc11 = nn.Linear(12, 6)
        #self.fc12 = nn.Linear(8, 6)
        self.fc13 = nn.Linear(6, output_dim)
        # self.dropout_a = nn.Dropout(p=0.5)
        # self.dropout_b = nn.Dropout(p=0.2)
        # self.dropout_c = nn.Dropout(p=0.1)
        # she had 12 layers, up to 2560 neurons, all relu with droppout from 0.3, up to 0.5, and then progressively down to 0.05

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout_a(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout_a(x)
        # x = F.relu(self.fc3(x))
        # x = self.dropout_a(x)
        # x = F.relu(self.fc4(x))
        # x = self.dropout_b(x)
        # x = F.relu(self.fc5(x))
        # x = self.dropout_c(x)
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))

        return x

# Instantiate the model
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
#print(output_dim)
model = SimpleDNN(input_dim, output_dim)

# Loss and optimizer
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for features, labels in train_loader:
        # Forward pass
        outputs = model(features)
        #print(features[0])
        #print(len(features))
        #print(outputs.squeeze())
        loss = criterion(outputs.squeeze(), labels)
        #calculated_tau_mass = inv_tau_mass(features.detach().numpy(), outputs.squeeze().detach().numpy())
        #loss = criterion(calculated_tau_mass.requires_grad_().float(), torch.full(calculated_tau_mass.shape, 1.7769).float())
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # scheduler.step(loss)


# Evaluation
model.eval()
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(f'Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}')
        count = count + param.numel()
print("Total param count:" + str(count))
with torch.no_grad():
    # correct = 0
    # total = 0
    # for features, labels in test_loader:
    #     outputs = model(features)
    #     predicted = (outputs.squeeze() > 0.5).float()
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

    for features, labels in test_loader:
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        print(loss)

    print(f'Eval Loss: {loss.item():.4f}')


# Create a scatter plot for the specified column
plt.figure(figsize=(5, 5))
# print(Y_test[:,0].size)
# print(outputs[:,0].size())
plt.scatter(Y_test[:, 0], outputs[:, 0])
plt.title(f'Prediction of pT')
plt.xlabel('Validation Set pT')
plt.ylabel('Prediction of pT from Model')
plt.savefig('pt_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 1], outputs[:, 1])
plt.title(f'Prediction of eta')
plt.xlabel('Validation Set eta')
plt.ylabel('Prediction of eta from Model')
plt.savefig('eta_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], outputs[:, 2])
plt.title(f'Prediction of phi')
plt.xlabel('Validation Set phi')
plt.ylabel('Prediction of phi from Model')
plt.savefig('phi_plot.png')
