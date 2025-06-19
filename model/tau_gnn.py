import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/home/zfang31/tau-decay-ml/data/reco_alldata.csv')
df_gen = pd.read_csv('/home/zfang31/tau-decay-ml/data/gen_alldata.csv')

X = df_gen[['pi1_pt', 'pi1_eta', 'pi1_phi',
            'pi2_pt', 'pi2_eta', 'pi2_phi',
            'pi3_pt', 'pi3_eta', 'pi3_phi']].values
Y = df[['neu_pt', 'neu_eta', 'neu_phi']].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).reshape(-1, 3, 3)
X_test = scaler.transform(X_test).reshape(-1, 3, 3)

# Graph dataset
class TauDecayGraphDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def len(self):
        return len(self.X)

    def get(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(0)
        edge_index = torch.tensor([[0, 1, 0, 2, 1, 2],
                                   [1, 0, 2, 0, 2, 1]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

train_dataset = TauDecayGraphDataset(X_train, Y_train)
test_dataset = TauDecayGraphDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Model
class TauGNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TauGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
predictions, truths = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        predictions.append(out.cpu())
        truths.append(batch.y.cpu())

predictions = torch.cat(predictions, dim=0).numpy()
truths = torch.cat(truths, dim=0).numpy()

# Plot results
plt.figure(figsize=(5, 5))
plt.scatter(truths[:, 0], predictions[:, 0])
plt.title('Prediction of pT')
plt.xlabel('True pT')
plt.ylabel('Predicted pT')
plt.savefig('gnn_histograms/pt_plot_gnn.png')

plt.figure(figsize=(5, 5))
plt.scatter(truths[:, 1], predictions[:, 1])
plt.title('Prediction of eta')
plt.xlabel('True eta')
plt.ylabel('Predicted eta')
plt.savefig('gnn_histograms/eta_plot_gnn.png')

plt.figure(figsize=(5, 5))
plt.scatter(truths[:, 2], predictions[:, 2])
plt.title('Prediction of phi')
plt.xlabel('True phi')
plt.ylabel('Predicted phi')
plt.savefig('gnn_histograms/phi_plot_gnn.png')
