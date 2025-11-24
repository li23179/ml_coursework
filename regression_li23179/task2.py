import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score


# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)

X = data.drop('charges', axis=1)
y = data['charges']

# Define numerical and categorical features
numerical_features = ['age', 'bmi', 'children']
catagorical_features = ['sex', 'smoker', 'region']

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformer for numerical featuers
scaler = StandardScaler()

# One-Hot Encoder for categorical features
encoder = OneHotEncoder()

# Combine the preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', scaler, numerical_features),
        ('categorical', encoder, catagorical_features),
    ]
)

# fit the train data only into preprocessor
preprocessor.fit(X_train)

# apply both standard scaler and one hot encoder to the training and testing data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# convert into float np array for tensor in pytorch
X_train_np = X_train_transformed.astype(np.float32)
X_test_np = X_test_transformed.astype(np.float32)

y_train_np = y_train.values.astype(np.float32).reshape(-1, 1)
y_test_np = y_test.values.astype(np.float32).reshape(-1, 1)

# convert np array into pytorch tensor
X_train_tensor = torch.from_numpy(X_train_np)
X_test_tensor = torch.from_numpy(X_test_np)

y_train_tensor = torch.from_numpy(y_train_np)
y_test_tensor = torch.from_numpy(y_test_np)

# create a Data Loader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train a neural network model using PyTorch.
input_dim = X_train_tensor.shape[1]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32), # hidden layer
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()

# Use mean squared error as the loss function and an optimiser
# such as Adam. 
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-1)

# training the netural network
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    # reset running loss
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # reset the gradient
        optimiser.zero_grad()
        # forward pass
        y_pred = model(X_batch)
        # Compute loss function
        loss = criterion(y_pred, y_batch)
        # backprop
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * X_batch.size(0)
        
    epoch_loss = running_loss / X_batch.size(0)
    epoch_loss_RMSE = np.sqrt(epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train RMSE: {epoch_loss_RMSE: .3f}")

model.eval()

# Print training and test RMSE
with torch.no_grad():
    y_pred_train_tensor = model(X_train_tensor)
    y_pred_test_tensor = model(X_test_tensor)
    
    train_RMSE = np.sqrt(criterion(y_pred_train_tensor, y_train_tensor).item())
    test_RMSE = np.sqrt(criterion(y_pred_test_tensor, y_test_tensor).item())
    
    train_r2 = r2_score(y_pred_train_tensor, y_train_tensor)
    test_r2 = r2_score(y_pred_test_tensor, y_test_tensor)

print(f"Train RMSE: {train_RMSE: .3f}")
print(f"Test RMSE: {test_RMSE: .3f}")

print(f"Train R^2: {train_r2:.3f}")
print(f"Test R^2: {test_r2:.3f}")


# Plot predicted versus actual charges on the test set
y_test_flat = y_test_np.flatten()
y_pred_test_flat = y_pred_test_tensor.numpy().flatten()

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_test_flat, y_pred_test_flat, edgecolors='black', linewidths=0.5)

min_val = min(y_test_flat.min(), y_pred_test_flat.min())
max_val = max(y_test_flat.max(), y_pred_test_flat.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Perfect Prediction (y = x)")

ax.set_xlabel("Actual Charges")
ax.set_ylabel("Predicted Charges")
ax.set_title("Neural Netword: Predicted vs. Actual Charges (Test Set)")
ax.legend()

plt.savefig("regression_plots/neural_network_plot.png", dpi=300)
plt.show()

