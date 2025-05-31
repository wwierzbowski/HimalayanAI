import kagglehub
import pandas as pd
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

from model import NN

def download_data(target_path, source_path):
    if source_path.exists():
        # Copy the file from the cache to the project folder
        try:
            shutil.copy(source_path, target_path)
            # Load the CSV file from the project folder
        except Exception as e:
            print(f"An error occurred during file copy or read: {e}")

# Download the dataset
main_dir = Path(__file__).parent.resolve()
subfolder = "dataset"
data_dir = main_dir / subfolder
data_dir.mkdir(parents=True, exist_ok=True)
datas_cache: Path = Path(kagglehub.dataset_download("siddharth0935/himalayan-expeditions"))

# Expedition file
exped_file = "exped.csv"
source_path = datas_cache / exped_file
download_data(data_dir / exped_file, datas_cache / exped_file)

exped_df = pd.read_csv(main_dir / data_dir / exped_file)

# Data preprocessing
# Drop rows with missing values in key columns
required_columns = ['peakid', 'season', 'route1', 'route2', 'approach', 'success1']
exped_df = exped_df.dropna(subset=required_columns)

# Prepare features
features = ['peakid', 'season', 'route1', 'route2', 'approach']
X = exped_df[features].copy()
y = exped_df['success1'].copy()

# Encode categorical variables
label_encoders = {}
for column in features:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Encode target variable (assuming it's boolean T/F or 1/0)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

# Convert to numpy arrays
X = X.values.astype(np.float32)
y = y.astype(np.int64)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = NN().to(device)  # Move model to the appropriate device

# For binary classification, use BCEWithLogitsLoss or CrossEntropyLoss
# Since your model outputs 1 value, let's use BCEWithLogitsLoss for binary classification
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
train_losses = []
train_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data).squeeze() 
        targets = targets.float()  # Convert to float for BCEWithLogitsLoss
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)
    
    # Calculate average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data).squeeze()
        targets = targets.float()
        
        test_loss += criterion(outputs, targets).item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        test_correct += (predicted == targets).sum().item()
        test_total += targets.size(0)

test_accuracy = test_correct / test_total
test_loss = test_loss / len(test_loader)

print('\nTest Results:')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'expedition_model.pth')
print("Model saved as 'expedition_model.pth'")

# Print final training statistics
print(f'\nFinal Training Loss: {train_losses[-1]:.4f}')
print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')