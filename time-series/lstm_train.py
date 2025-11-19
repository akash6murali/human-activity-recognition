import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score


class ActivityDataset(Dataset):
    def __init__(self, X, Y, label_encoder=None):
        self.X = torch.FloatTensor(X)
        
        if label_encoder is None:
            unique_labels = np.unique(Y)
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_encoder = label_encoder
        
        self.Y = torch.LongTensor([self.label_encoder[label] for label in Y])
        self.idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Simple 1-layer LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_classes=10, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # No dropout for single layer
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, 1000, 3)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.dropout(h_n[-1])  # (batch, hidden_size)
        out = self.fc(out)  # (batch, num_classes)
        
        return out

# Loading function
def load_prepared_data(data_dir='/courses/DS5500.202610/data/team3/preprocessed', schema='WillettsSpecific2018'):
    
    print(f"Loading prepared data from: {data_dir}")
    
    X = np.load(os.path.join(data_dir, 'X.npy'))
    Y = np.load(os.path.join(data_dir, f'Y_{schema}.npy'), allow_pickle=True)
    T = np.load(os.path.join(data_dir, 'T.npy'), allow_pickle=True)
    P = np.load(os.path.join(data_dir, 'P.npy'), allow_pickle=True)
    
    print(f"\nLoaded data:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Number of participants: {len(np.unique(P))}")
    
    return X, Y, T, P

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return total_loss / len(val_loader), 100 * correct / total, f1


# Main training function
def train_lstm(X, Y, P, n_participants=100):
    """
    Train LSTM on N participants
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Filter for N participants
    train_pids = [f'P{i:03d}' for i in range(1, n_participants + 1)]
    test_pids = [f'P{i:03d}' for i in range(101, 150)]  # P101-P110 for testing
    
    train_mask = np.isin(P, train_pids)
    test_mask = np.isin(P, test_pids)
    
    # Remove NaN labels
    valid_train = train_mask & (Y != 'nan') & (~pd.isna(Y))
    valid_test = test_mask & (Y != 'nan') & (~pd.isna(Y))
    
    X_train = X[valid_train]
    Y_train = Y[valid_train]
    X_test = X[valid_test]
    Y_test = Y[valid_test]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} windows from {n_participants} participants")
    print(f"  Test: {len(X_test):,} windows from 10 participants")
    
    # Create datasets
    train_dataset = ActivityDataset(X_train, Y_train)
    test_dataset = ActivityDataset(X_test, Y_test, label_encoder=train_dataset.label_encoder)
    
    # Train/val split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model
    num_classes = len(train_dataset.dataset.label_encoder)
    model = SimpleLSTM(input_size=3, hidden_size=256, num_classes=num_classes)
    model = model.to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Training loop
    best_val_acc = 0
    epochs = 50
    
    print("\n" + "="*70)
    print("Training LSTM")
    print("="*70)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'lstm_{n_participants}p.pth')
            print(f"✓ Saved best model")
    
    # Final test
    print("\n" + "="*70)
    print("Test Evaluation")
    print("="*70)
    
    model.load_state_dict(torch.load(f'lstm_{n_participants}p.pth'))
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1-Score: {test_f1:.4f}")
    
    return model, test_acc, test_f1



X, Y, T, P = load_prepared_data(schema='WillettsSpecific2018')
model, test_acc, test_f1 = train_lstm(X, Y, P, n_participants=100)

