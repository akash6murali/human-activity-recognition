import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize W&B
def init_wandb(config):
    """
    Initialize Weights & Biases
    """
    wandb.init(
        project="human-activity-recognition",  # project name
        name=f"1DCNN-local-{config['window_size']}s",  # Run name
        config=config,
        tags=["CNN", "baseline", "capture24"]
    )
    
    return wandb.config


# Dataset class
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


# Model definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        # Input: (batch, 3, 1000)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, 1000, 3) -> (batch, 3, 1000)
        x = x.transpose(1, 2)
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    
    for inputs, labels in pbar:
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


# Validation function
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} - Validation")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    # Calculate F1 score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, f1_macro, f1_weighted, all_preds, all_labels


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, epoch):
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - Epoch {epoch}')
    plt.tight_layout()
    
    return fig


# Main training function
def train_model(config):
    """
    Main training function with W&B integration
    """
    # Initialize W&B
    wandb_config = init_wandb(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    X = np.load(config['data_dir'] + '/X.npy')
    Y = np.load(config['data_dir'] + f"/Y_{config['label_schema']}.npy", allow_pickle=True)
    P = np.load(config['data_dir'] + '/P.npy', allow_pickle=True)
    
    print(f"Data loaded: {X.shape}, {Y.shape}")
    print(f"Unique labels: {np.unique(Y)}")
    
    # Split by participant (as in paper)
    train_pids = [f'P{i:03d}' for i in range(1, 101)]  # P001-P100
    train_mask = np.isin(P, train_pids)
    
    X_train, X_test = X[train_mask], X[~train_mask]
    Y_train, Y_test = Y[train_mask], Y[~train_mask]
    
    print(f"\nTrain set: {len(X_train):,} windows")
    print(f"Test set: {len(X_test):,} windows")
    
    # Create datasets
    train_dataset = ActivityDataset(X_train, Y_train)
    test_dataset = ActivityDataset(X_test, Y_test, label_encoder=train_dataset.label_encoder)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    num_classes = len(train_dataset.dataset.label_encoder)
    model = SimpleCNN(num_classes=num_classes, dropout=config['dropout'])
    model = model.to(device)
    
    # Log model architecture to W&B
    wandb.watch(model, log='all', log_freq=100)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1_macro, val_f1_weighted, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/f1_macro': val_f1_macro,
            'val/f1_weighted': val_f1_weighted,
            'learning_rate': current_lr
        })
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val F1 (macro): {val_f1_macro:.4f}, Val F1 (weighted): {val_f1_weighted:.4f}")
        
        # Log confusion matrix every 5 epochs
        if epoch % 5 == 0:
            class_names = [train_dataset.dataset.idx_to_label[i] for i in range(num_classes)]
            cm_fig = plot_confusion_matrix(val_labels, val_preds, class_names, epoch)
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_encoder': train_dataset.dataset.label_encoder
            }, 'best_model.pth')
            
            # Save to W&B
            wandb.save('best_model.pth')
            
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final test evaluation
    print("\n" + "="*70)
    print("Final Test Evaluation")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 'Final'
    )
    
    # Classification report
    class_names = [train_dataset.dataset.idx_to_label[i] for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        digits=3
    )
    
    print("\nTest Classification Report:")
    print(report)
    
    # Log final test metrics
    wandb.log({
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        'test/f1_macro': test_f1_macro,
        'test/f1_weighted': test_f1_weighted
    })
    
    # Log final confusion matrix
    cm_fig = plot_confusion_matrix(test_labels, test_preds, class_names, 'Final Test')
    wandb.log({"test/confusion_matrix": wandb.Image(cm_fig)})
    
    # Log classification report as table
    report_dict = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True
    )
    
    report_table = wandb.Table(
        columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        data=[[k, v['precision'], v['recall'], v['f1-score'], v['support']] 
              for k, v in report_dict.items() if k in class_names]
    )
    wandb.log({"test/classification_report": report_table})
    
    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Final test accuracy: {test_acc:.2f}%")
    
    wandb.finish()
    
    return model, test_acc


# Main execution
if __name__ == "__main__":
    
    # Configuration
    config = {
        # Data
        'data_dir': '/Users/akashmurali/Documents/capstone/project/capture24/preprocessed',
        'label_schema': 'WillettsSpecific2018',
        'window_size': 10,
        
        # Model
        'dropout': 0.5,
        
        # Training
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 0.001,
        'patience': 10,
        
        # System
        'num_workers': 4,
        'seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Train model
    model, test_acc = train_model(config)