"""
TabTransformer Module
Transformer architecture for tabular data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

class TabTransformerModel(nn.Module):
    """TabTransformer neural network"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TabTransformerModel, self).__init__()
        
        # Feature embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.fc1 = nn.Linear(d_model * input_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)
        
        self.input_dim = input_dim
        self.d_model = d_model
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Embed each feature
        embedded_features = []
        for i in range(self.input_dim):
            feature = x[:, i:i+1]
            embedded = self.feature_embeddings[i](feature)
            embedded_features.append(embedded)
        
        # Stack: (batch, features, d_model)
        x = torch.stack(embedded_features, dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TabTransformer:
    """TabTransformer wrapper class"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        """
        Initialize TabTransformer
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🏗️  Building TabTransformer...")
        print(f"   Device: {self.device}")
        
        # Create model
        self.model = TabTransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.model = self.model.to(self.device)
        
        print(f"   ✓ Input features: {input_dim}")
        print(f"   ✓ Model dimension: {d_model}")
        print(f"   ✓ Attention heads: {nhead}")
        print(f"   ✓ Transformer layers: {num_layers}")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train(self, X_train, X_val, X_test, y_train, y_val, y_test,
             epochs=20, batch_size=128, learning_rate=0.001):
        """
        Train TabTransformer
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        print(f"\n🚀 Training TabTransformer...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc = self._eval_epoch(
                val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        
        print(f"\n   ✓ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def _train_epoch(self, loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def _eval_epoch(self, loader, criterion):
        """Evaluate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, X_test, y_test, batch_size=128):
        """
        Evaluate on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Test metrics
        """
        print(f"\n📊 Evaluating TabTransformer on test set...")
        
        self.model.eval()
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'predictions': all_preds,
            'true_labels': all_labels,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        print(f"   ✓ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ✓ Precision: {metrics['precision']:.4f}")
        print(f"   ✓ Recall:    {metrics['recall']:.4f}")
        print(f"   ✓ F1 Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def save(self, output_dir):
        """
        Save model
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model
            }
        }, output_path / 'tabtransformer.pth')
        
        print(f"   ✓ TabTransformer saved to: {output_path}")