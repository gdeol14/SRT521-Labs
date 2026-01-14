"""
Hybrid Model Module
Combines BERT text features with TabTransformer numerical features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

class HybridDataset(Dataset):
    """Custom dataset for hybrid model"""
    
    def __init__(self, text_embeddings, numerical_features, labels):
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text_emb': self.text_embeddings[idx],
            'num_feat': self.numerical_features[idx],
            'label': self.labels[idx]
        }

class HybridModelNet(nn.Module):
    """Hybrid neural network combining text and numerical features"""
    
    def __init__(self, text_dim=768, num_dim=100, hidden_dim=256, dropout=0.3):
        super(HybridModelNet, self).__init__()
        
        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Numerical processing branch
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_input_dim = 192 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
    def forward(self, text_emb, num_feat):
        # Process text embeddings
        text_out = self.text_branch(text_emb)
        
        # Process numerical features
        num_out = self.num_branch(num_feat)
        
        # Concatenate features
        combined = torch.cat([text_out, num_out], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output

class HybridModel:
    """Hybrid model wrapper class"""
    
    def __init__(self, text_dim=768, num_dim=100, hidden_dim=256, dropout=0.3):
        """
        Initialize Hybrid Model
        
        Args:
            text_dim: Dimension of text embeddings (BERT output)
            num_dim: Number of numerical features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🔗 Building Hybrid Model...")
        print(f"   Device: {self.device}")
        
        # Create model
        self.model = HybridModelNet(
            text_dim=text_dim,
            num_dim=num_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.model = self.model.to(self.device)
        
        print(f"   ✓ Text embedding dim: {text_dim}")
        print(f"   ✓ Numerical features: {num_dim}")
        print(f"   ✓ Hidden dimension: {hidden_dim}")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_bert_embeddings(self, bert_model, texts):
        """Extract BERT embeddings for text data"""
        print(f"\n📝 Extracting BERT embeddings...")
        
        from transformers import AutoTokenizer
        
        tokenizer = bert_model.tokenizer
        model = bert_model.model
        model.eval()
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts.tolist() if hasattr(batch_texts, 'tolist') else list(batch_texts),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Use [CLS] token embedding from last hidden state
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"   ✓ Extracted embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def train(self, text_embeddings_train, text_embeddings_val, text_embeddings_test,
              X_train_num, X_val_num, X_test_num,
              y_train, y_val, y_test,
              epochs=20, batch_size=64, learning_rate=0.001):
        """
        Train Hybrid Model
        
        Args:
            text_embeddings_train, text_embeddings_val, text_embeddings_test: BERT embeddings
            X_train_num, X_val_num, X_test_num: Numerical features
            y_train, y_val, y_test: Labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        print(f"\n🚀 Training Hybrid Model...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Create datasets
        train_dataset = HybridDataset(text_embeddings_train, X_train_num, y_train)
        val_dataset = HybridDataset(text_embeddings_val, X_val_num, y_val)
        
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
        
        for batch in loader:
            text_emb = batch['text_emb'].to(self.device)
            num_feat = batch['num_feat'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(text_emb, num_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * text_emb.size(0)
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
            for batch in loader:
                text_emb = batch['text_emb'].to(self.device)
                num_feat = batch['num_feat'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(text_emb, num_feat)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * text_emb.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, text_embeddings_test, X_test_num, y_test, batch_size=64):
        """
        Evaluate on test set
        
        Args:
            text_embeddings_test: BERT embeddings for test set
            X_test_num: Numerical features for test set
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Test metrics
        """
        print(f"\n📊 Evaluating Hybrid Model on test set...")
        
        self.model.eval()
        
        test_dataset = HybridDataset(text_embeddings_test, X_test_num, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                text_emb = batch['text_emb'].to(self.device)
                num_feat = batch['num_feat'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(text_emb, num_feat)
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
    
    def predict(self, text_embeddings, numerical_features):
        """
        Make predictions
        
        Args:
            text_embeddings: BERT embeddings
            numerical_features: Numerical features
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        text_tensor = torch.FloatTensor(text_embeddings).to(self.device)
        num_tensor = torch.FloatTensor(numerical_features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(text_tensor, num_tensor)
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
        }, output_path / 'hybrid_model.pth')
        
        print(f"   ✓ Hybrid Model saved to: {output_path}")