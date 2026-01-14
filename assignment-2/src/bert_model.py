"""
BERT Model Module
Fine-tuning BERT for text classification
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
from pathlib import Path

class BERTModel:
    """BERT model for text classification"""
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        """
        Initialize BERT model
        
        Args:
            model_name: Hugging Face model name
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🤖 Initializing BERT model: {model_name}")
        print(f"   Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model = self.model.to(self.device)
        
        print(f"   ✓ Model loaded: {self.model.num_parameters():,} parameters")
        
        # Load metrics
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=labels)
        precision = self.precision_metric.compute(predictions=predictions, references=labels, average='binary')
        recall = self.recall_metric.compute(predictions=predictions, references=labels, average='binary')
        f1 = self.f1_metric.compute(predictions=predictions, references=labels, average='binary')
        
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'recall': recall['recall'],
            'f1': f1['f1']
        }
    
    def tokenize_data(self, texts, labels):
        """
        Tokenize text data
        
        Args:
            texts: List of text samples
            labels: List of labels
            
        Returns:
            Hugging Face Dataset
        """
        encodings = self.tokenizer(
            texts.tolist() if hasattr(texts, 'tolist') else list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def prepare_datasets(self, X_train_text, X_val_text, X_test_text,
                        y_train, y_val, y_test, sample_size=50000):
        """
        Prepare datasets for training
        
        Args:
            X_train_text, X_val_text, X_test_text: Text features
            y_train, y_val, y_test: Labels
            sample_size: Maximum training samples (for speed)
            
        Returns:
            Dictionary of datasets
        """
        print(f"\n📊 Preparing BERT datasets...")
        
        # Sample training data if too large
        if len(X_train_text) > sample_size:
            print(f"   ⚠️  Sampling {sample_size:,} training samples for speed")
            np.random.seed(42)
            indices = np.random.choice(
                len(X_train_text),
                size=sample_size,
                replace=False
            )
            X_train_text = X_train_text[indices]
            y_train = y_train[indices]
        
        # Tokenize
        train_dataset = self.tokenize_data(X_train_text, y_train)
        val_dataset = self.tokenize_data(X_val_text, y_val)
        test_dataset = self.tokenize_data(X_test_text, y_test)
        
        print(f"   ✓ Train: {len(train_dataset):,} samples")
        print(f"   ✓ Val:   {len(val_dataset):,} samples")
        print(f"   ✓ Test:  {len(test_dataset):,} samples")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=32):
        """
        Train BERT model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results
        """
        print(f"\n🚀 Training BERT model...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./bert_output",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
            logging_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        train_result = trainer.train()
        
        # Store trainer for later use
        self.trainer = trainer
        
        print(f"\n   ✓ Training complete!")
        print(f"   Training loss: {train_result.training_loss:.4f}")
        print(f"   Training time: {train_result.metrics['train_runtime']/60:.2f} min")
        
        # Get training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        # Extract from trainer logs
        for log in trainer.state.log_history:
            if 'loss' in log:
                history['train_loss'].append(log['loss'])
            if 'eval_loss' in log:
                history['val_loss'].append(log['eval_loss'])
                history['val_accuracy'].append(log.get('eval_accuracy', 0))
                history['val_f1'].append(log.get('eval_f1', 0))
        
        return history
    
    def evaluate(self, test_dataset):
        """
        Evaluate on test set
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Test metrics
        """
        print(f"\n📊 Evaluating BERT on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        pred_labels = predictions.predictions.argmax(axis=-1)
        true_labels = predictions.label_ids
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, 
            recall_score, f1_score,
            confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
            'predictions': pred_labels,
            'true_labels': true_labels,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels)
        }
        
        print(f"   ✓ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ✓ Precision: {metrics['precision']:.4f}")
        print(f"   ✓ Recall:    {metrics['recall']:.4f}")
        print(f"   ✓ F1 Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Predictions and probabilities
        """
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        results = classifier(list(texts))
        return results
    
    def save(self, output_dir):
        """
        Save model and tokenizer
        
        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        print(f"   ✓ BERT model saved to: {output_path}")