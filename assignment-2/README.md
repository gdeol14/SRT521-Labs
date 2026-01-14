# Assignment 2 Transformer-Based ML Pipeline 

**Student Name:** Gurmandeep Deol  
**Student ID:** 104120233  
**Assignment Number:** Assignment 2   
**Completion Date:** 2025-11-30  
**Course:** SRT521 - Advanced Data Analysis for Security  
**Course Section Number:** NBB  

---

# Overview
__**This assignment implements a advance machine learning pipeline for phishing website detection using transformer-based models**__ 

- **BERT** (text-based transformer)  
- **TabTransformer** (tabular data transformer)  
- **Hybrid Model** (fusion of BERT + TabTransformer)  
- **Baseline Models** (Random Forest, XGBoost, Logistic Regression)  

## Assignment 2 Objectives
- Implement transformer models for security data
- Apply transfer learning and fine-tuning techniques
- Compare transformer performance with traditional ML methods
- Understand attention mechanisms and model interpretability
- Deploy models using modern ML frameworks
---
# Features
- **Multi-Model Architecture**: Implements up to 5 different models  
- **Hybrid Fusion**: Combines text and numerical features using a custom hybrid architecture  
- **Hyperparameter Tuning**: Automated grid search and random search  
- **Efficiency Analysis**: Training time, resource usage metrics  
- **Visualizations**: Training curves, ROC curves, confusion matrices, feature importance  
- **Interactive CLI**: Step-by-step execution with a menu interface  

## Key Findings
- **__Near-Perfect Performance: All models achieved >99.8 accuracy__**
- **__Hybrid Model: Sucesfully combined text and numerical features__**

# Project Structure

| Path                                      | Description                          |
|-------------------------------------------|--------------------------------------|
| `assignment-2/`                           | Root project directory               |
| ├── `src/`                                | Source code package                  |
| │   ├── `__init__.py`                      | Package initialization               |
| │   ├── `data_loader.py`                   | Data loading and preprocessing       |
| │   ├── `bert_model.py`                    | BERT implementation                  |
| │   ├── `tabtransformer.py`                | TabTransformer implementation        |
| │   ├── `baseline_models.py`               | Traditional ML models                |
| │   ├── `hybrid_model.py`                  | Hybrid fusion model                  |
| │   ├── `evaluation.py`                    | Model evaluation utilities           |
| │   ├── `visualization.py`                 | Visualization generation             |
| │   ├── `hyperparameter_tuning.py`         | Hyperparameter optimization          |
| │   ├── `computational_efficiency.py`      | Efficiency analysis                  |
| │   └── `utils.py`                         | Helper functions                     |
| ├── `main_pipeline.py`                     | Main execution script                |
| ├── `requirements.txt`                     | Python dependencies                  |
| ├── `README.md`                            | Project README                        |
| └── `results/`                             | Output directory (auto-created)     |
|     ├── `models/`                          | Saved models                          |
|     ├── `plots/`                           | Generated visualizations             |
|     ├── `logs/`                            | Execution logs                        |
|     ├── `model_comparison.csv`             | Performance comparison               |
|     ├── `hyperparameter_tuning_results.json` | Hyperparameter tuning results      |
|     ├── `computational_efficiency_metrics.json` | Efficiency metrics              |
|     └── `confusion_matrices.json`         | Confusion matrices                   |

# Installation

### Prerequisites
- Python 3.8 or higher  
- CUDA-compatible GPU (recommended for faster BERT training)  
- 8-16 GB RAM recommended  

# Usage

**Interactive Mode:**  
```bash
python main_pipeline.py
```
Then a menu will come up. Select what you would like to do. Number 11 will run the entire pipeline.
---
# Pipeline Options

| #  | Action                                      |
|----|--------------------------------------------|
| 1  | Load and Prepare Data                        |
| 2  | Train BERT Model (Text-based)               |
| 3  | Train TabTransformer (Numerical features)  |
| 4  | Train Baseline Models (Random Forest, XGBoost, etc.) |
| 5  | Train Hybrid Model (BERT + TabTransformer fusion) |
| 6  | Evaluate All Models                          |
| 7  | Generate Visualizations                      |
| 8  | Hyperparameter Tuning                        |
| 9  | Computational Efficiency Analysis           |
| 10 | Save Results                                 |
| 11 | Run Complete Pipeline (All steps)           |
| 12 | Exit                                         |
---

# Models Implemented

## BERT
- **Architecture:** DistilBERT  
- **Input:** Combined URL, Domain, and Title Text  
- **Fine-tuning:** 3 epochs with early stopping  
- **Use Case:** Capture patterns in phishing URLs  

## TabTransformer
- **Architecture:** Custom transformer for tabular data  
- **Input:** Numerical features  
- **Use Case:** Learn relationships between numerical features  

## Hybrid Model
- **Architecture:** Fusion of BERT embeddings and tabular features  
- **Numerical:** Processes tabular features  
- **Fusion Layer:** Combines both text and numerical information  
- **Use Case:** Leverage both text and numerical information  

## Baseline Models
- **Random Forest:** Optimized via grid search  
- **XGBoost:** Boosting with hyperparameter tuning  
- **Logistic Regression:** Simple baseline


# Results
## Model Comparison

📊 Comparing 6 models:

| Model               | Accuracy   | Precision  | Recall    | F1 Score  |
|--------------------|-----------|-----------|----------|-----------|
| XGBoost            | 1.000000  | 1.000000  | 1.000000 | 1.000000  |
| Random Forest      | 1.000000  | 1.000000  | 1.000000 | 1.000000  |
| Hybrid Model       | 0.999972  | 0.999951  | 1.000000 | 0.999975  |
| TabTransformer     | 0.999887  | 0.999802  | 1.000000 | 0.999901  |
| Logistic Regression| 0.999887  | 0.999802  | 1.000000 | 0.999901  |
| BERT               | 0.998502  | 0.998468  | 0.998912 | 0.998690  |

🏆 **Best Model:** XGBoost

## 📈 Outputs Generated

### 📊 Visualizations
- **training_curves.png** – Training/validation loss and accuracy  
- **model_comparison.png** – Performance metrics comparison  
- **confusion_matrices.png** – Confusion matrices for all models  
- **roc_curves.png** – ROC curves with AUC scores  
- **feature_importance.png** – Top 20 important features  
- **attention_analysis.png** – BERT prediction confidence  
- **training_time_comparison.png** – Training time comparison  
- **inference_speed_comparison.png** – Inference speed vs batch size  
- **resource_usage_comparison.png** – Memory usage comparison  

### 📁 Data Files
- **model_comparison.csv** – Complete performance metrics  
- **metrics_summary.json** – Detailed metrics for all models  
- **confusion_matrices.json** – Raw confusion matrix data  
- **hyperparameter_tuning_results.json** – Tuning results  
- **hyperparameter_tuning_summary.txt** – Human-readable tuning summary  
- **computational_efficiency_metrics.json** – Efficiency metrics  
- **computational_efficiency_summary.txt** – Efficiency analysis summary  

### 🧠 Model Files
- **bert_model/** – Saved BERT model and tokenizer  
- **tabtransformer_model/** – Saved TabTransformer weights  
- **hybrid_model/** – Saved Hybrid model weights  
- **baseline_models/** – Saved sklearn baseline model pickles  

