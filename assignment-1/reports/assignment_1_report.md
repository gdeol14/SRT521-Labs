# Machine Learning Pipeline For Phishing Detection: A Comprehensive Approach

**Student Name:** Gurmandeep Deol  
**Student ID:** 104120233  
**Assignment Number:** Assignment 1  
**Completion Date:** 2025-11-01  
**Course:** SRT521 - Advanced Data Analysis for Security  
**Course Section Number:** (NBB)  
**instructor:** Hamed Haddadpajouh
## Abstract
This report presents a comprehensive machine learning pipeline for detecting phishing websites which is a critical cybersecurity challenge affecting millions of internet users daily My solution implements supervised learning (classification) and unsupervised learning (clustering) techniques to identify malicious urls with up to 99.61% F1 score. This pipeline has been tested with six diverse phishing datasets engineers 8 domain-specific features and compares four machine learning algorithms (Random Forest, XGBoost, Logistic Regression, SVM) The results I have gotten demonstrate that tree based models such as XGBoost have far better performance in distinguishing between phishing websites from legitimate websites this pipelines is production ready handling data preprocessing model training evaluation hyperparameter tuning and automated reporting

## Introduction

### Problem Statement

Phishing is a cybersecurity attack where malicious actors create fake websites to impersonate legitimate organizations such as banks social media to steal user credentials, financial information and more according to the anti phishing working group over 1.2 million phishing attacks occur monthly costing businesses and individuals billion of dollars annually 

### Traditional phishing detection relies on
Blacklist: A predefined database of known phishing urls this is good but is outdated and not practical nowadays
Patterns: The patterns for traditional phishing are manual can easily be bypassed by threat actors
User Reporting: This is unreliable as not everyone will report phishing emails to some people a phishing email might be obvious they might not report it but to some the opposite happens.

### Problem: These methods have high false negative rates they miss new phishing sites and cant adapt to attackers as they are always evolving 

### The solution: Machine learning can automatically learn patterns from millions of examples detecting known and new phishing attempts with higher accuracy and minimal human intervention

## Research Questions

- Which machine learning algorithms are most effective for phishing detection

- What URL and content features are best at distinguishing phishing from legitimate websites

- Can unsupervised learning reveal hidden pattern in phishing behavior

## Dataset Description
- I evaluated this pipeline on six different real world phishing datasets

| Dataset | Samples  | Features | Classes                | Notes                    |
|----------|-----------|-----------|------------------------|---------------------------|
| Dataset 1 | 11,055   | 32        | 2 (binary: -1, 1)     | Classic phishing features |
| Dataset 2 | 88,647   | 112       | 23 (multi-class)      | URL characteristics       |
| Dataset 3 | 11,430   | 89        | 2 ("legitimate", "phishing") | Mixed data types      |
| Dataset 4 | 235,795  | 56        | 2 (binary: 0, 1)      | Large-scale dataset       |
| Dataset 5 | 88,000+  | 111+      | Various               | Domain features           |
| Dataset 6 | 10,000   | 50        | 2 (binary: 0, 1)      | Balanced classes          |

## Key Characteristics
- Imbalanced classes some datasets had 60% phishing and 40% legitimate
- Mixed data types: The datasets had multiple data types such as string numeric categorical text
- Missing values: There were some missing values but my pipeline took care of it 

## Methodology
- Pipeline Architecture
- Raw CSV -> Load -> EDA -> Preprocess -> Train -> Evaluate -> Tune -> Cluster -> Deploy

## Design Principles
- Modularity: Each stage is independent in it's own file
- Robustness: Handles encoding errors,missing values,label inconsistencies 
- Scalability: Efficiently processes datasets from 10k to 235k samples
- Data Preprocessing
- Data Loading and Validations
- ChallengeL Different datasets use different character encodings
- Solution Implemented auto detection that tries up to 8 encodings
## Validation Checks
- Missing values detection 
- Duplicate rows identifications
- class imbalance analysis
## Features engineering
- I created 8 phishing-specific features 
## Feature Engineering Summary

| Feature                  | Formula / Logic                     | Rationale                                                                 |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------|
| **URL_Length_Category**   | Binned: 0–50, 50–75, 75–150, 150+   | Phishing sites often use long URLs to hide malicious domains.             |
| **Has_Multiple_Subdomains** | NoOfSubDomain > 1                 | Attackers add subdomains to mimic trusted brands (e.g., paypal.com.fake.com). |
| **Content_URL_Complexity** | LineOfCode / (URLLength + 1)       | Legitimate sites have richer content, while phishing pages are minimal.   |
| **External_Dependency_Ratio** | ExternalRefs / (TotalRefs + 1)  | Phishing pages often depend on external resources to avoid detection.     |
| **Form_Security_Risk**    | HasForm + HasPassword (0–2)         | Forms with password fields indicate potential credential harvesting.      |
| **Tech_Stack_Indicator**  | log(CSS + JS + 1) (normalized)      | Legitimate sites use modern web technologies; phishing sites usually don’t. |
| **Trust_Signals_Count**   | Favicon + Description + Title        | Phishing pages often lack proper branding and metadata.                   |
| **Special_Char_Density**  | SpecialChars / (URLLength + 1)      | High usage of @, -, _, = often means URL obfuscation.                     |

## Label Encoding
- Challenge: Datasets use inconsistency labels such as 
- Dataset1 has -1 as phishing and 1 as legitimate
- Dataset 3 has phishing as a string and legitimate as a string
- Dataset 2 has 1,2,3,24 with missing classes
- Solution: Had to implement a label encoder that detects and maps the labels XGBoost needs labels to train properly
## Data Splitting
- Training set: 80%
- Test set: 20%
## Model Selection and training
- I trained 4 baseline models 
- Random Forest
- XGBoost
-Logistic Regression
- Support Vector Machine
- Hyperparameter tuning
- Tested 10 random combination forms

## 2.4.1 Evaluation Metrics

Here we describe the key metrics used to evaluate models:

| Metric       | Formula                                           | Interpretation                                                   |
|--------------|--------------------------------------------------|------------------------------------------------------------------|
| **Accuracy**  | (TP + TN) / Total                                | Overall correctness                                              |
| **Precision** | TP / (TP + FP)                                   | Of predicted phishing, how many are correct?                     |
| **Recall**    | TP / (TP + FN)                                   | Of actual phishing, how many did we catch?                       |
| **F1 Score**  | 2 × (Precision × Recall) / (Precision + Recall)  | Harmonic mean — balances precision and recall                    |
| **AUC-ROC**   | Area under ROC curve                             | Measures separability across thresholds                          |

## F1 Score
- Balances false positives and false negatives
- Robust to class imbalance
## Cross Validation
- 5 Fold cross validation
- Process
- Split training data into 5 equal parts
- Train on 4 folds test on 1 fold
- Repeat 5 times
- Purpose
## Verify model stability
- detect overfitting(High train score Low CV Score)
- Fair comparison between models
## Unsupervised Learning
- Objective: Discover hidden patterns in phishing behaviors without labels
- K-Means Clustering
- Test K=2 to K=10 clusters
- Cluster 1: Simple Phishing(Short URLS, no HTTPS)
- Cluster 2 Sophisticated Phishing(Long URLs, use HTTPS)
- Cluster 3 Legitimate sites(Branded Domains)
## Results
- Supervised Learning Performance
- Dataset 4

## Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|----------------------|-----------|------------|---------|-----------|----------|
| **XGBoost**          | 99.96%   | 99.94%     | 99.99% | 99.96%    | 0.9998   |
| **Random Forest**    | 99.91%   | 99.87%     | 99.97% | 99.92%    | 0.9995   |
| **Logistic Regression** | 99.81% | 99.81%     | 99.86% | 99.84%    | 0.9991   |
| **SVM**              | 99.19%   | 99.30%     | 99.28% | 99.29%    | 0.9980   |

- Best model XGBoost
- Cross validation results
- XGBoost 99.95 + 0.01%
- Random Forest 99.90 + 0.02%
- Logistic Regression 99.80+ 0.03%
- SVM 99.15+0.05%
## Conclusion
- In this assignment I developed a machine learning pipeline for phishing detection achieving
  99.96 F1 score (XGBoost on 235k Samples)
- 8 Phishing Features
- 4 algorithms compared
- Robust Preprocessing
- Comprehensive evaluations
- Unsupervised Insights(Accuracy,Precision,Recall,F1,AUC,cross-validation)
- Automated Reporting(Saves Models, Generate visualizations,produces reports)