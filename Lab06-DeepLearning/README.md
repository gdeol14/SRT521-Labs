# Lab06-DeepLearning Part 1 - Advanced Model Analysis & Persistence
**Student Name:** Gurmandeep Deol  
**Student ID:** 104120233  
**Lab Number:** Lab06  
**Completion Date:** 2025-10-19  
**Course:** SRT521 - Advanced Data Analysis for Security  
**Course Section Number:** NBB
## Lab Objectives
- **Perform comprehensive error analysis with detailed visualizations**
- **Implement cross-validation for both supervised and - unsupervised learning**
- **Fine-tune hyperparameters using grid search and random search**
- **Save and load models using pickle files**
- **Compare model performance across different configurations**
- **Apply advanced evaluation techniques to security datasets**
## Dataset Information
- **Dataset Name:** **engineered_dataset4.csv**
- **Source:** **Lab 3 Output Dataset**
- **Size:** **235,795 rows 68 original columns 52 features after preprocessing 77.0 MB**
- **Domain:** **Phishing**
- **Class distribution**
- **Phishing: 134850 (57.2%)**
- **Legitimate: 100945 (42.8%)**
## Key Findings
- **Best Model: XGBoost F1 Score: 99.98% Recall: 100.00 Precision: 99.96 Accuracy: 99.97% Errors: only 14 mistakes**
- **I successfully identified and removed 10 leakage features such as URLSimilarityIndex,Security_Risk_Score**
- **False Positives occurred mainly on legitimate sites while false negatives were on phishing attacks**
![false positives and negatives](outputs/false%20postives%20and%20negatives.png)
![error](outputs/error%20analysis%20models.png)
![Cross Validation Results](outputs/cross%20validation%20results.png)
---
## Technical Implementation
- **Algorithms Used:** **Random Forest, Logistic Regression, XGBoost Classifier, K-Means Clustering DBSCAN, Isolation Forest Grid Search Cv Random Search CV, Stratified K-Fold Cross-Validation** 
- **Libraries:** **pandas,numpy,matplotlib,matplotlib,seaborn scikit-learn,xgboost, pickle, datetime, os** 
- **Preprocessing Steps:** **Data Leakage Removal removed 10 engineered features eliminated identifier columns filename,url,domain title,tld Started with 68 columns removed 16 columns after processing had 52 predictive features 80/20 train-test split with stratification ensured class distribution maintained in both sets there were no missing values.**
- **Model Performance:** 
- **Training Random Forest...**
  - **Accuracy:  0.9992**
  - **Precision: 0.9989**
  - **Recall:    0.9998**
  - **F1 Score:  0.9993**

- **Training Logistic Regression...**
  - **Accuracy:  0.9976**
  - **Precision: 0.9983**
  - **Recall:    0.9975**
  - **F1 Score:  0.9979**

- **Training XGBoost...**
  - **Accuracy:  0.9996**
  - **Precision: 0.9995**
  - **Recall:    0.9998**
  - **F1 Score:  0.9997**
---
## Challenges and Solutions
- **Initially models showed unrealistic performance suggesting data leakage**
- **I removed features and made sure model performance dropped down to the 99 range instead of 100** 
- **DBSCAN created 1,304 micro-clusters with 73% noise**
- **I fixed this by setting eps to 3.0 and min_samples to 50** 
## Reflection
- **I learned that understanding where models fail is very important such as how many false positives and negatives they have as this way you can determine the best model and in security having false negatives is more dangerous because those are threats that the models missed potentially attacking the user also removing some features was necessary to ensure the model is able to detect phishing and legitimate.**
- **This relates to security because these pkl files are saved models and pretty small so you can run them on edge devices such as routers switches mobile apps etc.** 
- **I would not do anything differently as I feel I have made the models tested them I feel I did everything**

## Files Description
- `lab-06-deep-learning.ipynb` - Main lab notebook with analysis
- `outputs/` - Generated plots, model files, results
- `data/` - Dataset files (if applicable)
