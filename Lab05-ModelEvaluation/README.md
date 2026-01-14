# Lab05-ModelEvaluation - Phishing Detection System
**Student Name:** Gurmandeep Deol  
**Student ID:** 104120233  
**Lab Number:** Lab 05  
**Completion Date:** 2025-10-12  
**Course:** SRT521 - Advanced Data Analysis for Security  
**Course Section Number:** NBB

## Lab Objectives
- Calculate and interpret key evaluation metrics for security problems
- Create and analyze confusion matrices with comprehensive visualizations
- Plot ROC curves and calculate AUC scores
- Perform detailed error analysis
- Evaluate multi-class classification models
- Apply evaluation techniques to unsupervised learning results
- Compare model performance using multiple metrics
- Identify and explain model limitations
## Dataset Information
- **Dataset Name:** engineered_dataset4.csv
- **Source:** Lab 3 Feature Engineering Dataset Output
- **Size:** 235,795 samples 68 columns 77.0 MB
- **Domain:** Phishing
- **target** 0 = legitimate 1 = phishing
- **class balance** 57.1 phishing 42.8 legitimate
![class balance](images/class%20balance.png)
## Key Findings
- Achieved random forest model achieved 100% accuracy
- Logistic Regression achieved 99.99% accuracy
both models also got perfect ROC-AUC scores of 1.000 showing they can actually tell phishing from legitimate sites
![model performance](images/model%20performance.png)
- the confusion matrix shows 20,189 legitimate sites and 26,970 phishing sites all classified correctly zero false negatives and zero false positives means no phishing sites were missed and no legitimate sites got caught as spam.
![confusion matrix](outputs/Model%20Accuracy.png)
- K-means clustering got a silhouette score of 0.9957 which means the data naturally forms tight clusters but the ari was only 0.0005 which means those clusters do not match with phishing vs legitimate labels this means we need supervised learning to detect phishing because unsupervised does not do the job based on the data from unsupervised learning
## Technical Implementation
### **__Algorithms Used:__** 
- Random Forest classifier achieved 100% accuracy 
- Logistic Regression achieved 99.99% accuracy
- K-Means Clustering (k=2)
- DBSCAN (eps=0.5,min_samples=5)
### **__Libraries:__** 
- pandas,numpy Data Manipulation
- scikit-learn Model Training and Evaluation
- matplotlib,seaborn Visualization
- StandardScaler Feature Normalization
### **__Preprocessing Steps:__** 
- Removed non-numeric columns such as FILENAME,URL,Domain,TLD,Title
- did a train-test split (80/20)
- standard scaling for logistic regression
- After removing the non numeric columns the final features were 62 features
![final number of features](images/final%20number%20of%20features.png)
![Features](outputs/important%20features.png)
### **__Model Performance:__** 
- Random Forest Accuracy = 1.0000 Precision = 1.0000, Recall = 1.0000 F1-score = 1.0000
- Logistic Regression Accuracy = 0.9999 Precision = 0.9998 Recall = 1.00000 F1-score = 0.9999
![Model Performance](outputs/Model%20Accuracy.png) 
## Challenges and Solutions
- Dataset contained text columns such as Domain,TLD,Title causing issues
- Implemented automated non-numeric column detection and removal keeping only numeric feature for modeling
- Determining whether to skip multi class
- I identified that my dataset only has two classes therefore no need to do the multi class part since I only have two classes
- Initially when my model showed perfect results I was a little suspicious
- But I validated it through sampling properly training and split and using clustering analysis to show feature quality's
## Reflection
### **What did you learn from this lab?**
- I learned that evaluating a model requires multiple metrics to verify the model's accuracy
- I also learned I didn't need to do multi class because my dataset only has two classes one for phishing and one for legitimate I did not have a third class thus no need for multi class so I don't need to do everything in a lab some things don't apply 
### **How does this relate to security applications?**
- This relates to security application because in this lab since I had Zero false negatives this would mean all phishing attempts in the dataset were caught which is important for protecting users
- False negatives are more dangerous than false positives because missing a phishing site means that could get through to the user.
### **What would you do differently next time?**
- I would test the models on different datasets to see the precision of the models and I would explore other models and train my dataset on them
## Files Description
- `README.md` - readme file 
- `lab-05-model-evaluation.ipynb` - Main lab notebook with analysis
- `outputs/`- Generated plots, models, results
- `images/` - Screenshots and diagram