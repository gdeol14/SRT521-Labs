# Lab 04 Baseline Machine Learning Models - Phishing URL Detection

**Student Name:** Gurmandeep Deol  
**Student ID:** 104120233  
**Lab Number:** Lab04  
**Completion Date:** 2025-10-06  
**Course:** SRT521 - Advanced Data Analysis for Security  
**Course Section Number:** (NBB)

## Lab Objectives
- **Implement both supervised and unsupervised machine learning algorithms**
- **Train and evaluate baseline models on security datasets**
- **Perform clustering analysis and visualization**
- **Split data appropriately for training and testing**
- **Compare model performance with comprehensive evaluation metrics**
- **Visualize results and interpret model performance**

## Dataset Information
- **Dataset Name:** engineered_dataset4.csv
- **Source:** Lab 3 Feature Engineering Dataset Output
- **Size:** 235,795 samples 68 columns 77.0 MB
- **Domain:** Phishing

## Key Findings
- **Random Forest and XGBoost achieved 100% accuracy on the test set overall all of my models had high performance with Logistic Regression up to 99.99 percent and SVM 99.77%** 
- **The most important features were URLSimilarityIndex, Content_URL_Ratio, NoOfExternalRef these were top predictors** 
- **K-means identified 3 optimal clusters with a 0.194 silhouette score and DBSCAN found 7 clusters with 10.5% noise** 
- **The features I had engineered from lab 3 proved to be highly successful for phishing detection**
![Model Accuracy](outputs/Model%20Accuracy.png)
## Technical Implementation
- **Algorithms Used:** **Supervised Random Forest, XGBoost, Logistic Regression, SVM Unsupervised: K-means, DBSCAN**
- **Libraries:** **scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, standardscaler, labelencoder, pca**
- **Preprocessing Steps:** **Removed identifier columns such as FILENAME,URL,DOMAIN,TITLE Label encoded categorical features such as TLD filled in missing values with median split the data into 80-20 train-test split**
- **Model Performance:** **Random Forest 100% accuracy XGBoost 100% accuracy Logistic Regression 99.99% accuracy SVM 99.77% accuracy**

## Challenges and Solutions
- **When I initially saw the 100% accuracy I had some concerns about data leakage so the solution was to remove identifier columns such as FILENAME,URL,Domain,Title which could leak information** 
- **DBSCAN with the default parameters eps=0.5 classified 98.1% as noise tested multiple eps values and selected eps=5 reducing noise to 10.5%**
- **Large Dataset slowed SVM training implemented random sampling of up to 5000 samples for svm while maintaining full dataset for other models**

## Reflection
- **I learned that data preprocessing is essential such as removing identifiers which solves data leakage and how high accuracy can be legitimate if the features are really useful such as TLD,URL patterns and that different clustering algorithm reveal different patterns.**
- **These baselines models can actually serve as real time phishing filters for browser or email providers and 100% accuracy means that phishing urls have strong detectable patterns making automated detection possible** 
- **Test on real world data and train more models to see their performances**

## Files Description
- `lab-04-baseline-ml.ipynb` - Main lab notebook with analysis
- `outputs/` - Generated plots, model files, results
- `images/` - Screenshots and diagrams