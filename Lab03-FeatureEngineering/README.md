# Lab03-FeatureEngineering - Phishing Website Detection

- **Student Name:** Gurmandeep Deol
- **Student ID:** 104120233
- **Lab Number:** Lab03
- **Completion Date:** 2025-09-27
- **Course:** SRT521 - Advanced Data Analysis for Security
- **Course Section Number:** (NBB)

## Lab Objectives
- Create domain-specific features aligned to your problem
- Validate features to avoid leakage and instability
- Perform simple feature selection
- Document engineering decisions

## Dataset Information
- **Dataset Name:** dataset4.csv
- **Source:** Kaggle https://www.kaggle.com/datasets/mdsultanulislamovi/phishing-website-detection-datasets
- **Size:** 235,795 Rows, 56 columns, 54.2mb
- **Domain:** Phishing

## Key Findings
- Successfully engineered 12 domain-specific features from the 56 original features
- URLSimilarityINdex was the top feature with a score of 671,857
- No data leakage was found across all features
- 11 out of 12 engineered feature were stable with a 20% drift for the last one
- Final feature set was reduced to 25 most informative features for modeling
- Dataset showed 57.2 phishing rate indicating that most of these were phishing attacks with a 42.8 legitimate rate
![Phishing Vs Legitimate](Output/Phishing%20Vs%20Legitimate.png)
![Engineered Features](Output/Enginerred%20Features.png)

## Technical Implementation
- **Algorithms Used:** ANOVA F-test, Random Forest, Pearson correlation analysis
- **Libraries:** pandas,numpy,matplotlib,seaborn,scikit-learn,scripy.stats
- **Preprocessing Steps:** Created composite scores for URL complexity,domain trust,content quality applied statistical validation and stability testing and normalized selection scores for a fair comparison
- **Model Performance:** Feature selection metrics indicate strong power with composite scoring methodology achieving balanced feature ranking
![Feature Selection](Output/Methods%20Of%20Selecting%20features.png)

## Challenges and Solutions
- The first challenge I faced was trying to determine the best way to make features with such a large dataset with 235k samples the second challenge was making sure that the features I chose would be features that would work across different type of phishing attacks making my model useful and more accurate
- I used different methods to engineer features to solve the first challenge and for the second one I used different type of validation metrics to ensure the features worked across all types of phishing attacks

## Reflection
- I learned the importance of domain knowledge in cybersecurity feature engineering. How to understand phishing attack patterns and based on that how to make meaningful features such as Domain_Trust_Score and Security_Risk_score

- This relates to real world security because feature quality determines how accurate the model will be in determining whether something is phishing or not if the model has good features to determine if something is phishing or not it will be more accurate?

- I would actually test this by giving it a dataset and see if it can use the features to detect if something is phishing or not based on all the features I engineered

## Files Description
- `lab-03-feature-engineering.ipynb` - Main lab notebook with analysis
- `outputs/` - Generated plots, model files, results
- `images/` - Screenshots and diagrams
- `README.md` - README Markdown Document