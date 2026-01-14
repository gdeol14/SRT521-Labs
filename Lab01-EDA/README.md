# Lab01-EDA - Exploratory Data Analysis on a phishing dataset to understand the data structure,quality

- **Student Name:** Gurmandeep Deol
- **Student ID:** 104120233
- **Lab Number:** Lab01
- **Completion Date:** 2025-09-11
- **Course:** SRT521 - Advanced Data Analysis for Security
- **Course Section Number:** (NBB)

## Lab Objectives
- The main learning objectives are
    - Load and explore datasets
    - Perform basic exploratory data analysis [EDA]
    - Create visualizations to understand data patterns
    - document your findings and insights

## Dataset Information
- **Dataset Name:** [dataset1.csv]
- **Source:** [Kaggle https://www.kaggle.com/datasets/mdsultanulislamovi/phishing-website-detection-datasets]
- **Size:** [11005 rows, 32 columns, file size:5,101kb]
- **Domain:** [Phishing]

## Key Findings
- The dataset is clean there are no missing values, no duplicates, no infinite values
My target value is results based on that I got 6,135 records of phishing (55.7%) and 4,870 records of Legitimate (44.3%) so there was more phishing than non-phishing The dataset is balanced with an imbalance of 1.3:1 which is acceptable as there is some imbalance expected when going through data The class balance means that there are more phishing attacks but it is well balanced.

## Technical Implementation
- **Libraries:** Python libraries used: pandas, numpy, matplotlib.pyplot, seaborn, plotly.express, plotly.graph_objects, warning
- **Preprocessing Steps:** Removed index column stripped extra spaces of column names made sure there were no missing values or duplicates or infinite values created a result label to differentiate between legitimate and phishing

## Challenges and Solutions
- Some technical challenges I incurred were when I tried to upload the dataset1.csv which was to be used I kept getting a file not found error even though the file was on my desktop another challenge I had was that sometimes the code would not run properly in jupyter I kept having kernel problems
- The first challenge I solved it by putting my dataset1.csv in the same folder as where the notebook was that way it was able to find the dataset and I was able to proceed with the lab For the second challenge I switched over to Google Colab which was much better than jupyter notebook

## Reflection
- I learned how to use EDA on the phishing cybersecurity dataset I also learned how to identify class balance issues to know which one is more is it phishing or legitimate.
- This relates to security application because some of the dataset will be very messy and dirty and I will need to make sure that the dataset is clean the quality of the data is good and I can understand all the features in the data possibly even use machine learning for the dataset.
- I would try to apply machine learning for the dataset so as it gets more and more datasets the more accurate it becomes at cleaning the data identifying features.

## Files Description
- `lab-01-eda.ipynb` - Main lab notebook with analysis
- `outputs/` - Generated plots, model files, results
- `data/` - Dataset files (if applicable)
- `images/` - Screenshots and diagrams