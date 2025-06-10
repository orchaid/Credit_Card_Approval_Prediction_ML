


# Credit Card Approval Prediction

## Overview
This project aims to build a machine learning model to predict whether a credit card client is a "good" or "bad" credit risk. It uses two datasets:
- One with client demographic and financial information
- Another with detailed monthly credit history for each client

By engineering custom labels from credit behavior over time and building aggregate features, the project simulates a real-world credit scoring pipeline from raw data to prediction.


## Objective
- **Business Goal:** Help financial institutions identify high-risk applicants before approving credit cards.
- **Analytical Goal:** Predict client risk (good vs. bad) using ML models trained on time-series credit behavior and static client features.


## Business Impact
This model could be deployed in a pre-approval stage to flag high-risk applicants, helping reduce default rates and improve portfolio quality.


## Data Sources
1. **application_record.csv**  
   Contains demographic and economic details for ~43,000 clients.

2. **credit_record.csv**  
   Contains ~1 million records of monthly credit behavior (status, delay, date) for each client.


## Label Engineering
Since no target variable was provided, a custom binary label was created:
- A client is labeled **bad (0)** if they had a **DPD (Days Past Due) of 60+ days** (`status` of 2 or higher) for **3 consecutive months**.
- Otherwise, the client is labeled **good (0)**.

This mimics real-world *vintage analysis* used by banks to evaluate creditworthiness over time.


## Data Processing
- Merged time-series credit data into client-level features.
- Cleaned outliers and irrelevant features from client data.
- Merged with the application dataset to get a full feature set per client.
- Derived the target variable from the `status` variable.



## Imbalanced Data Handling
- After labeling, only ~8% of clients were labeled as "bad."
- Used techniques like:
  - Resampling (oversampling \ SMOTE)
  - Model tuning for imbalanced datasets
  - Focusing on metrics like F1-score, Precision, and Recall (not just Accuracy)



## Modeling
Tested several classifiers:
- KNeighborsClassifier
- Logistic Regression
- Decision Tree 
- Random Forest
- XGBoost
- SVC

The best-performing model was **K-Nearest Neighbors (KNN)**, evaluated using a classification report: ( F1-score: **0.97** - Precision: **0.23** - Recall: **0.43** )  
- Accuracy (97%) can be misleading due to imbalance, predicting most clients as "good" still gives a high accuracy.
and thats applies to all the models tested due to class imbalance as bad clients only represent ~1.6% of the data, the majority of the dataset belongs to class 1 (good clients)

Performance was evaluated on the test set, with special focus on identifying high-risk clients accurately.

Then now what really is the best model?
- If your goal is overall balanced performance, especially treating both classes fairly → KNN is better (higher macro & weighted F1).

- If your goal is catching as many of the rare cases (bad clients) as possible (i.e., prioritize recall for Class 0), SVC with SMOTE is better (recall = 0.79, but precision is very poor).

### Verdict
- KNN with ADASYN is the better model overall because it offers a significantly better balance across precision, recall, and F1-score for both classes — especially when you consider macro and weighted averages.

But:

- SVC with SMOTE might be preferable only if minimizing false negatives in Class 0 is your top priority (e.g., wrongly approving bad credit customers).
