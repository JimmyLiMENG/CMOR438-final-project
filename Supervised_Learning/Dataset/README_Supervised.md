## Supervised Learning 

# 1_Explore_Understand_Data
This folder contains our first step in the supervised‐learning pipeline: a thorough exploratory data analysis of the Telco Customer Churn dataset. Using a Jupyter notebook (1_Explore_Understand_Data.ipynb), we:

- **Load** the dataset and inspect its shape, data types, and missing values  
- **Compute** summary statistics for numerical and categorical features
- **Visualize** distributions of key variables (e.g., tenure, MonthlyCharges) and the target variable (Churn)
- **Analyze** relationships and correlations to inform feature engineering and model selection  
- **Document** any data‐quality issues (e.g., 11 missing TotalCharges entries) and how they were handled

---
**Dataset Summary**

**File:** telco_churn.csv
**Description:** Customer account and service usage records for a telecommunications company, with the goal of predicting churn.  

| Feature            | Type        | Notes                                           |
|--------------------|-------------|-------------------------------------------------|
| `customerID`       | string      | Unique customer identifier                      |
| `gender`           | categorical | Male, Female                                    |
| `SeniorCitizen`    | integer     | 1 = yes, 0 = no                                 |
| `Partner`          | categorical | Yes, No                                         |
| `Dependents`       | categorical | Yes, No                                         |
| `tenure`           | integer     | Months with the company                         |
| `PhoneService`     | categorical | Yes, No                                         |
| `MultipleLines`    | categorical | Yes, No, No phone service                       |
| `InternetService`  | categorical | DSL, Fiber optic, No                            |
| `OnlineSecurity`   | categorical | Yes, No, No internet service                    |
| `OnlineBackup`     | categorical | Yes, No, No internet service                    |
| `DeviceProtection` | categorical | Yes, No, No internet service                    |
| `TechSupport`      | categorical | Yes, No, No internet service                    |
| `StreamingTV`      | categorical | Yes, No, No internet service                    |
| `StreamingMovies`  | categorical | Yes, No, No internet service                    |
| `Contract`         | categorical | Month-to-month, One year, Two year              |
| `PaperlessBilling` | categorical | Yes, No                                         |
| `PaymentMethod`    | categorical | Electronic check, Mailed check, Bank transfer…  |
| `MonthlyCharges`   | numeric     | Dollar amount                                   |
| `TotalCharges`     | numeric     | Dollar amount (11 missing → imputed/dropped)    |
| **`Churn`**        | categorical | **Target**: Yes, No                             |

Total records: **7,043**  
Total features: **21 (20 predictors + 1 target)**

-------------------------------------------------------------------

# 2_Data_Cleaning_Preprocessing

This folder contains all the code and data needed to clean and preprocess the Telco Customer Churn dataset prior to supervised learning model development.

**Algorithms & Steps Implemented**

1. **Data Ingestion**  
   - Load the raw CSV (telco_churn.csv) into a pandas DataFrame.  

2. **Initial Cleanup**  
   - Drop the customerID column and any exact duplicates.  
   - Convert TotalCharges from string to numeric, coercing errors to NaN.  

3. **Missing-Value Imputation**
   - Identify missing or blank TotalCharges entries (originally empty strings).  
   - Drop rows with missing TotalCharges  

4. **Type Conversion & Scaling**  
   - Ensure all numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are of numeric dtype.

5. **Categorical Encoding**  
   - **Binary features** (e.g. `gender`, `Partner`, `Churn`, etc.): map “Yes”/“No” to 1/0.  
   - **Multi-class features** (e.g. `InternetService`, `Contract`, etc.): apply one-hot encoding via `pd.get_dummies()`.  
---

**Dataset Summary**

| File                       | Rows  | Columns | Description                                       |
|----------------------------|-------|---------|---------------------------------------------------|
| **telco_churn.csv**        | 7,043 | 21      | Raw customer records with demographic, services, and churn flag.
| **telco_churn_clean.csv**  | 7,032 | 31      | Cleaned & feature-engineered data, ready for modeling.

- **Key features**  
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`  
  - Binary categorical: `gender`, `Partner`, `Dependents`, `PaperlessBilling`, `Churn`  
  - One-hot encoded: Internet, Phone, Contract, PaymentMethod, and other service options  

- **Target variable:**  
  - `Churn` → 1 if the customer churned, 0 otherwise.  

-------------------------------------------------------------------

# 3.1_Supervised_Learning_Perceptron

**Algorithm**
This notebook implements the Perceptron algorithm for binary classification. The Perceptron is a linear classifier that learns a set of weights via iterative updates: for each misclassified example it adjusts its weight vector in the direction of the error. This simple algorithm serves as the foundation for more advanced neural network models.

**Dataset**
We use the Telco Customer Churn dataset in its cleaned form (telco_churn_clean.csv), which contains 7,032 customer records and 31 columns. The features and the target are stated under **2_Data_Cleaning_Preprocessing**.

This cleaned file omits any missing values or raw identifiers for reproducibility.

**Insights**
1. Customers on longer contracts and with more months of tenure are far less likely to leave.

2. OnlineSecurity and TechSupport both show strong protective effects. Upselling these services may meaningfully reduce churn.

3. Electronic-check payers churn at a higher rate—targeted retention offers (e.g. discounts) could be prioritized here.


-------------------------------------------------------------------

# 3.2_Supervised_Learning_Linear_Regression

**Algorithm**
In this notebook, I implement ordinary least squares linear regression to predict customer tenure (in months) using a set of numeric and one-hot encoded categorical features from the Telco churn dataset. We use scikit-learn’s LinearRegression model, training on an 80/20 train/test split. Model performance is evaluated via R^2 and RMSE on the hold-out test set.

**Dataset**
We use telco_churn_clean.csv, a cleaned subset of the original Telco Customer Churn data. Details of the dataset is stated above.

**Insights**
1. The test R^2 of 0.2522 shows that only about 25% of the variability is captured by this linear model.

2. A test MSE of 0.1459 indicates the model reduces error somewhat but still leaves substantial residuals.

3. The Actual vs Predicted scatter reveals:
   - True non-churn (0) points fall below ~0.2 more than churn (1).
   - True churn (1) points lie above ~0.8 more than non-churn (0).
   - However, a broad overlap between ~0.2–0.8 shows the linear fit cannot cleanly separate the two classes.

-------------------------------------------------------------------

# 3.3_Supervised_Learning_Logistic_Regression

**Algorithm**  
We implement Logistic Regression, a linear classification model that estimates the probability of a binary outcome (customer churn) by fitting input features to a sigmoid‐transformed weighted sum. Logistic Regression is well suited to interpret feature contributions and to produce calibrated class probabilities.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Good overall discrimination but imbalanced performance.
   - Accuracy: 0.79  
   - ROC–AUC: 0.83  
   The model ranks churners vs. non-churners reasonably well (AUC ≈0.83), yet applying the default 0.5 threshold yields uneven recall across classes.

2. Non-churn (class 0) is well-captured.
   - Precision (0): 0.84  
   - Recall (0): 0.89  
   Out of 1300 true non-churners, 1156 are correctly identified (only 144 false positives), giving high confidence in “stay” predictions.

3. Churners (class 1) are under-detected. 
   - Precision (1): 0.62  
   - Recall (1): 0.50  
   Only 231 of 458 actual churners are caught, meaning half of the customers who leave slip through undetected (227 false negatives).

-------------------------------------------------------------------

# 3.4_Supervised_Learning_Neural_Networks

**Algorithm** 
In this notebook we build a feed-forward neural network in Keras/TensorFlow to classify customer churn. The model consists of two hidden layers with ReLU activations, a final sigmoid output neuron for binary classification, and is trained with the Adam optimizer and binary cross-entropy loss.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Fast convergence with slight over-fitting
   - The network quickly jumps from ~0.75 to ~0.80 train accuracy in the first 2–3 epochs, then slowly climbs to ~0.83 by epoch 50.  
   - Validation accuracy peaks around ~0.79 early (epochs 5–15) and then drifts downward, while validation loss bottoms out near epoch 5 before rising—classic signs that the model starts to overfit after ~10 epochs.

2. Strong “stay” detection, weak “churn” recall 
   - Non-churn (class 0): precision = 0.83, recall = 0.90 → 929/1033 stays correctly identified.  
   - Churn (class 1): precision = 0.63, recall = 0.48 → only 179/374 churners caught (195 false negatives).  

3. Overall balanced accuracy (~0.79)  
   - The overall accuracy is ~79%, and weighted-avg F1 is ~0.78, showing the model generalizes reasonably.

4. Opportunity to boost churn recall
   - Introduce early stopping around epoch 5–10 to avoid overfitting.  
   - Apply oversampling to penalize missed churners more heavily.  

-------------------------------------------------------------------

# 3.5_Supervised_Learning_K_Nearest_Neighbors

**Algorithm** 
We implement a K-Nearest Neighbors classifier to predict churn by first standardizing all numeric features so distances are comparable, then using Euclidean distance to find the k closest training examples and assigning labels by majority vote (either uniformly or weighted by inverse distance). We tune both k and the voting scheme via stratified grid search with cross-validation, select the best model, fit it on the full training set, and evaluate its performance on the test set using accuracy, precision/recall/F1.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Optimal smoothing at k = 19 
   The cross‐validation curve rises steadily from ~0.71 at k = 1 to ~0.787 at k = 19, indicating that larger neighborhoods reduce variance without over‐smoothing the decision boundary.  

2. Strong generalization
   The hold‐out test accuracy (78.04%) closely matches the CV peak (78.7%), suggesting the chosen k generalizes well and the model is not over‐fitting.  

3. Reliable “stay” predictions  
   - Precision (no‐churn): 0.85  
   - Recall (no‐churn): 0.86  
   Out of 1291 non‐churners, 1105 are correctly identified, with only 186 false positives—KNN did good at recognizing the majority class.  

4. Under‐detection of churners 
   - Precision (churn): 0.59  
   - Recall (churn): 0.57  
   Only 267 of 467 true churners are caught, leaving 200 false negatives. The high‐dimensional distance metric struggles to isolate minority‐class patterns.  


-------------------------------------------------------------------

# 3.6_Supervised_Learning_Decision_Trees_Regression_Trees

**Algorithm** 
In this notebook we implement two fundamental tree-based supervised learning methods from scikit-learn. First, we train a Decision Tree Classifier on the Telco Customer Churn dataset to predict customer churn status, visualizing the resulting tree and evaluating its performance through accuracy, precision, recall, and F1 metrics. Next, we fit a Decision Tree Regressor to model and predict each customer’s MonthlyCharges using tenure, demographic, and service usage features, assessing predictive accuracy via mean squared error and R^2 on a held-out test set. Both workflows include data splitting, hyperparameter choices, model visualization, and performance reporting to illustrate how tree depth and splitting criteria affect overfitting and generalization.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Moderate overall classification accuracy  
   The Decision Tree classifier achieves an accuracy of 0.7782.

2. Reliable “stay” predictions
   - Precision (no-churn): 0.83 
   - Recall (no-churn): 0.88 
   Out of 1291 non-churners, 1133 are correctly identified, with only 158 false positives—showing the tree did good at spotting customers who remain.

3. Under-detection of churners  
   - Precision (churn): 0.60
   - Recall (churn): 0.50 
   Only 235 of 467 true churners are caught, leaving 232 false negatives. The model struggles to isolate the minority churn class.

4. Key drivers of churn
   - tenure (importance ≈ 0.52)  
   - MonthlyCharges (importance ≈ 0.38)  
   Together these two features account for ~90% of split decisions, indicating that long-standing, lower-billed customers are far less likely to churn.

5. Regression tree reveals data leakage  
   - MSE: 0.64  
   - R^2: 0.9993 
   The regressor splits exclusively on the target MonthlyCharges (importance = 1.0), effectively “predicting” what it already sees.

-------------------------------------------------------------------

# 3.7_Supervised_Learning_Random_Forests

**Algorithm** 
This notebook applies a Random Forest classifier to the Telco Customer Churn dataset: we first train a baseline model with default settings, then perform a grid search to tune key hyperparameters (number of trees, maximum depth, minimum samples per split), and finally evaluate the best estimator using accuracy, ROC-AUC, and a confusion matrix—complemented by a feature-importance analysis that surfaces the strongest predictors of churn.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Strong overall discrimination  
   The baseline Random Forest achieves 78.61% accuracy and AUC = 0.817, improving to 79.47% accuracy and AUC = 0.836 after grid-search tuning.

2. Reliable “stay” detection  
   - Precision (no-churn): 0.83
   - Recall (no-churn): 0.89  
   Out of 1291 non-churners, 1152 are correctly identified, with only 139 false positives.

3. Under-detection of churners  
   - Precision (churn): 0.62
   - Recall (churn): 0.49 
   Only 230 of 467 true churners are caught, leaving 237 false negatives.

4. Key drivers of churn  
   - TotalCharges (~0.20)  
   - tenure (~0.18)  
   - MonthlyCharges (~0.17)  
   Together these three account for over 55% of the model’s split decisions; secondary factors include electronic-check payments and fiber-optic service.


-------------------------------------------------------------------

# 3.8_Supervised_Learning_Boosting

**Algorithm** 
This notebook uses scikit-learn’s GradientBoostingClassifier to predict customer churn by sequentially fitting an ensemble of 200 shallow decision trees (each of maximum depth 4) to the residual errors of the previous iteration, with a learning rate of 0.05 and a fixed random_state for reproducibility. By minimizing the binomial deviance at each stage, the model builds an additive prediction function that is evaluated on a stratified 75/25 train-test split. Performance is quantified via accuracy, precision/recall/F1, confusion matrix, and ROC-AUC, and the fitted model’s feature importances are plotted to highlight the strongest drivers of churn.

**Dataset**
File: telco_churn_clean.csv

**Insights**
1. Strong overall discrimination  
   The Gradient Boosting classifier reaches 79% accuracy on the hold-out set with a ROC AUC = 0.84, edging out the baseline Random Forest.

2. Reliable “stay” predictions  
   - Precision (no-churn): 0.84
   - Recall (no-churn): 0.89
   Of 1291 retained customers, 1151 are correctly identified, with only 140 false alarms.

3. Moderate churn detection  
   - Precision (churn): 0.64
   - Recall (churn): 0.52 
   Only 245 of 467 true churners are caught, leaving 222 false negatives.

4. Dominant risk factors  
   - tenure (~30% importance): shorter-tenured customers are far more likely to churn  
   - InternetService_Fiber optic (~20%): fiber users churn at higher rates  
   - TotalCharges (~11%) and MonthlyCharges (~9%): larger bills correlate with churn

5. Contract & payment effects  
   - Electronic check (~7%): customers paying by e-check churn more  
   - Contract_Two year (~5%) and Contract_One year (~4%) reduce churn risk


