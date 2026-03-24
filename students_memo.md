# Data Science Class Memo: Machine Learning Models & Evaluation

Welcome to our session on Machine Learning Models! This memo serves as a quick reference guide to the algorithms we covered, how to interpret their performance, and some exciting challenges for you to tackle next.

---

## 🧠 Model Characteristics Overview

### 1. Logistic Regression
* **Type:** Classification
* **How it works:** A foundational linear model that predicts the probability of an outcome using a logistic (sigmoid) curve. 
* **Best for:** Baseline models, binary classification, and when you need high interpretability (you can easily see how much each feature impacts the prediction).

### 2. Support Vector Machines (SVC / SVR)
* **Type:** Classification & Regression
* **How it works:** Finds the optimal "hyperplane" (a line or boundary in high-dimensional space) that best separates classes or fits continuous data within a defined margin of error.
* **Best for:** Complex, high-dimensional spaces where there is a clear margin of separation.

### 3. Random Forest (Classifier / Regressor)
* **Type:** Ensemble (Bagging)
* **How it works:** Builds dozens (or hundreds) of independent Decision Trees on random subsets of your data, then averages their predictions (or takes a majority vote). 
* **Best for:** Reducing overfitting (high variance) commonly seen in single decision trees. Highly robust out-of-the-box.

### 4. Gradient Boosting (GBC, GBR, & XGBoost)
* **Type:** Ensemble (Boosting)
* **How it works:** Builds trees *sequentially*. Each new tree is specifically designed to correct the errors (residuals) made by the previous trees. XGBoost is a highly optimized, lightning-fast implementation of this concept.
* **Best for:** Achieving state-of-the-art predictive accuracy. Prone to overfitting if not tuned correctly.

---

## 📊 Metrics & Interpretation

### Classification Metrics
When predicting categories (e.g., "Will this customer buy a BMW?"), accuracy isn't everything—especially if the classes are imbalanced!

* **Accuracy:** Overall percentage of correct predictions. (Dangerous to rely on if 99% of your data is one class!)
* **Precision:** "Of all the times the model predicted 'Positive', how many were actually 'Positive'?" 
* **Recall (Sensitivity):** "Of all the actual 'Positive' cases in the data, how many did the model find?"
* **The Precision vs. Recall Trade-off:** 
  * If you want to be *absolutely sure* before predicting positive, you increase **Precision** (but you might miss some real cases, lowering Recall). 
  * If you want to catch *every possible* positive case, you increase **Recall** (but you will likely guess positive on some negative cases, lowering Precision).
* **F1-Score:** The harmonic mean of Precision and Recall. Best used when you need a balance between the two.
* **AIC (Akaike Information Criterion):** Measures model quality while penalizing complexity. **Lower is better.** It helps answer: *"Is this tiny increase in accuracy worth adding 50 more features?"*

### Regression Metrics
When predicting continuous numbers (e.g., "What is the price of this BMW?"):

* **R² (R-Squared):** The proportion of variance in the target variable explained by the model. Closer to 1.0 is better. (e.g., 0.85 means the model explains 85% of the price variations).
* **MAE (Mean Absolute Error):** The average absolute distance between predictions and actual values. Very interpretable (e.g., "Our predictions are off by $2,000 on average").
* **MSE (Mean Squared Error):** Averages the *squares* of the errors. Heavily penalizes large mistakes (outliers).

---

## 🚀 Hands-On Challenges

Ready to experiment? Try these tasks in the notebook to deepen your understanding:

1. **Expand the Grid Search:**
   * Currently, the `GridSearchCV` is kept very small for class time (`n_estimators: [50, 100]`). Try adding more parameters! For Random Forest, add `min_samples_split: [2, 5, 10]` or `max_features: ['sqrt', 'log2']`. Observe how computation time increases alongside potential accuracy gains.

2. **Feature Engineering:**
   * Create new, combined features before the Train/Test split. For example, can you combine `Year` and current date to create a `Vehicle_Age` feature? Does passing this new feature to the model improve the $R^2$ score?

3. **Tweak the SMOTE Strategy:**
   * Our script applies SMOTE to completely balance the dataset. What happens if you use an under-sampling technique instead? Try importing `RandomUnderSampler` from `imblearn` and see how it impacts your Recall vs. Precision.

4. **Change the Feature Selection Threshold:**
   * We currently limit `SelectKBest` to a maximum of 10 features (`k=10`). Change this to `k=5` and `k=20`. Watch how the **AIC score** reacts. Does adding more features always improve the AIC, or does the penalty for complexity eventually outweigh the accuracy gains?