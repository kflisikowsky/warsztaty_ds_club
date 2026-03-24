#%%
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (StandardScaler, LabelEncoder)

from sklearn.metrics import (ConfusionMatrixDisplay, 
                            mean_squared_error,
                            mean_absolute_error,
                            confusion_matrix, 
                            accuracy_score,
                            classification_report,
                            log_loss)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestRegressor,
                            GradientBoostingRegressor, 
                            RandomForestClassifier, 
                            GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.svm import SVC, SVR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


#%% Data set
df_bmw_final=pd.read_csv('df_bmw_final.csv')
df_bmw_final.head()
#%% Data visualization
g = sns.relplot(
    data=df_bmw_final,
    x="Year",
    y="Price_USD",
    row="Region",
    col="Model",
    kind="line",
    height=3,
    aspect=1.2,
    marker="o",
    errorbar=None,
    facet_kws={'sharey': False}
)

g.set_axis_labels("Year", "Price (USD)")
g.set_titles(row_template="{row_name}", col_template="{col_name}")

# Rotate x-axis labels for better readability
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45)

plt.subplots_adjust(top=0.92)
g.fig.suptitle('BMW Price Trends by Region and Model', fontsize=16)

plt.show()

#%% Data to model preparation, diffs of data sets

data_to_model=df_bmw_final.select_dtypes(include=['int','float'])
data_to_model_with_dummies_k_minus1=pd.get_dummies(df_bmw_final, columns=['Region', 'Model', 'Color', 'Fuel_Type', 'Transmission', 'Sales_Classification'], 
                                    drop_first=True) #Linear Regression (with intercept), Interpretation Required
data_to_model_with_dummies_k_all=pd.get_dummies(df_bmw_final, columns=['Region', 'Model', 'Color', 'Fuel_Type', 'Transmission', 'Sales_Classification']
                                ) #Decision Trees / Ensemble, Distance-based (KNN)
#%%
print(df_bmw_final.columns)
print(data_to_model.columns)
print(data_to_model_with_dummies_k_minus1.columns)
print(data_to_model_with_dummies_k_all.columns)


#%% Correlation matrix

plt.figure(figsize=(18,16))
cor1 = data_to_model.corr()
sns.heatmap(cor1, annot=True, cmap=plt.cm.Reds)
plt.show()


#%% Path for data to model and feature slection

features = data_to_model_with_dummies_k_all.iloc[:,0:-1]
target = data_to_model_with_dummies_k_all.iloc[:,-1]

scaler = StandardScaler() # use for distance models: knn, kmeans, pca, no distribution knowledge; when significant outliers min-max
features = scaler.fit_transform(features) # features only!!!

# Partition the dataset into training and validation sets to objectively evaluate model generalization on unseen data
X_train, X_valid, y_train, y_valid = train_test_split(
    features, target, test_size=0.2, random_state=42)
print("Original shapes:", X_train.shape, X_valid.shape)

#%% AIC Calculation Functions (Educational Proxy)
# AIC balances model fit against complexity. Lower AIC means a better model.
def calculate_aic_class(y_true, y_prob, k):
    n = len(y_true)
    ll = log_loss(y_true, y_prob)
    return 2 * n * ll + 2 * k

def calculate_aic_reg(y_true, y_pred, k):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    return n * np.log(mse) + 2 * k

# List to store classification metrics
class_metrics = []

print("\n--- Base Model (No SMOTE) ---")
rfc_no_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rfc_no_smote.fit(X_train, y_train)
y_pred_no_smote = rfc_no_smote.predict(X_valid)
y_prob_no_smote = rfc_no_smote.predict_proba(X_valid)

acc_no_smote = accuracy_score(y_valid, y_pred_no_smote)
aic_no_smote = calculate_aic_class(y_valid, y_prob_no_smote, X_train.shape[1])
class_metrics.append({'Model': 'RF (No SMOTE)', 'Accuracy': acc_no_smote, 'AIC': aic_no_smote})
print("Validation Accuracy (No SMOTE):", acc_no_smote)
print("AIC (No SMOTE):", aic_no_smote)

#%% SMOTE and Feature Selection (Educational)
print("\n--- Applying SMOTE (Handling Imbalanced Data) ---")
# Generate synthetic samples for the minority class to prevent the algorithm from learning a bias towards the majority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Shape after SMOTE:", X_train_smote.shape)

print("\n--- Applying Feature Selection (SelectKBest) ---")
# Isolate the most statistically significant predictors (features) to improve interpretability and reduce the risk of overfitting
k_features = min(10, X_train_smote.shape[1])
selector = SelectKBest(score_func=f_classif, k=k_features)
X_train_fs = selector.fit_transform(X_train_smote, y_train_smote)
# Apply the exact same feature filtering to the validation set to prevent data leakage
X_valid_fs = selector.transform(X_valid)
print("Shape after Feature Selection:", X_train_fs.shape)

# Assigning to standard variables for the models below
X_train_cls, y_train_cls = X_train_fs, y_train_smote
X_valid_cls, y_valid_cls = X_valid_fs, y_valid


#%% Base model

# Initialize an ensemble of decision trees that uses bootstrap aggregating (bagging) to reduce variance and improve robustness
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model by allowing the decision trees to extract patterns from the selected features
rfc.fit(X_train_cls, y_train_cls)

# Generate predictions on the validation set to test how well the learned rules apply to new data
y_pred=rfc.predict(X_valid_cls)

# Evaluate performance by comparing predicted classes against the actual ground truth labels
acc_rf_smote = rfc.score(X_valid_cls,y_valid_cls)
y_prob_rf = rfc.predict_proba(X_valid_cls)
aic_rf_smote = calculate_aic_class(y_valid_cls, y_prob_rf, X_train_cls.shape[1])
class_metrics.append({'Model': 'RF (With SMOTE)', 'Accuracy': acc_rf_smote, 'AIC': aic_rf_smote})
print("Validation Accuracy:", acc_rf_smote) #accuracy
print("AIC:", aic_rf_smote)
print("Train Accuracy:", rfc.score(X_train_cls,y_train_cls)) # avoid such mistake

# Construct a confusion matrix to visualize the distribution of True/False Positives and Negatives across all classes
r = confusion_matrix(y_valid_cls, y_pred)
print("Confusion Matrix:\n", r)
disp=ConfusionMatrixDisplay(confusion_matrix=r,
                              display_labels=rfc.classes_)
disp.plot()

plt.title("Random Forest - Base Model")
plt.show()

print("Classification Report (Base Model):\n", classification_report(y_valid_cls, y_pred))

#%% feature importance
# Quantify the contribution of each feature to the model's decision-making process based on the decrease in Gini impurity
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

# Get names of selected features
selected_mask = selector.get_support()
all_feature_names = data_to_model_with_dummies_k_all.iloc[:,0:-1].columns
selected_feature_names = all_feature_names[selected_mask]

forest_importances = pd.Series(importances, index=selected_feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances (Selected Features)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

#%% Other Classification Models (Logistic Regression, XGBoost, GBC)
print("\n--- Logistic Regression Example ---")
# Fit a linear classifier that models the probability of a discrete outcome using a logistic (sigmoid) function
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cls, y_train_cls)
y_pred_lr = log_reg.predict(X_valid_cls)
y_prob_lr = log_reg.predict_proba(X_valid_cls)
acc_lr = accuracy_score(y_valid_cls, y_pred_lr)
aic_lr = calculate_aic_class(y_valid_cls, y_prob_lr, X_train_cls.shape[1])
class_metrics.append({'Model': 'Logistic Regression (SMOTE)', 'Accuracy': acc_lr, 'AIC': aic_lr})
print("Logistic Regression Accuracy:", acc_lr)
print("Logistic Regression AIC:", aic_lr)
print(classification_report(y_valid_cls, y_pred_lr))

print("\n--- XGBClassifier Example ---")
# Encode categorical target labels into numerical values (0, 1, 2...) as required by XGBoost's mathematical implementation
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train_cls)
y_valid_xgb = le.transform(y_valid_cls)

# Initialize a gradient boosting framework that builds trees sequentially, minimizing the errors of prior trees
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_cls, y_train_xgb)
y_pred_xgb = xgb_model.predict(X_valid_cls)
y_prob_xgb = xgb_model.predict_proba(X_valid_cls)
acc_xgb = accuracy_score(y_valid_xgb, y_pred_xgb)
aic_xgb = calculate_aic_class(y_valid_xgb, y_prob_xgb, X_train_cls.shape[1])
class_metrics.append({'Model': 'XGBoost (SMOTE)', 'Accuracy': acc_xgb, 'AIC': aic_xgb})
print("XGBoost Accuracy:", acc_xgb)
print("XGBoost AIC:", aic_xgb)
print(classification_report(y_valid_xgb, y_pred_xgb))

print("\n--- GradientBoostingClassifier Example ---")
# Builds an additive model in a forward stage-wise fashion; allows for the optimization of arbitrary differentiable loss functions
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_cls, y_train_cls)
y_pred_gbc = gbc.predict(X_valid_cls)
y_prob_gbc = gbc.predict_proba(X_valid_cls)
acc_gbc = accuracy_score(y_valid_cls, y_pred_gbc)
aic_gbc = calculate_aic_class(y_valid_cls, y_prob_gbc, X_train_cls.shape[1])
class_metrics.append({'Model': 'Gradient Boosting (SMOTE)', 'Accuracy': acc_gbc, 'AIC': aic_gbc})
print("Gradient Boosting Accuracy:", acc_gbc)
print("Gradient Boosting AIC:", aic_gbc)
print(classification_report(y_valid_cls, y_pred_gbc))

print("\n--- Support Vector Classifier (SVC) Example ---")
# Finds the optimal hyperplane that maximizes the margin between different classes in a high-dimensional space
svc_model = SVC(random_state=42, probability=True)
svc_model.fit(X_train_cls, y_train_cls)
y_pred_svc = svc_model.predict(X_valid_cls)
y_prob_svc = svc_model.predict_proba(X_valid_cls)
acc_svc = accuracy_score(y_valid_cls, y_pred_svc)
aic_svc = calculate_aic_class(y_valid_cls, y_prob_svc, X_train_cls.shape[1])
class_metrics.append({'Model': 'SVC (SMOTE)', 'Accuracy': acc_svc, 'AIC': aic_svc})
print("SVC Accuracy:", acc_svc)
print("SVC AIC:", aic_svc)
print(classification_report(y_valid_cls, y_pred_svc))
#%% Simple GridSearchCV for Classification (Educational Example)
BEST_RFC_PARAMS_FILE = "best_rfc_params_simple.json" # Kept simple for fast computation

# Define a discrete hyperparameter space to explore during model tuning
rfc_params = {'n_estimators': [50, 100],
              'max_depth': [None, 5]}
best_params = None

# Check if best parameters file exists
if os.path.exists(BEST_RFC_PARAMS_FILE):
    with open(BEST_RFC_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded best parameters from {BEST_RFC_PARAMS_FILE}: {best_params}")
else:
    # Initialize RandomForestClassifier
    rfc_model = RandomForestClassifier(random_state=42)

    # Systematically search for the optimal combination of hyperparameters using cross-validation to prevent overfitting
    grid_search = GridSearchCV(estimator=rfc_model, param_grid=rfc_params, cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1)

    print("Starting GridSearchCV for Classification...")
    grid_search.fit(X_train_cls, y_train_cls)
    print("GridSearchCV complete.")

    best_params = grid_search.best_params_
    print(f"\nBest parameters found: {best_params}")

    # Save best parameters to file
    with open(BEST_RFC_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f)
    print(f"Saved best parameters to {BEST_RFC_PARAMS_FILE}")
#%% Evaluate Tuned Model
best_rfc_model = RandomForestClassifier(random_state=42, **best_params)
best_rfc_model.fit(X_train_cls, y_train_cls)
y_pred=best_rfc_model.predict(X_valid_cls)
y_prob_tuned = best_rfc_model.predict_proba(X_valid_cls)
acc_tuned_rf = best_rfc_model.score(X_valid_cls,y_valid_cls)
aic_tuned_rf = calculate_aic_class(y_valid_cls, y_prob_tuned, X_train_cls.shape[1])
class_metrics.append({'Model': 'Tuned RF (SMOTE)', 'Accuracy': acc_tuned_rf, 'AIC': aic_tuned_rf})
print("Tuned RFC Validation Accuracy:", acc_tuned_rf)
print("Tuned RFC AIC:", aic_tuned_rf)
print("Tuned RFC Train Accuracy:", best_rfc_model.score(X_train_cls,y_train_cls))

r = confusion_matrix(y_valid_cls, y_pred)
print("Confusion Matrix:\n", r)
disp=ConfusionMatrixDisplay(confusion_matrix=r, display_labels=best_rfc_model.classes_)
disp.plot()
plt.title("Random Forest - Tuned Model")
plt.show()

print("Classification Report (Tuned Model):\n", classification_report(y_valid_cls, y_pred))

#%% Classification Metrics Summary
print("\n--- Classification Models Comparison ---")
df_class_metrics = pd.DataFrame(class_metrics).sort_values(by='Accuracy', ascending=False)
print(df_class_metrics.to_string(index=False))

#%% Regression Problem Setup
print("\n--- Regression Problem Setup ---")
# For regression, we predict a continuous variable, such as 'Price_USD'
if 'Price_USD' in data_to_model_with_dummies_k_all.columns:
    # Isolate the continuous target variable and the independent predictors for the regression task
    reg_features = data_to_model_with_dummies_k_all.drop(columns=['Price_USD'])
    reg_target = data_to_model_with_dummies_k_all['Price_USD']
else:
    # Fallback if Price_USD is not the exact column name
    reg_features = data_to_model_with_dummies_k_all.iloc[:, 0:-1]
    reg_target = data_to_model_with_dummies_k_all.iloc[:, -1]

# Partition the dataset for the regression problem
X_train_r, X_valid_r, y_train_r, y_valid_r = train_test_split(
    reg_features, reg_target, test_size=0.2, random_state=42)

#%% Simple GridSearchCV for Regression (Educational Example)
print("\n--- GridSearchCV for Regression (RandomForestRegressor) ---")
# Kept simple with only 1 grid search for regression to avoid long computation
rfr_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

# Evaluate the regression model using Negative Mean Squared Error, as GridSearch attempts to maximize the scoring metric
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42), 
                        rfr_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

print("Starting GridSearchCV for Regression...")
rfr_grid.fit(X_train_r, y_train_r)
print("GridSearchCV complete.")

best_rfr = rfr_grid.best_estimator_
y_pred_r_grid = best_rfr.predict(X_valid_r)

# Dictionary to store regression metrics
reg_metrics = []

print("Best Regression Params:", rfr_grid.best_params_)
# Evaluate proportion of variance in the dependent variable that is predictable from the independent variables (R-squared)
r2_rfr = best_rfr.score(X_valid_r, y_valid_r)
mse_rfr = mean_squared_error(y_valid_r, y_pred_r_grid)
mae_rfr = mean_absolute_error(y_valid_r, y_pred_r_grid)
aic_rfr = calculate_aic_reg(y_valid_r, y_pred_r_grid, X_train_r.shape[1])

print("Train R^2 Score:", best_rfr.score(X_train_r, y_train_r))
print("Validation R^2 Score:", r2_rfr)
# Quantify the average magnitude of the prediction errors squared (MSE) and the absolute differences (MAE)
print("Validation MSE:", mse_rfr)
print("Validation MAE:", mae_rfr)
print("Validation AIC:", aic_rfr)

reg_metrics.append({'Model': 'Tuned RF Regressor', 'R2 Score': r2_rfr, 'MSE': mse_rfr, 'MAE': mae_rfr, 'AIC': aic_rfr})

print("\n--- GradientBoostingRegressor Example ---")
# Build an additive model in a forward stage-wise fashion for regression tasks
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_r, y_train_r)
y_pred_gbr = gbr.predict(X_valid_r)

r2_gbr = gbr.score(X_valid_r, y_valid_r)
mse_gbr = mean_squared_error(y_valid_r, y_pred_gbr)
mae_gbr = mean_absolute_error(y_valid_r, y_pred_gbr)
aic_gbr = calculate_aic_reg(y_valid_r, y_pred_gbr, X_train_r.shape[1])

print("Validation R^2 Score:", r2_gbr)
print("Validation MSE:", mse_gbr)
print("Validation MAE:", mae_gbr)
print("Validation AIC:", aic_gbr)

reg_metrics.append({'Model': 'Gradient Boosting Regressor', 'R2 Score': r2_gbr, 'MSE': mse_gbr, 'MAE': mae_gbr, 'AIC': aic_gbr})

print("\n--- Support Vector Regressor (SVR) Example ---")
# Fits the error within a certain threshold (epsilon) to find a hyperplane that best represents the continuous target variable
svr_model = SVR()
svr_model.fit(X_train_r, y_train_r)
y_pred_svr = svr_model.predict(X_valid_r)

r2_svr = svr_model.score(X_valid_r, y_valid_r)
mse_svr = mean_squared_error(y_valid_r, y_pred_svr)
mae_svr = mean_absolute_error(y_valid_r, y_pred_svr)
aic_svr = calculate_aic_reg(y_valid_r, y_pred_svr, X_train_r.shape[1])

print("Validation R^2 Score:", r2_svr)
print("Validation MSE:", mse_svr)
print("Validation MAE:", mae_svr)
print("Validation AIC:", aic_svr)

reg_metrics.append({'Model': 'Support Vector Regressor (SVR)', 'R2 Score': r2_svr, 'MSE': mse_svr, 'MAE': mae_svr, 'AIC': aic_svr})

#%% Regression Metrics Summary
print("\n--- Regression Models Comparison ---")
df_reg_metrics = pd.DataFrame(reg_metrics).sort_values(by='R2 Score', ascending=False)
print(df_reg_metrics.to_string(index=False))

# %%
