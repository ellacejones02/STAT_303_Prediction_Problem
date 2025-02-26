import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# load data  
# reading csv files, parsing date columns to manipulate them later  
train_X = pd.read_csv("train_X.csv", parse_dates=["ORDER_DATE", "PURCHASE_ORDER_DUE_DATE"])
train_y = pd.read_csv("train_y.csv")
test_X = pd.read_csv("public_private_X.csv", parse_dates=["ORDER_DATE", "PURCHASE_ORDER_DUE_DATE"])

# feature engineering  
# extracting useful datetime components + log transform to deal with skewed distribution  
for df in [train_X, test_X]:
    df["ORDER_YEAR"] = df["ORDER_DATE"].dt.year
    df["ORDER_MONTH"] = df["ORDER_DATE"].dt.month
    df["ORDER_DAY"] = df["ORDER_DATE"].dt.day
    df["DUE_YEAR"] = df["PURCHASE_ORDER_DUE_DATE"].dt.year
    df["DUE_MONTH"] = df["PURCHASE_ORDER_DUE_DATE"].dt.month
    df["DUE_DAY"] = df["PURCHASE_ORDER_DUE_DATE"].dt.day
    df["DISTANCE_IN_MILES_LOG"] = np.log1p(df["DISTANCE_IN_MILES"])  # log1p to avoid log(0) issues

    # drop original datetime columns since we extracted what we needed  
    df.drop(columns=["ORDER_DATE", "PURCHASE_ORDER_DUE_DATE"], inplace=True, errors="ignore")

    # applying decision rules (basically converting continuous values into binary features based on thresholds)  
    df["rule_DUE_DAY_high"] = (df["DUE_DAY"] > 16).astype(int)
    df["rule_ORDER_DAY_high"] = (df["ORDER_DAY"] > 17).astype(int)
    df["rule_DAYS_BETWEEN_ORDER_AND_DUE_DATE_high"] = (df["DAYS_BETWEEN_ORDER_AND_DUE_DATE"] > 15).astype(int)
    df["rule_PRODUCT_CLASSIFICATION_high"] = (df["PRODUCT_CLASSIFICATION"] > 21).astype(int)
    df["rule_TRANSIT_LEAD_TIME_high"] = (df["TRANSIT_LEAD_TIME"] > 10).astype(int)
    df["rule_AVERAGE_ORDER_CYCLE_DAYS_high"] = (df["AVERAGE_ORDER_CYCLE_DAYS"] > 16.64).astype(int)
    df["rule_AVERAGE_VENDOR_ORDER_CYCLE_DAYS_high"] = (df["AVERAGE_VENDOR_ORDER_CYCLE_DAYS"] > 9.75).astype(int)
    df["rule_GIVEN_TIME_TO_LEAD_TIME_RATIO_high"] = (df["GIVEN_TIME_TO_LEAD_TIME_RATIO"] > 1.05).astype(int)
    df["rule_PURCHASE_FROM_VENDOR_high"] = (df["PURCHASE_FROM_VENDOR"] > 1578).astype(int)

# define features & target  
# dropping irrelevant columns (id, target variable, and any other that doesn't contribute)  
X = train_X.drop(columns=["ID", "ON_TIME_AND_COMPLETE", "PURCHASING_LEAD_TIME"], errors="ignore")
y = train_y["ON_TIME_AND_COMPLETE"]

# separate numerical and categorical columns for preprocessing  
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# preprocessing pipeline  
# numeric: impute missing values with median (robust to outliers) + scale using robust scaler (also outlier-resistant)  
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# categorical: one-hot encoding (handle unknown categories smoothly)  
categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# combining both numeric and categorical pipelines  
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# build logistic regression pipeline  
# includes preprocessing, optional polynomial interactions, and logistic regression with balanced class weights  
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),  # only interactions, no quadratic terms
    ('logreg', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'))
])

# hyperparameter tuning for logistic regression  
# testing different regularization strengths (c values) using grid search  
param_grid = {'logreg__C': [0.1, 1, 10]}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# best model  
best_model = grid_search.best_estimator_
print(f"number of features: {len(list(best_model.feature_names_in_))}")

# get predicted probabilities on training data  
val_probs = best_model.predict_proba(X)[:, 1]  # extracting probabilities for the positive class  

# test multiple thresholds and find the one that maximizes accuracy  
thresholds = np.linspace(0, 1, 100)  # trying 100 different thresholds  
accuracy_scores = []

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)
    acc = accuracy_score(y, preds)
    accuracy_scores.append(acc)

# select the threshold that gives the highest accuracy  
best_threshold = thresholds[np.argmax(accuracy_scores)]
print(f"optimal threshold for maximum accuracy: {best_threshold}")

# apply model to test data  
test_X_processed = best_model[:-1].transform(test_X)  # apply preprocessing (everything except the final model step)
test_probs = best_model[-1].predict_proba(test_X_processed)[:, 1]  # get probability scores  

# use the optimized accuracy threshold  
test_predictions = (test_probs >= best_threshold).astype(int)

# save submission file  
submission = pd.DataFrame({"ID": test_X["ID"], "ON_TIME_AND_COMPLETE": test_predictions})
submission.to_csv("submission_004_thr.csv", index=False)
print("submission file with optimized accuracy threshold created!")
