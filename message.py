# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing and splitting
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# For evaluation metrics and curves
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve

# For models
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load both training and testing datasets
train_path = 'fraudTrain.csv'
test_path = 'fraudTest.csv'

try:
    df_train = pd.read_csv(train_path)
    print("Training data loaded successfully.")
except Exception as e:
    print(f"Error loading training data: {e}")

try:
    df_test = pd.read_csv(test_path)
    print("Testing data loaded successfully.")
except Exception as e:
    print(f"Error loading testing data: {e}")

# Basic exploration of the training dataset
print("\n--- Training Data Overview ---")
print("Shape:", df_train.shape)
print("Missing Values:\n", df_train.isnull().sum())

# Basic exploration of the testing dataset
print("\n--- Testing Data Overview ---")
print("Shape:", df_test.shape)
print("Missing Values:\n", df_test.isnull().sum())

# Check for class imbalance in training data
if 'is_fraud' in df_train.columns:
    print("\nTraining Data Class Distribution:")
    print(df_train['is_fraud'].value_counts(normalize=True) * 100)
else:
    print("Target column 'is_fraud' not found in training data.")

# Plot correlation heatmap for numerical features
plt.figure(figsize=(12, 10))
sns.heatmap(df_train.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap (Training Data)")
plt.show()

# For consistent preprocessing, we merge the training and testing datasets.

# Concatenate the training and testing datasets
df_combined = pd.concat([df_train, df_test], keys=['train', 'test'], names=['dataset_origin'])
df_combined = df_combined.reset_index(level=0)

# Check for missing values
missing_values = df_combined.isnull().sum()
print("Missing Values:\n", missing_values)

# No missing values found in previous exploration.
# If missing values were present, imputation or removal would be performed here.

# Define numerical features
numerical_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

# Initialize scaler
scaler = StandardScaler()

# Feature engineering: Extract time components
df_combined['trans_date_trans_time'] = pd.to_datetime(df_combined['trans_date_trans_time'])
df_combined['hour'] = df_combined['trans_date_trans_time'].dt.hour
df_combined['dayofweek'] = df_combined['trans_date_trans_time'].dt.dayofweek
df_combined['month'] = df_combined['trans_date_trans_time'].dt.month

# Define selected features for scaling
selected_features = numerical_features + ['hour', 'dayofweek', 'month']

# Fit the scaler on training data only
train_data = df_combined[df_combined['dataset_origin'] == 'train']
scaler.fit(train_data[numerical_features])

# Apply scaling to the entire dataset
df_combined[numerical_features] = scaler.transform(df_combined[numerical_features])

# Polynomial feature engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_combined[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_combined.index)
df_combined = pd.concat([df_combined, df_poly.add_prefix('poly_')], axis=1)

# Create transaction amount bins
df_combined['amount_bins'] = pd.cut(df_combined['amt'], bins=[0, 10, 100, 500, 5000], labels=[1, 2, 3, 4])

# Output confirmation
print("Data processing complete. New shape of combined dataset:", df_combined.shape)

from sklearn.preprocessing import LabelEncoder

# Function to preprocess categorical data
def safetransform(encoder, series):
    # Map each value; if not in encoder.classes, assign -1.
    knownlabels = set(encoder.classes_)
    return series.apply(lambda x: encoder.transform([x])[0] if x in knownlabels else -1)
def preprocess_dataframe_optimized(df, fit_encoders=False, encoders=None):
    if encoders is None:
        encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        if col == 'trans_date_trans_time':
            continue

        if fit_encoders:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
        else:
            if col in encoders:
                # Create a mapping dictionary for fast lookup
                mapping = {label: idx for idx, label in enumerate(encoders[col].classes_)}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)  # Handle unknowns with -1
            else:
                df[col] = -1  # Assign unknown categories as -1
    return df, encoders

# ------------------------- Preprocessing Categorical Data -------------------------

# Fit label encoders on training data
df_train, label_encoders = preprocess_dataframe_optimized(df_train, fit_encoders=True)

# Apply same encoders on validation and test sets
df_test, _ = preprocess_dataframe_optimized(df_test, fit_encoders=False, encoders=label_encoders)


# Split the dataset back to training and testing sets using 'dataset_origin'
df_train_processed = df_combined[df_combined['dataset_origin'] == 'train'].drop(columns=['dataset_origin'])
df_test_processed = df_combined[df_combined['dataset_origin'] == 'test'].drop(columns=['dataset_origin', 'is_fraud'])  # No need to keep 'is_fraud' for test

# Ensure train data is not empty
if df_train_processed.empty:
    raise ValueError("Error: Training dataset is empty after preprocessing. Check dataset merging.")

# Define features (X) and target (y)
X = df_train_processed.drop(columns=['is_fraud'])
y = df_train_processed['is_fraud']

# Ensure 'is_fraud' has at least two classes for stratification
if y.nunique() < 2:
    raise ValueError("Error: 'is_fraud' must have at least two classes for stratification.")

# Split the dataset into training and validation sets (keeping class balance)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print shapes
print("\nAfter splitting:")
print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Testing set shape:", df_test_processed.shape)

# Define a function to evaluate model performance
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.decision_function(X_val)
    auc = roc_auc_score(y_val, y_proba)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    return auc, prec, rec, f1

# Select features and target variable for logistic regression model
features = ['amt', 'hour', 'dayofweek', 'month', 'amount_bins']
target = 'is_fraud'
X_train_lr = df_train_processed[features]
y_train_lr = df_train_processed[target]

# Impute missing values using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # You can choose other strategies like 'median' or 'most_frequent'
X_train_lr = imputer.fit_transform(X_train_lr) # Fit and transform on X_train_lr

# Convert back to DataFrame (optional but recommended)
X_train_lr = pd.DataFrame(X_train_lr, columns=features, index=y_train_lr.index)

# Train a simple logistic regression model with cross-validation
model = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(model, X_train_lr, y_train_lr, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC score: {cv_scores.mean()}")

# Drop datetime column
X_train = X_train.drop(columns=['trans_date_trans_time'])
X_val = X_val.drop(columns=['trans_date_trans_time'])

# Convert 'amount_bins' to numeric, handling NaN values
X_train['amount_bins'] = pd.to_numeric(X_train['amount_bins'], errors='coerce').fillna(0).astype(int)
X_val['amount_bins'] = pd.to_numeric(X_val['amount_bins'], errors='coerce').fillna(0).astype(int)

# Identify object-type columns that need encoding
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", categorical_cols)

# Drop non-numeric categorical columns (if necessary) or encode them
X_train = X_train.drop(columns=categorical_cols)
X_val = X_val.drop(columns=categorical_cols)

# Define the model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define hyperparameter grids for optimization
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Perform Grid Search for XGBoost
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_

# Evaluate model
xgb_auc, xgb_prec, xgb_rec, xgb_f1 = evaluate_model(best_xgb, X_val, y_val)

# Print results
print("\nOptimized XGBoost Performance:")
print(f"ROC-AUC: {xgb_auc:.4f}, Precision: {xgb_prec:.4f}, Recall: {xgb_rec:.4f}, F1 Score: {xgb_f1:.4f}")

# Drop datetime column from test data before prediction
df_test_processed = df_test_processed.drop(columns=['trans_date_trans_time'], errors='ignore')

# Drop non-numeric categorical columns from test data
test_categorical_cols = df_test_processed.select_dtypes(include=['object']).columns.tolist()
df_test_processed = df_test_processed.drop(columns=test_categorical_cols)

# If 'amount_bins' is still categorical, convert it to numeric
if 'amount_bins' in df_test_processed.columns:
    df_test_processed['amount_bins'] = pd.to_numeric(df_test_processed['amount_bins'], errors='coerce').fillna(0).astype(int)

# Predict on test set
test_predictions = best_xgb.predict(df_test_processed)

submission = pd.DataFrame({
    'transaction_id': df_test['trans_num'],  # use trans_num as the unique identifier
    'is_fraud': test_predictions
})
submission.to_csv('fraud_predictions.csv', index=False)
print("Predictions saved to fraud_predictions.csv")

import joblib

# Save the best XGBoost model
joblib.dump(best_xgb, 'xgb_fraud_model.pkl')
print("XGBoost model saved as xgb_fraud_model.pkl")

# Save the scaler (ensure it was fitted on training data)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# Save the PolynomialFeatures transformer
joblib.dump(poly, 'poly_features.pkl')
print("PolynomialFeatures transformer saved as poly_features.pkl")

# Optionally, save label encoders if needed
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Label encoders saved as label_encoders.pkl")