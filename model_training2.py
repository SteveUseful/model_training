import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
df = pd.read_csv('C:/dev/Output/Churn_Data1.csv')

# Preprocessing steps
df['CustomerFeedback'] = df['CustomerFeedback'].astype(str)
df['ChurnRiskReason'] = df['ChurnRiskReason'].astype(str)

# Define the feature set and labels
X = df.drop('Churned', axis=1)
y = df['Churned']

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['ProductCustomerFitScore', 'OnboardingCompletion', 'CustomerServiceRating', 
             'ValueForMoneyRating', 'ReportedBugs', 'PaymentIssuesLastYear', 'FeatureRequests']),
        ('text_feedback', TfidfVectorizer(max_features=100), 'CustomerFeedback'),
        ('text_reason', TfidfVectorizer(max_features=100), 'ChurnRiskReason')
    ],
    remainder='drop'
)

# Define the model to train
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Create the pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# Hyperparameter tuning
param_distributions = {
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300]
}

# Stratified k-fold for cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(
    pipeline, param_distributions, n_iter=50, scoring='roc_auc',
    cv=stratified_kfold, random_state=42, n_jobs=-1
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the model
random_search.fit(X_train, y_train)

# Best hyperparameters
best_params = random_search.best_params_
print(f"Best hyperparameters: {best_params}")

# Train the model with best hyperparameters
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Save the model
import joblib
joblib.dump(pipeline, 'churn_prediction_model.pkl')
