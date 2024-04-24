import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load the dataset
df = pd.read_csv('C:/dev/Output/Churn_Data1.csv')

# Feature and label definition
X = df.drop(['CustomerID', 'Churned'], axis=1)
y = df['Churned']

# Define preprocessing for numerical features
num_features = ['ProductCustomerFitScore', 'OnboardingCompletion', 'CustomerServiceRating',
                'ValueForMoneyRating', 'ReportedBugs', 'PaymentIssuesLastYear', 'FeatureRequests',
                'InteractionScore', 'RiskFactor']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
cat_features = ['CustomerFeedback', 'CustomerSegment']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Model definition
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stacked_learner = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# Pipeline with SMOTE and model
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', stacked_learner)
])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search for hyperparameter tuning
param_grid = {
    'classifier__rf__max_depth': [10, 20],
    'classifier__rf__min_samples_split': [2, 5],
    'classifier__gb__learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Model training
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Predictions and evaluation
y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
