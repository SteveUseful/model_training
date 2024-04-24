import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from scipy.stats import randint

# Load the dataset
df = pd.read_csv('C:/dev/Output/Vectorized_Churn_Data.csv')

# Define categorical and numerical features
categorical_features = ['CustomerEducationLevel', 'ContractRenewal']
numerical_features = ['CustomerAge', 'CustomerIncome', 'SupportInteraction', 
                      'SatisfactionScore', 'UsageFrequency', 'FeatureUsageScore', 
                      'AccountLifetime', 'MonthlyCharges', 'TotalCharges']
target_variable = 'Churned'

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define multiple models
models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000, solver='saga', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(df[target_variable]), y=df[target_variable])
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'RandomForest': {
        'model__n_estimators': randint(200, 300),
        'model__max_depth': randint(10, 30),
        'model__min_samples_split': randint(2, 6),
        'model__min_samples_leaf': randint(1, 4),
        'model__bootstrap': [True],
        'model__class_weight': [class_weight_dict]
    },
    'LogisticRegression': {
        'model__C': np.logspace(-3, 2, 6),
        'model__class_weight': [class_weight_dict]
    },
    'GradientBoosting': {
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': randint(100, 300),
        'model__max_depth': randint(3, 10),
        'model__subsample': [0.8, 0.9, 1.0],
        'model__min_samples_split': randint(2, 6),
        'model__min_samples_leaf': randint(1, 4)
    }
}

# Create a pipeline with preprocessor and a placeholder for the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', None)])

# Split the data
X = df.drop([target_variable, 'ChurnRiskReason'], axis=1)  # Exclude 'ChurnRiskReason' from features
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and evaluation loop
best_score = 0
best_model = None
for model_name, model in models.items():
    pipeline.set_params(model=model)
    current_param_grid = param_grids[model_name]
    randomized_search = RandomizedSearchCV(pipeline, param_distributions=current_param_grid, n_iter=50, cv=5,
                                           scoring='roc_auc', random_state=42, n_jobs=-1)
    randomized_search.fit(X_train, y_train)
    if randomized_search.best_score_ > best_score:
        best_score = randomized_search.best_score_
        best_model = randomized_search.best_estimator_

# Make predictions with the best model
predictions = best_model.predict(X_test)

# Evaluate the best model
print(f"Results for {best_model.steps[-1][0]}:")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(f'ROC AUC Score: {roc_auc_score(y_test, predictions)}')

# Save the best model
joblib.dump(best_model, 'best_mode2.pkl')

# Print a success message
print("The best model has been trained and saved successfully.")
