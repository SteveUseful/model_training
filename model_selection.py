import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('C:/dev/Output/Churn_Data1.csv')

# Preprocessing steps
# Identify categorical and numerical columns
categorical_features = ['Gender', 'AgeGroup', 'Partner', 'Dependents', 'ServiceType', 
                        'MultipleServices', 'ServiceAddons', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod']
numerical_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']

# Preprocessing for numerical data: scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define multiple models
models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(random_state=42, n_jobs=-1),
    # Note: GradientBoostingClassifier does not support the n_jobs parameter
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Create a pipeline that combines the preprocessor with the model placeholder
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', None)])  # Placeholder for the actual model

# Split the data
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert Churn to numerical
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define separate parameter grids for each model
param_grids = {
    'RandomForest': {
        'model__n_estimators': [100, 300, 500],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False]
    },
    'LogisticRegression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'model__penalty': ['l2']
    },
    'GradientBoosting': {
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
}

# Model training and evaluation loop
best_score = 0
best_model = None
for model_name, model in models.items():
    # Set the model in the pipeline
    pipeline.set_params(model=model)
    
    # Use the correct parameter grid for the current model
    current_param_grid = param_grids[model_name]
    
    # Instantiate the grid search with the correct parameter grid
    grid_search = GridSearchCV(pipeline, current_param_grid, cv=5, scoring='roc_auc')
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator and its score
    current_best_score = grid_search.best_score_
    current_best_model = grid_search.best_estimator_
    
    # Compare with the best score so far
    if current_best_score > best_score:
        best_score = current_best_score
        best_model = current_best_model
    
    # Make predictions
    predictions = current_best_model.predict(X_test)
    
    # Evaluate the model
    print(f"Results for {model_name}:")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(f'ROC AUC Score: {roc_auc_score(y_test, predictions)}')

# Save the best model
joblib.dump(best_model, 'best_model.pkl')

# Print a success message
print("The best model has been trained and saved successfully.")
