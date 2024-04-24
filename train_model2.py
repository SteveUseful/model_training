import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score, log_loss, confusion_matrix)

# Load the test data
X_test = pd.read_csv('C:/dev/Output/X_test.csv').drop(['CustomerID', 'ChurnRiskReason'], axis=1, errors='ignore')
y_test = pd.read_csv('C:/dev/Output/y_test.csv')['Churned']

# Load the trained model
model_path = 'best_mode2.pkl'  # Ensure this is the correct path to your best model
model = joblib.load(model_path)

def evaluate_model(X_test, y_test, model):
    """
    Evaluate the model using various metrics and plot ROC, Precision-Recall curves, and confusion matrix.
    """
    # Predicting probabilities and classes using the pipeline
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    # Print classification report and additional metrics
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Accuracy Score:", accuracy_score(y_test, predictions))
    print("Log Loss:", log_loss(y_test, probabilities))

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, probabilities)
    print("ROC AUC Score:", roc_auc)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluate the model on the test data
evaluate_model(X_test, y_test, model)
