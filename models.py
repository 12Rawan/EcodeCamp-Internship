from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# usual way for evaluating the models , every algorithm has its own model evaluation function
def train_and_evaluate_models(X_train, X_val, y_train, y_val, X_test, test_df):
    models = {
        'RandomForest':
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
        'DecisionTree':
        DecisionTreeClassifier(random_state=1),
        'LogisticRegression':
        LogisticRegression(max_iter=1000, random_state=1),
        'SVM':
        SVC(kernel='rbf', C=1, gamma='scale', random_state=1)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        print(f"\n{model_name} Validation accuracy: ",
              accuracy_score(y_val, y_val_pred))
        test_predictions = model.predict(X_test)
        output = pd.DataFrame({
            'PassengerId': test_df.PassengerId,
            'Survived': test_predictions
        })
        output.to_csv(f'{model_name}_submission.csv', index=False)
        print(f"{model_name} submission successfully saved!")


#########################################################################

# Another way (Cross validation) to enhance the accuracy and select the best model


def select_and_save_best_model(X, y, X_train, y_train, scaler):
    models = {
        'RandomForest':
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
        'DecisionTree':
        DecisionTreeClassifier(random_state=1),
        'LogisticRegression':
        LogisticRegression(max_iter=1000, random_state=1),
        'SVM':
        SVC(kernel='rbf', C=1, gamma='scale', random_state=1)
    }

    best_model_name = None
    best_model = None
    best_score = 0

    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"{model_name} Cross-Validation Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = model_name
            best_model = model

    print(
        f"\nBest Model: {best_model_name} with Cross-Validation Accuracy: {best_score:.4f}"
    )

    # Fit the best model and save it along with the scaler
    best_model.fit(X_train, y_train)
    model_filename = f'{best_model_name}_model.pkl'
    scaler_filename = 'scaler.pkl'
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Best model ({best_model_name}) and scaler successfully saved!")
