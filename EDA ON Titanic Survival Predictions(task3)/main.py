from app import create_app
from preprocess import load_data, handle_missing_values, feature_engineering, encode_categorical_variables
from models import train_and_evaluate_models, select_and_save_best_model
from visualization import plot_age_distribution, plot_survival_by_sex, plot_correlation_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import plot_survival_counts, plot_age_distribution, plot_survival_by_sex, plot_correlation_heatmap


def main():
  # Load and prepare data
  Train_df, Test_df, _ = load_data('train.csv', 'test.csv',
                                   'gender_submission.csv')

  # Data Overview :
  print("Train dataset : ", Train_df.head())
  print("\n Test dataset : ", Test_df.head())

  print("Train dataset Summary : ", Train_df.describe())
  print("Test dataset Summary : ", Test_df.describe())

  print("Train dataset info: ", Train_df.info())
  print("Test dataset info: ", Test_df.info())

  print("\nMissing values in Training Data: ", Train_df.isnull().sum())
  print("\nMissing values in Testing Data:", Test_df.isnull().sum())

  # noticed that in Train dataset we found 177 missing value in (Age) , 687 in (Cabin) , 2 in (Embarked)
  # while in test data set we found 86 missing value in (Age) , 1 in (Fare) , 327 in (Cabin)

  ########################################################################

  Train_df, Test_df = handle_missing_values(Train_df, Test_df)
  # verify it , to make sure it handled
  print("\n Missing values of Training dataset :  ", Train_df.isnull().sum())
  print("\n Missisng values od Testing dayaset : ", Test_df.isnull().sum())

  #######################################################################
  Train_df, Test_df = feature_engineering(Train_df, Test_df)
  print(Train_df[['SibSp', 'Parch', 'Family_Size', 'Alone']].head())
  print(Test_df[['SibSp', 'Parch', 'Family_Size', 'Alone']].head())

  ########################################################################
  Train_df, Test_df = encode_categorical_variables(Train_df, Test_df)
  print(Train_df.head())
  print(Test_df.head())

  ##########################################################################
  # gender_submission file assumes that all females survived and all males died .
  # i will check if this is true in the data(Train.csv)

  women = Train_df.loc[Train_df.Sex == 1]['Survived']
  rate_women = sum(women) / len(women)

  print("% of women who survived:", rate_women)
  # Rate of survived women = 74 %

  men = Train_df.loc[Train_df.Sex == 0]['Survived']
  rate_men = sum(men) / len(men)

  print("% of men who survived:", rate_men)
  # Rate of survived men  = 19 %

  # I want to find patterns in train.csv that help us predict whether the passengers in test.csv survived.
  # "PassengerId" column in gender_submmision containing the IDs of each passenger from test.csv.
  # "Survived" column (that you will create!) with a "1" for the rows where you think the passenger survived, and a "0" where you predict that the passenger died.
  #########################################################################

  # Splitting the data
  Features = ["Pclass", "Sex", "SibSp", "Parch"]
  X = Train_df[Features]
  X_test = Test_df[Features]
  y = Train_df['Survived']

  # Standardize features
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  X_test = scaler.transform(X_test)

  X_train, X_val, y_train, y_val = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

  # Train and evaluate models
  train_and_evaluate_models(X_train, X_val, y_train, y_val, X_test, Test_df)

  # Select and save the best model
  select_and_save_best_model(X, y, X_train, y_train, scaler)

  # Visualization (Save plots as files to avoid blocking)
  plot_age_distribution(Train_df, save_as='age_distribution.png')
  plot_survival_by_sex(Train_df, save_as='survival_by_sex.png')
  plot_correlation_heatmap(Train_df, save_as='correlation_heatmap.png')
  plot_survival_counts(Train_df, save_as='survival_counts.png')

  # Run Flask app
  app = create_app()
  app.run(host='0.0.0.0', port=80, debug=True)


if __name__ == '__main__':
  main()
