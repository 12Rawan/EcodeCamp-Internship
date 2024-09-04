import pandas as pd
import numpy as np


def load_data(train_path, test_path, gender_submission_path):
    Train_df = pd.read_csv(train_path)
    Test_df = pd.read_csv(test_path)
    Gender_submission_df = pd.read_csv(gender_submission_path)
    return Train_df, Test_df, Gender_submission_df


def handle_missing_values(df_train, df_test):
    df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
    df_train['Has a Cabin'] = df_train['Cabin'].notna().astype(int)
    df_train.drop('Cabin', axis=1, inplace=True)
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)

    df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
    df_test['Has a Cabin'] = df_test['Cabin'].notna().astype(int)
    df_test.drop('Cabin', axis=1, inplace=True)
    df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

    return df_train, df_test



# Create a 'Family size' feature to add both of 'Sibsp' + 'Parch' and also create 'Alone' Feature if the person alone without family then Family size will be  = 1 else = 0
def feature_engineering(df_train, df_test):
    df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch'] + 1
    df_train['Alone'] = df_train['Family_Size'].apply(lambda x: 1
                                                      if x == 1 else 0)

    df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch'] + 1
    df_test['Alone'] = df_test['Family_Size'].apply(lambda x: 1
                                                    if x == 1 else 0)
    

    return df_train, df_test



# Encoding Categorical variable ('Sex' , 'Embarked')
def encode_categorical_variables(df_train, df_test):
    df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
    df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
    df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df_train, df_test
