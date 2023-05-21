import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score


def read_dataframe() -> pd.DataFrame:
    """
    This module retrieves the dataset using pandas library
    """
    # Get the directory of the script
    dir_path = Path(__file__).resolve().parent

    # Append the 'data' folder
    data_path = dir_path / 'data' / 'default of credit card clients.xls'

    # Convert the path to a string
    data_path_str = str(data_path)
    raw_df = pd.read_excel(data_path_str, skiprows=1)

    return raw_df


def describe_data(df):
    """
    This module includes certain descriptive analytics.
    """
    print(round(df.describe(), 2).to_string())
    print(df.info())


def run_model(df, model_type, tune):

    # subset with independent variables
    X = df.drop('default payment next month', axis=1)

    # subset with dependent variable
    y = df['default payment next month']

    print(y.head().to_string())

    # split the data on train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7)
    #y_train = y_train['default payment next month']

    # run logistic regression
    if model_type == 'logistic':
        log_reg = LogisticRegression(random_state=0)
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        # evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('F1 score: ', f1)

    elif model_type == 'dtree':

        # create an instance of a decision tree classifier model
        dtree = tree.DecisionTreeClassifier(random_state=42)

        # train it on the training subset
        dtree = dtree.fit(x_train, y_train)

        # evaluate the performance on the test subset
        y_pred = dtree.predict(x_test)

        # visualize the tree
        tree.plot_tree(dtree, feature_names=x_train.columns,
                       max_depth=2, filled=True)

        # performance measures for tree with default parameters
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('F1 score: ', f1)

        if tune == True:

            # for loop to perform hyperparameter tuning for tree depth
            for max_d in range(1, 25):

                # Create an instance of the tree
                model = tree.DecisionTreeClassifier(
                    max_depth=max_d, random_state=42,)

                # train the model
                model.fit(x_train, y_train)
                # print out the performance measures. I'll do accuracy.
                print('The Training Accuracy for max_depth {} is:'.format(
                    max_d), model.score(x_train, y_train))
                print('The Validation Accuracy for max_depth {} is:'.format(
                    max_d), model.score(x_test, y_test))
                print('')

            model = tree.DecisionTreeClassifier(
                max_depth=4, random_state=42)

            model.fit(x_train, y_train)

            # performance measures for tree with default parameters
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print('Accuracy:', accuracy)
            print('F1 score: ', f1)

    elif model_type == 'random_forest':
        rfc = RandomForestClassifier(n_estimators=100, criterion='gini', )
        rfc = rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('F1 score: ', f1)

    elif model_type == 'xgboost':
        pass

        #   used for testing purposes
if __name__ == '__main__':

    raw_df = read_dataframe()
    describe_data(raw_df)
    run_model(raw_df, 'dtree', True)
