import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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

        tree.plot_tree(dtree, feature_names=x_train.columns,
                       max_depth=2, filled=True)

        # performance measures for tree with default parameters
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('F1 score: ', f1)

        if tune == True:

            # create a dict of params and values to do grid search
            params_dt = {
                'max_depth': [2, 3, 4, 5, 6, 7],
                'min_samples_leaf': [0.04, 0.06, 0.08, 0.1],
                'max_features': [0.2, 0.4, 0.6, 0.8]
            }

            # create an instance of gridseach class add params
            grid_dt = GridSearchCV(
                estimator=dtree,
                param_grid=params_dt,
                scoring='accuracy',
                cv=10,
                n_jobs=-1
            )

            grid_dt.fit(x_train, y_train)
            best_hyperparams = grid_dt.best_params_

            print(f'best hyperparameters are: \n {best_hyperparams}')
            print(f'best cv accuracy: \n {grid_dt.best_score_}')

            tuned_dtree = tree.DecisionTreeClassifier(
                max_depth=best_hyperparams.get('max_depth'),
                max_features=best_hyperparams.get('max_features'),
                min_samples_leaf=best_hyperparams.get('min_samples_leaf'),
                random_state=42
            )

            tuned_dtree.fit(x_train, y_train)

            # performance measures for tree with default parameters

            # evaluate the performance on the test subset
            y_pred = tuned_dtree.predict(x_test)

            tree.plot_tree(tuned_dtree, feature_names=x_train.columns,
                           max_depth=2, filled=True)

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

        if tune == True:
            pass

    elif model_type == 'xgboost':
        pass

        #   used for testing purposes
if __name__ == '__main__':

    raw_df = read_dataframe()
    describe_data(raw_df)
    run_model(raw_df, 'dtree', True)
