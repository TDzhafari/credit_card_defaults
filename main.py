import pandas as pd
import xlrd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


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


def run_model(df, model_type):

    # subset with independent variables
    X = df  # .drop('default payment next month', axis=1)

    # subset with dependent variable
    y = df['default payment next month']

    print(y.head().to_string())

    # split the data on train and test
    x_train, y_train, x_test, y_test = train_test_split(
        X, y, train_size=0.7)
    #y_train = y_train['default payment next month']

    # run logistic regression
    if model_type == 'logistic':
        log_reg = LogisticRegression(random_state=0)

        # print(y_train.head().to_string())
        print(x_train.head().to_string())

        y_train = y_train['default payment next month']
        x_train = x_train.drop('default payment next month', axis=1)

        print(y_train)

        log_reg.fit(x_train, y_train)

        y_pred = log_reg.predict(x_test)
        # evaluate the accuracy of the model

        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)


#   used for testing purposes
if __name__ == '__main__':

    raw_df = read_dataframe()
    describe_data(raw_df)
    run_model(raw_df, 'logistic')
