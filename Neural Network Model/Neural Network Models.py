

""" VERY IMPORTMANT """
""" Before run choose to uncomment the excel sheet of New York or Cairo to use the data """

""" Imports """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class ModelOne(object):
    """ Daily Cases Model """
    def train_NN_daily(self):
        # excel_sheet_format = pd.read_csv("./NewYork_Daily.csv")     # Uncomment to use New York data
        # excel_sheet_format = pd.read_csv("./Cairo_Daily.csv")       # Uncomment to use Cairo data
        rows = 60   # Number of days taken
        features_number = 2
        excel_sheet_format_list = np.array(excel_sheet_format.values.tolist())
        factors_data = excel_sheet_format_list[:, 0:features_number]
        labels = excel_sheet_format_list[:, features_number].reshape(rows, 1)  # cases
        x_train, x_test, y_train, y_test = train_test_split(    # capture 30 % of samples as test
            factors_data, labels, test_size=0.2, random_state=42)
        xx_train = x_train.reshape(x_train.shape[0], features_number, 1)
        xx_test = x_test.reshape(x_test.shape[0], features_number, 1)

        model = Sequential([
            # input layer
            Dense(128, kernel_initializer='normal', input_shape=(
                features_number, 1), activation='relu'),
            # hidden layer
            Dense(256, kernel_initializer='normal', activation='relu'),
            Dense(256, kernel_initializer='normal', activation='relu'),
            Flatten(),
            # output layer
            Dense(1, kernel_initializer='normal', activation='linear'),
        ])

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['accuracy', 'mean_absolute_error'],
        )
        model.summary()
        model.fit(xx_train, y_train, epochs=500, batch_size=32,
                  validation_split=0.2)
        y_prediction = model.predict(xx_test)
        print("*"*25 + "Final Results" + "*"*25)
        print("Random Factors Samples")
        print(x_test)
        print("*"*25)
        print("Expected Daily Cases")
        print(y_test)
        print("*"*25)
        print("Model's Prediction")
        print(y_prediction)

    def train_NN_total(self):
        """ Total Cases Model """
        # excel_sheet_format = pd.read_csv("./NewYork_Total.csv")     # Uncomment to use New York data
        # excel_sheet_format = pd.read_csv("./Cairo_Total.csv")       # Uncomment to use Cairo data
        rows = 60   # Number of days taken
        features_number = 2
        excel_sheet_format_list = np.array(excel_sheet_format.values.tolist())
        factors_data = excel_sheet_format_list[:, 0:features_number]     # temp , humidity or avg temp, growth rate
        labels = excel_sheet_format_list[:, features_number].reshape(rows, 1)  # cases or total cases
        x_train, x_test, y_train, y_test = train_test_split(    # capture 30 % of samples as test
            factors_data, labels, test_size=0.2, random_state=42)
        xx_train = x_train.reshape(x_train.shape[0], features_number, 1)
        xx_test = x_test.reshape(x_test.shape[0], features_number, 1)

        model = Sequential([
            # input layer
            Dense(128, kernel_initializer='normal', input_shape=(
                features_number, 1), activation='relu'),
            # hidden layer
            Dense(256, kernel_initializer='normal', activation='relu'),
            Dense(256, kernel_initializer='normal', activation='relu'),
            Flatten(),
            # output layer
            Dense(1, kernel_initializer='normal', activation='linear'),
        ])

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['accuracy', 'mean_absolute_error'],
        )
        model.summary()
        model.fit(xx_train, y_train, epochs=500, batch_size=32,
                  validation_split=0.2)
        y_prediction = model.predict(xx_test)
        print("*"*25 + "Final Results" + "*"*25)
        print("Random Factors Samples")
        print(x_test)
        print("*"*25)
        print("Expected Daily Cases")
        print(y_test)
        print("*"*25)
        print("Model's Prediction")
        print(y_prediction)


def main():
    obj = ModelOne()
    print("Daily Cases Training")
    obj.train_NN_daily()
    print("*"*50)
    print("Total Cases Training ")
    obj.train_NN_total()


if __name__ == "__main__":
    main()
