import random
import string

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class TrainUtils:

    @staticmethod
    def find_pc_path():
        return Path(__file__).parent.__str__()

    @staticmethod
    def random_code_generator(digits: int) -> str:
        code = ''
        letters = string.ascii_letters
        for digit in range(0, digits):
            code += str(random.randint(0,9))
            code += random.choice(letters)
        return code

    @staticmethod
    def create_scaled_dataset(df_dataset: pd.DataFrame, test_size: float) -> (np.array, np.array):
        dataset = np.asarray(df_dataset)

        # setting the target features and target
        x = dataset[:, :len(df_dataset.values[0]) - 1]
        y = dataset[:, (len(df_dataset.values[0]) - 1):len(df_dataset.values[0])]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

        # scaling data for features and target to be within (0,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled, y_train_scaled = scaler.fit_transform(x_train), scaler.fit_transform(y_train)
        x_test_scaled, y_test_scaled = scaler.fit_transform(x_test), scaler.fit_transform(y_test)

        # shuffling dataset
        y_train_scaled, x_train_scaled = shuffle(y_train_scaled, x_train_scaled)
        y_test_scaled, x_test_scaled = shuffle(y_test_scaled, x_test_scaled)

        return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

    @staticmethod
    def storage_test_dataset(x_test: np.array, y_test: np.array, dataset_code: str, output_path: str) -> None:
        df_test = pd.DataFrame(x_test)
        column_names=["opt_type","S0","K", "vol", "r", "Dt"]
        df_test.columns = column_names
        df_test['target'] = y_test
        df_test.to_csv(join(output_path, f'bsm_test_dataset_{dataset_code}.xlsx'), index_label=False)

    @staticmethod
    def cut_dataset(x, y, dataset_cut_parameter: int = 0):
        if dataset_cut_parameter == 0:
            return x, y
        else:
            x = x[:dataset_cut_parameter]
            y = y[:dataset_cut_parameter]
            return x, y

    @staticmethod
    def plot_predictions(y_test, y_pred):
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Real Value")
        plt.ylabel("ANN Value")
        plt.show()