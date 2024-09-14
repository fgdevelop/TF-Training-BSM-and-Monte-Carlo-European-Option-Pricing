import random
import string
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from class_bsm_mc_opt_price import EurOptMC

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
    def storage_scenarios_csv(scenarios_data: pd.DataFrame, code: str) -> None:
        pc_path = TrainUtils.find_pc_path()
        scenarios_data.to_csv(join(pc_path, 'scenarios_mc', f'scenarios_data_mc_{code}.xlsx'), index=False)
        print(f'The dataset containing the scenarios was created and stored on the csv file scenarios_data_mc_{code}.xlsx')
        del scenarios_data
        print('The dataset dataframe was deleted.')

    @staticmethod
    def create_chunk_files(size_of_each_chunk: int, size_scenarios: int, code: str) -> None:
        pc_path = TrainUtils.find_pc_path()
        number_chunks = int(size_scenarios / size_of_each_chunk)
        for i in range(1, number_chunks+1):
            curr_chunks = i * size_of_each_chunk
            last_chunk = (i - 1) * size_of_each_chunk
            scenarios_chunk = pd.read_csv(join(pc_path, 'scenarios_mc', f'scenarios_data_mc_{code}.xlsx'))[last_chunk:curr_chunks]
            # creating training and test data using MC
            priced_chunk_mc = EurOptMC.df_pricing_mc(scenarios_chunk, simulations=5000)
            # storing training dataset or test dataset
            priced_chunk_mc.to_csv(f'mc_training_datasets\\mc_dataset_{code}_{i}.xlsx', index=False)
            print(f'Chunk {i} was created and stored on the csv file mc_dataset_{code}_{i}.xlsx')

    @staticmethod
    def compile_chunks_into_one(size_of_each_chunk: int, size_scenarios: int, code: str):
        full_df_dataset = pd.DataFrame()
        number_chunks = int(size_scenarios / size_of_each_chunk)
        for i in range(1, number_chunks+1):
            file_path = f'mc_training_datasets\\mc_dataset_{code}_{i}.xlsx'
            priced_chunk_mc = pd.read_csv(file_path)
            full_df_dataset = pd.concat([full_df_dataset, priced_chunk_mc], ignore_index=True)
            try:
                os.remove(file_path)
                print(f"File '{file_path}' deleted successfully.")
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
        full_df_dataset.to_csv(f'mc_training_datasets\\mc_dataset_{code}_full.xlsx', index=False)
        print(f'The csv containing the full dataset (features and targets) was created on mc_training_datasets with name mc_dataset_{code}_full.xlsx.')

    @staticmethod
    def create_scaled_dataset(df_dataset: pd.DataFrame, test_size: float) -> (np.array, np.array):
        dataset = np.asarray(df_dataset)

        # setting the features and target
        x = dataset[:, :len(df_dataset.values[0]) - 1]
        y = dataset[:, (len(df_dataset.values[0]) - 1):len(df_dataset.values[0])]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        print('The dataset was separated into features and targets for training and test.')

        # scaling data for features and target to be within (0,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled, y_train_scaled = scaler.fit_transform(x_train), scaler.fit_transform(y_train)
        x_test_scaled, y_test_scaled = scaler.fit_transform(x_test), scaler.fit_transform(y_test)
        print('The dataset features and targets were scaled between 0 and 1.')

        # shuffling dataset
        y_train_scaled, x_train_scaled = shuffle(y_train_scaled, x_train_scaled)
        y_test_scaled, x_test_scaled = shuffle(y_test_scaled, x_test_scaled)
        print('The dataset features and targets were shuffled.')

        print('The dataset is now preprocessed for training and testing.')
        return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

    @staticmethod
    def setting_test_dataset(df_dataset: pd.DataFrame):
        dataset = np.asarray(df_dataset)

        # setting the features and target
        x = dataset[:, :len(df_dataset.values[0]) - 1]
        y = dataset[:, (len(df_dataset.values[0]) - 1):len(df_dataset.values[0])]
        print('The test dataset was separated into features and targets.')
        return x, y

    @staticmethod
    def storage_test_dataset(x_test: np.array, y_test: np.array, dataset_code: str, output_path: str) -> None:
        df_test = pd.DataFrame(x_test)
        column_names=["opt_type","S0","K", "vol", "r", "Dt"]
        df_test.columns = column_names
        df_test['target'] = y_test
        df_test.to_csv(join(output_path, f'test_dataset_{dataset_code}.xlsx'), index_label=False)
        print(f'The test dataframe was created and stored as test_dataset_{dataset_code}.xlsx.')

    @staticmethod
    def plot_predictions(y_test, y_pred):
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Real Value")
        plt.ylabel("ANN Value")
        plt.show()