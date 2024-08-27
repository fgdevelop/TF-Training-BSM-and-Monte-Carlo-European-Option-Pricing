import random
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

class TrainUtils:

    @staticmethod
    def random_code_generator(digits: int) -> str:
        code = ''
        letters = string.ascii_letters
        for digit in range(0, digits):
            code += str(random.randint(0,9))
            code += random.choice(letters)
        return code

    @staticmethod
    def create_scaled_dataset(df_dataset: pd.DataFrame()) -> tf.data.Dataset.from_tensor_slices:
        # scaling data for features to be in (0,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_dataset_scaled = pd.DataFrame(scaler.fit_transform(df_dataset.values), columns=df_dataset.columns,
                                         index=df_dataset.index)

        # setting the target features and target
        target = df_dataset_scaled.pop('price')

        # shuffling dataset
        target, df_dataset_scaled = shuffle(target, df_dataset_scaled)

        # changing to array format
        target = target.values
        df_dataset_scaled = df_dataset_scaled.values

        for feat, targ in zip(df_dataset_scaled[:10], target[:10]):
            print(f'Features: {feat}, Target: {targ}')

        return df_dataset_scaled, target

    @staticmethod
    def create_train_test_set(dataset: np.array, perc_of_dataset: float):
        len_dataset = len(dataset)
        len_test_set = int(len_dataset * perc_of_dataset)
        train_dataset = dataset[:len_test_set]
        test_dataset = dataset[len_test_set:]
        return train_dataset, test_dataset
