import os
import tensorflow as tf
import pandas as pd

from os.path import join
from keras.models import Sequential
from keras.layers import Dense
from utils import TrainUtils

"""
    Labeling for each pricing model:
    BS: 'training_datasets\\bsm_dataset_{dataset_code}.xlsx'
    MC: 'mc_training_datasets\\mc_dataset_{dataset_code}.xlsx'
"""

# input: insert here the dataset identification when generated
dataset_code = '2h2X9o'

# setting paths
pc_path = TrainUtils.find_pc_path()
test_path = join(pc_path, 'test_datasets')
dataset_path = f'mc_training_datasets\\mc_dataset_{dataset_code}_full.xlsx'
model_path = f'models/mc_model_{dataset_code}.keras'

if __name__ == '__main__':
    # importing one of the possible datasets for training
    df_dataset = pd.read_csv(dataset_path)

    # separating the dataset between features and targets for training and testing
    x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = TrainUtils.create_scaled_dataset(df_dataset, 0.05)

    # storing the test dataset to use it later
    TrainUtils.storage_test_dataset(x_test_scaled, y_test_scaled, dataset_code, test_path)

    # ANN configuration
    model = Sequential()
    model.add(Dense(10, input_dim=6, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # printing the ANN configuration
    model.summary()

    # defining compiler
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

    # training the model using the training dataset, printing the loss function process
    model.fit(x=x_train_scaled, y=y_train_scaled,
                  batch_size=16,
                  epochs=150,
                  shuffle=True,
                  verbose=2)

    # saving everything (architecture, weights, configuration etc.)
    if os.path.isfile(model_path) is False:
        model.save(model_path)
        print(f'The model was saved as mc_model_{dataset_code}.keras.')
