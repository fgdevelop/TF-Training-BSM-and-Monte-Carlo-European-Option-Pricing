import os
import tensorflow as tf
import pandas as pd
import numpy as np

from os.path import join
from keras.models import Sequential
from keras.layers import Dense
from training_utils import TrainUtils

pc_path = TrainUtils.find_pc_path()

# inputs
dataset_cut_parameter = 84000
dataset_code = '1T6L8y'
test_path = join(pc_path, 'test_datasets')

# importing one of the possible datasets for training
df_dataset = pd.read_csv(f'training_datasets\\bsm_dataset_{dataset_code}.xlsx')

x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = TrainUtils.create_scaled_dataset(df_dataset, 0.25)

TrainUtils.storage_test_dataset(x_test_scaled, y_test_scaled, dataset_code, test_path)

x_train_scaled, y_train_scaled = TrainUtils.cut_dataset(x_train_scaled, y_train_scaled, dataset_cut_parameter=0)

model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

model.fit(x=x_train_scaled, y=y_train_scaled,
              batch_size=16,
              epochs=150,
              shuffle=True,
              verbose=2)

model_path = 'models/test_bsm_model_scaling_func.keras'

# saving everything (architecture, weights, configuration etc.)
if os.path.isfile(model_path) is False:
    model.save(model_path)
