import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from utils import TrainUtils

# input: insert here the dataset identification when generated
dataset_code = '2h2X9o'

# setting paths
model_path = f'models/mc_model_{dataset_code}.keras'

if __name__ == '__main__':
    print('Importing the test dataset from csv file.')
    df_test_dataset = pd.read_csv(f'test_datasets\\test_dataset_{dataset_code}.xlsx')
    x_test_scaled, y_test_scaled = TrainUtils.setting_test_dataset(df_test_dataset)

    print('Loading model from keras file.')
    model = load_model(model_path)
    model.summary()

    predictions = model.predict(x=x_test_scaled, verbose=2)
    rounded_predictions = np.argmax(predictions, axis=-1)
    for feat, target, pred in zip(x_test_scaled[:20], y_test_scaled[:20], predictions[:20]):
        print(f"Features: {feat}, Target: {target} | Prediction: {pred[0]}")

    TrainUtils.plot_predictions(y_test_scaled, predictions)