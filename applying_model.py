import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score


from training_utils import TrainUtils

def plot_predictions(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Real Value")
    plt.ylabel("ANN Value")
    plt.show()

df_test_dataset = pd.read_csv(f'test_datasets\\bsm_test_dataset_cross_test.xlsx')

# include column names for cross test
column_names=["opt_type","S0","K", "vol", "r", "Dt", "price"]
df_test_dataset.columns = column_names

test_dataset_scaled, test_target_scaled = TrainUtils.create_scaled_dataset(df_test_dataset)
model_path = 'models/test_bsm_model_cross_test.keras'
model = load_model(model_path)
model.summary()

predictions = model.predict(x=test_dataset_scaled, verbose=2)
rounded_predictions = np.argmax(predictions, axis=-1)
for feat, target, pred in zip(test_dataset_scaled[:20], test_target_scaled[:20], predictions[:20]):
    print(f"Features: {feat}, Target: {target} | Prediction: {pred[0]}")

plot_predictions(test_target_scaled, predictions)