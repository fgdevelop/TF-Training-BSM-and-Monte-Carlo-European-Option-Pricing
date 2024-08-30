import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

from training_utils import TrainUtils

df_test_dataset = pd.read_csv('C:\\Área de Trabalho\\Trabalho\\Quant Programs\\deep_learning_classes\\training_datasets\\bsm_dataset_df_test.xlsx')

dataset_code = '1T6L8y'
# df_test_dataset = pd.read_csv(f'test_datasets\\bsm_test_dataset_{dataset_code}.xlsx')
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = TrainUtils.create_scaled_dataset(df_test_dataset, 0.25)

model_path = 'models/test_bsm_model_scaling_func.keras'
model = load_model(model_path)
model.summary()

predictions = model.predict(x=x_test_scaled, verbose=2)
rounded_predictions = np.argmax(predictions, axis=-1)
for feat, target, pred in zip(x_test_scaled[:20], y_test_scaled[:20], predictions[:20]):
    print(f"Features: {feat}, Target: {target} | Prediction: {pred[0]}")

TrainUtils.plot_predictions(y_test_scaled, predictions)