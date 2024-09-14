# TF Training BSM and Monte Carlo European Option Pricing

Project involving the creation of a Black-Scholes-Merton (BSM) and Monte Carlo (MC) pricing machine using a deep learning ANN from tensorflow/keras package. This project has the goal of implementing a machine capable of pricing European options in order to have a faster performance when compared with the straight forward pricing models application. The scripts are organized as it follows:

- dataset_generator_bsm/_mc.py: Responsible for the creation of the raw datasets for BSM (on dataset_generator_bsm) and MC (on dataset_generator_mc) both follow a similar structure 
  creating a Pandas dataframe where each line  identify a set of features and a target value calculated by one of the two possible models (the model calculators are available on 
  class_bsm_mc_opt_price.py). The scenarios contained on the dataframes are setted by a dictionary containing the limits and steps of each one of the features. The main difference 
  between the BSM and MC dataset generators is that the BSM generator creates the full dataset all at once and store it on a csv file, meanwhile the MC generator, avoiding 
  possible Memory Errors, storage chunks of datasets on multiple CSV files and then compile all of then into one, deleting the chunk files.

- training_model.py (BSM/MC): This script uses the dataset created previously on either dataset_generator_bsm or dataset_generator_mc to train a ANN. The script separates the dataset 
  into training data (applied to the ANN) and test data (used to understand later the effectiveness of the trained machine). Both datasets are properly preprocessed using sklearn for 
  scaling and shuffle of the data. After the training, the model is stored as a .keras file.

- applying_model.py (BSM/MC): The purpose of this script is for a test of the created machine. It imports the test dataset (previously stored into a CSV file) to compare prediction of 
  the targets with the targets itself through a plot from matplotlib of this two instances. The graphic in case the model succeds must be a 45 degree crescent line due to the proximity 
  of the labeled and predicted targets.

- utils: Contains most of the data treatment functions along with other tools used on the previously described scripts.

- class_bsm_mc_opt_price.py: Contains most of the pricing functions and it is responsible for the creation of the scenarios/scenarios + price dataset.

OBS: 
- Currently the scripts must be on the same directory with 4 folders labeled as: mc_scenarios, mc_training_datasets, models, test_datasets, training_datasets for the storage of the CSV 
  and keras files.

- Each one of the dataset created will have a code of 3 numbers and 3 letters identifying it (ex. 2h2X9o), this must be the inputs for training_model.py and applying_model.py scripts.
