from class_bsm_mc_opt_price import PricingUtils
from utils import TrainUtils

if __name__ == '__main__':
    dataset_code = TrainUtils.random_code_generator(3)
    size_of_each_chunk = 10000

    # inputs to create training data
    scenario_configuration = {"S0": [40, 100, 5],
                              "K": [50, 155, 5],
                              "vol": [0.1, 1.1, .05],
                              "r": [.0, .1, .01],
                              "Dt": [0.1, 2.1, 0.1]}

    # creating scenarios for training and test data
    scenarios_data = PricingUtils.create_scenarios(scenario_configuration, dataset_code)
    size_scenarios = len(scenarios_data)

    # cuts the scenarios df into section and storages the scenarios in a csv with the given random code
    TrainUtils.storage_scenarios_csv(scenarios_data, dataset_code)

    # create xlsx files with chunks from the full scenarios dataset and storage them in a directory
    TrainUtils.create_chunk_files(size_of_each_chunk, size_scenarios, dataset_code)

    # compile the xlsx chunk datasets into one and send it to the same folder (deleting the chunk ones)
    TrainUtils.compile_chunks_into_one(size_of_each_chunk, size_scenarios, dataset_code)