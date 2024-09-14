from class_bsm_mc_opt_price import EurOptBS, PricingUtils
from utils import TrainUtils

if __name__ == '__main__':
    dataset_code = TrainUtils.random_code_generator(3)

    # inputs to create training data
    scenario_configuration = {"S0": [40, 100, 5],
                              "K": [50, 155, 5],
                              "vol": [0.1, 1.1, .05],
                              "r": [.0, .1, .01],
                              "Dt": [0.1, 2.1, 0.1]}

    # creating scenarios for training and test data
    scenarios_data = PricingUtils.create_scenarios(scenario_configuration, dataset_code)

    # creating training and test data using BS
    output_priced_bsm = EurOptBS.df_pricing_bsm(scenarios_data)

    # storing training dataset or test dataset
    output_priced_bsm.to_csv(f'training_datasets\\bsm_dataset_{dataset_code}.xlsx', index=False)
    print(f'CSV file was created on training_datasets with name bsm_dataset_{dataset_code}.xlsx.')