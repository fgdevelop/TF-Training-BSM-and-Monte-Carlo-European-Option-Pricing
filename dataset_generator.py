from class_bsm_mc_opt_price import EurOptBS, PricingUtils
from sklearn.model_selection import train_test_split

from training_utils import TrainUtils

if __name__ == '__main__':
    # inputs to create training data
    scenario_configuration_1 = {"S0": [45, 120, 5],
                              "K": [50, 155, 5],
                              "vol": [0.1, 2.1, .1],
                              "r": [.0, .1, .01],
                              "Dt": [0.1, 2.1, 0.1]}

    # use this to build new scenarios in the future
    base_scenario = {"opt_type": 1, "S0": 48.0, "K": 46.0, "vol": 0.35, "r": 0.124, "Dt": 0.1627}
    base_scenario_1 = {"opt_type": 0, "S0": 32.0, "K": 28.0, "vol": 0.40, "r": 0.11, "Dt": 0.1191}
    base_scenario_2 = {"opt_type": 1, "S0": 100.0, "K": 104.0, "vol": 0.20, "r": 0.15, "Dt": 0.12}

    # creating training data
    scenarios_data = PricingUtils.create_scenarios(scenario_configuration_1)
    output_priced_bsm = EurOptBS.df_pricing_bsm(scenarios_data)

    # storing training dataset or test dataset
    code = TrainUtils.random_code_generator(3)
    output_priced_bsm.to_csv(f'training_datasets\\bsm_dataset_{code}.xlsx', index=False)