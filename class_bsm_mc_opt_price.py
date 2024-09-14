import pandas as pd
import numpy as np
import random
from numpy.random import standard_normal
from scipy.stats import norm
from typing import List, Dict

class EurOptBS:
    def __init__(self, opt_type: int, S0: float, K: float, vol: float, r: float, Dt: float):
        self.opt_type = opt_type
        self.S0 = S0
        self.K = K
        self.vol = vol
        self.r = r
        self.Dt = Dt

    def bsm_opt_price(self) -> float:
        """ Calculates the european option price according to Black Scholes Merton model."""
        d1 = (np.log(self.S0 / self.K) + (self.r + self.vol ** 2 / 2) * self.Dt) / (self.vol * np.sqrt(self.Dt))
        d2 = d1 - self.vol * np.sqrt(self.Dt)
        if self.opt_type == 1:  # call
            C = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.Dt) * norm.cdf(d2)
            return C
        elif self.opt_type == 0:  # put
            P = self.K * np.exp(-self.r * self.Dt) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
            return P
        else:
            raise Exception(f"There's no such type {self.opt_type}.")

    @staticmethod
    def df_pricing_bsm(df_scenarios: pd.DataFrame) -> pd.DataFrame:
        df_scenarios_bsm = df_scenarios.copy()
        df_scenarios_bsm["price"] = np.zeros(len(df_scenarios_bsm))
        for index, row in df_scenarios_bsm.iterrows():
            option = EurOptBS(row["opt_type"], row["S0"], row["K"], row["vol"], row["r"], row["Dt"])
            price_bsm = option.bsm_opt_price()
            df_scenarios_bsm.at[index, "price"] = price_bsm
        return df_scenarios_bsm


class EurOptMC:
    def __init__(self, opt_type: int, S0: float, K: float, vol: float, r: float, Dt: float):
        self.opt_type = opt_type
        self.S0 = S0
        self.K = K
        self.vol = vol
        self.r = r
        self.Dt = Dt

    @staticmethod
    def S_price(rand: float, Dt: float, r: float, vol: float, S0: float) -> float:
        return S0 * np.exp((r - vol ** 2 / 2) * Dt + vol * np.sqrt(Dt) * rand)

    @staticmethod
    def calculate_asset_prices(rand_vec: np.array, Dt: float, r: float, vol: float, S0: float) -> np.array:
        S = np.zeros(len(rand_vec))
        for i in range(0, len(rand_vec)):
            S[i] = EurOptMC.S_price(float(rand_vec[i]), Dt, r, vol, S0)
        return S

    @staticmethod
    def calculate_intrinsic_value(opt_type: int, K: float, S_vec: np.array) -> np.array:
        VI = np.zeros(len(S_vec))
        if opt_type == 1: # call
            for i in range(0, len(S_vec)):
                VI[i] = max(S_vec[i] - K, 0)
        elif opt_type == 0: # put
            for i in range(0, len(S_vec)):
                VI[i] = max(K - S_vec[i], 0)
        else:
            raise Exception(f"There's no option type such as {opt_type}.")
        return VI

    @staticmethod
    def calculate_opt_price(Dt: float, r: float, VI_vec: np.array) -> float:
        return np.exp(-r * Dt) * (VI_vec.sum() / len(VI_vec))

    def mc_opt_price(self, simulations: int) -> float:
        """ Calculates european option price according to Hull White model through Monte Carlo simulation."""
        random_numbers = standard_normal(size=simulations)
        S = EurOptMC.calculate_asset_prices(random_numbers, self.Dt, self.r, self.vol, self.S0)
        VI = EurOptMC.calculate_intrinsic_value(self.opt_type, self.K, S)
        return EurOptMC.calculate_opt_price(self.Dt, self.r, VI)

    @staticmethod
    def df_pricing_mc(df_scenarios: pd.DataFrame, simulations: int) -> pd.DataFrame:
        df_scenarios_mc = df_scenarios.copy()
        df_scenarios_mc["price"] = np.zeros(len(df_scenarios_mc))
        for index, row in df_scenarios_mc.iterrows():
            option = EurOptMC(row["opt_type"], row["S0"], row["K"], row["vol"], row["r"], row["Dt"])
            price_mc = option.mc_opt_price(simulations)
            df_scenarios_mc.at[index, "price"] = price_mc
        return df_scenarios_mc

class PricingUtils:

    @staticmethod
    def create_scenarios(scenario_config: Dict, code: str):

        S0_list = np.arange(scenario_config['S0'][0], scenario_config['S0'][1], scenario_config['S0'][2])
        K_list = np.arange(scenario_config['K'][0], scenario_config['K'][1], scenario_config['K'][2])
        vol_list = np.arange(scenario_config['vol'][0], scenario_config['vol'][1], scenario_config['vol'][2])
        r_list = np.arange(scenario_config['r'][0], scenario_config['r'][1], scenario_config['r'][2])
        Dt_list = np.arange(scenario_config['Dt'][0], scenario_config['Dt'][1], scenario_config['Dt'][2])

        scenarios = []
        for r in r_list:
            for K in K_list:
                for Dt in Dt_list:
                    for vol in vol_list:
                        for S0 in S0_list:
                            call_put = random.choice([int(0), int(1)])
                            scenario = {
                                "opt_type": call_put,
                                "S0": S0,
                                "K": K,
                                "vol": vol,
                                "r": r,
                                "Dt": Dt,
                            }
                            scenarios.append(scenario)

        scenarios_df = pd.DataFrame(scenarios)
        print(f'The dataset containing the scenarios with code {code} was created.')
        return scenarios_df
