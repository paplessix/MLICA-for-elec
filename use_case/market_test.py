from mlica_for_elec.env import *
from mlica_for_elec.market import *

import matplotlib.pyplot as plt
import numpy as np
# Create market instance and test orders

market = Market()


# import profile


import json

household_1 = json.load(open("config\household\default_household.json"))

house = HouseHold(household_1)

generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
consumption_path = "data\consumption\Residential.csv"
spot_price_path = "data/spot_price/2020.csv"
fcr_price_path = "data/fcr_price/random_fcr.csv"
for i in tqdm(range (5)):
    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path)
    house.param["battery"]["power"] = np.random.randint(1,10)
    house.param["generation"]["max_generation"] = np.random.randint(1,10)
    house.param["battery"]["enabled"] = np.random.randint(0,2)
    house.param["battery"]["fcr_enabled"] = np.random.randint(0,2)
    print(f"Customer {i} has battery {house.param['battery']['enabled']} and fcr {house.param['battery']['fcr_enabled']}")
    for _ in range(i+200):
        house.next_data()
    SW = house.get_value_function((0,11))
    if house.param["generation"]["type"] =="solar":
        house.param["generation"]["type"] = "wind"
    else:
        house.param["generation"]["type"] ="solar"
    prices =np.array(list(map(lambda x:x[1],SW)))
    quantities = np.array(list(map(lambda x:x[0],SW)))
    marginal_prices = prices[1:]-prices[:-1]
    for p in marginal_prices:
        market.AddOrder(Order(CreatorID=i, TimeSlot=1,Side=True, Quantity=1, Price=-p))
market.close_gate()
market.plot_orders()
market.ClearMarket(20)
market.LMP_payments()
market.VCG_payments()
market.report_clearing()
market.plot_clearing()
market.plot_clearing_per_participant()
market.plot_payments()



