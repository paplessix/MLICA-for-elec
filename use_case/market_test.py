from mlca_for_elec.env.env import *
from mlca_for_elec.env.market import *

import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
#################3

def induce_elasticity(bids, elasticity):
        new_bids = []
        for bid in bids:
            remaining_qtty = bid["qtty"]
            running_price =0
            while remaining_qtty > 1 :
                running_price += bid["price"]* (- elasticity)
                remaining_qtty-=1
                if running_price < bid["price"]:
                          new_bid= {"time":bid["time"], "qtty":1, "price":running_price, "node" : bid["node"]}
                          new_bids.append(new_bid)
                else:
                    break
            new_bid= {"time":bid["time"], "qtty":remaining_qtty, "price": bid["price"], "node" : bid["node"]}
            new_bids.append(new_bid)
        return new_bids


# Create market instance and test orders

market = Market("config\microgrid_profile/non_constrained_microgrid.json")


# import profile


import json

print("Start loading household profiles")
folder_path = "config\household_profile\\"
houses = []
for file in os.listdir(folder_path)[:5]:
    if file.endswith(".json"):
        household = json.load(open(folder_path+"/"+ file))
    house = HouseHold(household)
    generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
    consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
    spot_price_path = "data/spot_price/2020.csv"
    fcr_price_path = "data/fcr_price/random_fcr.csv"
    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path)
    house.next_data()
    houses.append(house)
print(f"Loaded {len(houses)} households")
print("Start compute social welfare")


for i,house in tqdm(enumerate(houses)):
    bids = house.get_bids()
    ELASTICITY = -0.11
    bids = induce_elasticity(bids,ELASTICITY)
    for bid in bids:
        market.AddOrder(Order(CreatorID=i, Side=True, TimeSlot=bid["time"], Quantity=bid["qtty"], Price=  bid["price"], Node=bid["node"]))
       
spot_prices =  houses[0].get_spot_price()
market.close_gate()
market.plot_orders()
market.ClearMarket(spot_prices)
market.LMP_payments()
market.VCG_payments()
market.report_clearing()
market.plot_clearing()

market.plot_payments()



