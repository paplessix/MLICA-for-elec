from mlica_for_elec.env import *
from mlica_for_elec.pvm import *

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # used only for MRVM

# LOG DEBUG TO CONSOLE
logging.basicConfig(level=logging.DEBUG, format='%(message)s', filemode='w')
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

print("Start loading household profiles")
folder_path = "config\household_profile\\"
houses = []
for file in os.listdir(folder_path)[:3]:
    if file.endswith(".json"):
        household = json.load(open(folder_path+"/"+ file))
    house = HouseHold(household)
    generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
    consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
    spot_price_path = "data/spot_price/2020.csv"
    fcr_price_path = "data/fcr_price/random_fcr.csv"
    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path)
    houses.append(house)
print(f"Loaded {len(houses)} households")
print("Start compute social welfare")

microgrid_1 =json.load(open("config\microgrid_profile\default_microgrid.json"))
MG = Microgrid(houses, microgrid_1)
caps = [2,40]
L=3000
sample_weight_on = False
sample_weight_scaling = None
min_iteration = 1
seed_instance = 12
epochs = 300
batch_size = 32
regularization_type = 'l1'  # 'l1', 'l2' or 'l1_l2'
model_name = 'Kernel_Linear'
Mip_bounds_tightening = "IA"
warm_start=False
parameters = {f"Bidder_{i}" : {} for i in range(len(MG.households))}
RESULT = pvm(MG, 
    scaler=False,
    caps = caps,
    L = L,
    parameters = parameters,
    epochs = epochs,
    batch_size = batch_size,
    model_name = model_name,
    sample_weight_on = sample_weight_on,
    sample_weight_scaling = sample_weight_scaling,
    min_iteration = min_iteration,
    seed_instance = seed_instance,
    regularization_type = regularization_type,
    Mip_bounds_tightening = Mip_bounds_tightening,
    warm_start = warm_start
    )
