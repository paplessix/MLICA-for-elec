from mlca_for_elec.env.env import *
from mlca_for_elec.mlca_elec.mlca import *
from mlca_for_elec.mlca_elec.mlca_util import create_value_model, problem_instance_info

import argparse
import json
import logging
import os
from collections import defaultdict
from functools import partial

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
Qinit =200
Qmax=202
L=30000
sample_weight_on = False
sample_weight_scaling = None
min_iteration = 1
seed_instance = 12
epochs = 150
batch_size = 32
regularization_type = 'l1'  # 'l1', 'l2' or 'l1_l2'
model_name = 'PlainNN'
Mip_bounds_tightening = "IA"
warm_start=False
parameters = {f"Bidder_{i}" : {} for i in range(len(MG.households))}

NN_parameters = defaultdict(dict)


base ={"batch_size": 1,
        "epochs":200,
        "l2": 1e-5,
        "loss_func": "F.l1_loss",
        "lr": 0.0001,
        "num_hidden_layers":3,
        "num_neurons":160,
        "optimizer": "Adam"
    }


for house in MG.households:
    for key, value in base.items():
        NN_parameters[f"Bidder_{house.ID}"][key] = value
    
    NN_parameters[f"Bidder_{house.ID}"]['layer_type'] = 'PlainNN'

    NN_parameters[f"Bidder_{house.ID}"]['num_hidden_units'] = int(max(1, np.round(
        NN_parameters[f"Bidder_{house.ID}"]['num_neurons'] / NN_parameters[f"Bidder_{house.ID}"]['num_hidden_layers'])))
    NN_parameters[f"Bidder_{house.ID}"].pop('num_neurons')

# NN_parameters = value_model.parameters_to_bidder_id(NN_parameters)


MIP_parameters = {
        'bigM': 20000,
        'mip_bounds_tightening': None,
        'warm_start': False,
        'time_limit' :300,
        'relative_gap': 1e-2,
        'integrality_tol': 1e-6,
        'attempts_DNN_WDP': 5
    }



RESULT = mlca_mechanism(value_model = MG, 
    
    Qinit = Qinit,
    Qmax = Qmax,
    Qround = 1,
    MIP_parameters=MIP_parameters,
    NN_parameters=NN_parameters,
    scaler=None,

    )
