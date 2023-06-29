import torch 
import logging
import random 
from mlca_for_elec.networks.main import eval_config
import os
import matplotlib.pyplot as plt
from mlca_for_elec.env.env import Microgrid, HouseHold
from mlca_for_elec.mlca_elec.mlca import *
import json
import numpy as np
from collections import defaultdict
import optuna
import pandas as pd

# Choose experiment number

exp_number = 1

# Run parameter range 

list_of_Qinit = [50,75, 100, 200]
list_of_seed = list(range(1,6))


# set path to relevant data

household_path = f"config\experiment{exp_number}\households"
microgrid_path = f"config\experiment{exp_number}\microgrid\exp{exp_number}_microgrid.json"
dataset_path = f"config\experiment{exp_number}\dataset"

# Load Microgrid 

print("Start loading household profiles")
folder_path = household_path
houses = []

microgrid_1 =json.load(open( microgrid_path))


for file in os.listdir(folder_path)[:5]:
    if file.endswith(".json"):
        household = json.load(open(folder_path+"/"+ file))
    house = HouseHold(household, microgrid_1["horizon"])

    generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
    consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
    spot_price_path = "data/spot_price/2020.csv"
    fcr_price_path = "data/fcr_price/random_fcr.csv"
    profile_path_train = dataset_path + f"/dataset_{house.ID}.csv"
    profile_path_valtest = dataset_path + f"/test_dataset_{house.ID}.csv"
    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path, profile_path_train, profile_path_valtest,type = float)
    for i in range(1):
        house.next_data()
    houses.append(house)
print(f"Loaded {len(houses)} households")
print("Start compute social welfare")
print(list(houses[0].data['consumption'].to_numpy()))

MG = Microgrid(houses, microgrid_1)
optimal_allocation = {}
for house in MG.households:
    print(house.data['consumption'].sum())
    optimal_allocation_tuple = MG.get_efficient_allocation()
    optimal_allocation[house.ID] = (optimal_allocation_tuple[0][house.ID] , MG.calculate_value(house.ID, optimal_allocation_tuple[0][house.ID]))


for seed in list_of_seed : 
    for nitems in list_of_Qinit : 
        layer= "CALayerReLUProjected"
        Qinit =nitems
        Qmax = nitems + 50
        Qround=3
        L=3000

        sample_weight_on = False
        sample_weight_scaling = None
        min_iteration = 1
        seed_instance = seed
        model_name = 'MVNN'
        Mip_bounds_tightening = "IA"
        warm_start=False
        NN_parameters = {f"Bidder_{i}" : {} for i in range(len(MG.households))}
        loacal_scaling_factor = 1



        NN_parameters = defaultdict(dict)


        bidder_id =1 # TODO : change bidder id
        study = optuna.load_study(study_name=f"Exp_{exp_number}_bidder{bidder_id}_nitems_{nitems}_layer_{layer}", storage="sqlite:///db_sqlite.db")
        config_dict = study.best_params
        print(config_dict)
        # add relevant parameters

        config_dict["ts"] = [1,1]
        config_dict["loss_func"] = "F.l1_loss"
        config_dict["state_dict"] = {0 : None, 1 : None, 2 :None, 3 : None, 4: None, 5: None} 
        # base parameters

        normalize_factor = 1





        # {0 : None, 1 : None, 2 :None} 
        #  {0 : "config\experiment1\model_0.pt", 1 : "config\experiment1\model_1.pt", 2 : "config\experiment1\model_2.pt"}
        for house in MG.households:
            for key, value in config_dict.items():
                if key == "state_dict":
                    NN_parameters[f"Bidder_{house.ID}"]['state_dict'] = value[house.ID]
                else:    
                    NN_parameters[f"Bidder_{house.ID}"][key] = value
            
            NN_parameters[f"Bidder_{house.ID}"]['layer_type'] = config_dict["model"]    

            NN_parameters[f"Bidder_{house.ID}"]['num_hidden_units'] = int(max(1, np.round(
                NN_parameters[f"Bidder_{house.ID}"]['num_neurons'] / NN_parameters[f"Bidder_{house.ID}"]['num_hidden_layers'])))
            NN_parameters[f"Bidder_{house.ID}"].pop('num_neurons')

        # NN_parameters = value_model.parameters_to_bidder_id(NN_parameters)


        MIP_parameters = {
                'bigM': 3000,
                'mip_bounds_tightening': None,
                'warm_start': False,
                'time_limit' :300,
                'relative_gap': 1e-2,
                'integrality_tol': 1e-6,
                'attempts_DNN_WDP': 5
            }

        print("=====================================")
        print(f"Start experiment {exp_number} with seed {seed_instance} and Qinit {Qinit} and layer {layer}")
        
        res_path = f"results/Exp_{exp_number}_seed_{seed_instance}_Qinit_{nitems}_Qround_{Qround}_layer_{layer}.json"

        if res_path[8:] in os.listdir("results"):
            print("Already computed")
            continue

        RESULT = mlca_mechanism(value_model = MG, 
            SATS_auction_instance_seed=seed_instance,
            
            Qinit = Qinit,
            Qmax = Qmax,
            Qround = Qround,
            MIP_parameters=MIP_parameters,
            NN_parameters=NN_parameters,
            res_path =res_path,
            scaler=None,
            calc_efficiency_per_iteration=True, 
            local_scaling_factor=loacal_scaling_factor,
            return_payments = True,
            parallelize_training=True
            )

        print(RESULT[1]['MLCA Payments'])