import argparse
import json
import random
import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
from mlca_for_elec.networks.main import eval_config
from mlca_for_elec.env.env import Microgrid, HouseHold

network_type_to_layer_type = {
    'MVNN': 'CALayerReLUProjected',
    'NN': 'PlainNN'
}

def evaluate_network(cfg: dict, seed: int, MicroGrid_instance: str, bidder_id: str, num_train_data: int, layer_type: str,
                     normalize: bool, normalize_factor: float, eval_test=False, save_datasets=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return eval_config(
        seed=seed, SAT_instance=MicroGrid_instance, bidder_id=bidder_id,
        layer_type=layer_type, batch_size=cfg['batch_size'], num_hidden_layers=cfg['num_hidden_layers'],
        num_hidden_units=int(max(1, np.round(cfg['num_neurons'] / cfg['num_hidden_layers']))), l2=cfg['l2'], l1 =cfg['l1'],
        lr=cfg['lr'], normalize_factor=normalize_factor, optimizer=cfg['optimizer'], num_train_data=num_train_data,
        eval_test=True, epochs=cfg['epochs'], loss_func=cfg['loss_func'], normalize=normalize, save_datasets=False, log_path="logs", ts = cfg["ts"], state_dict = cfg["state_dict"], plot=True)



if __name__=="__main__":
    print("Start loading household profiles")

    exp_number = 1

    household_path = f"config\experiment{exp_number}\households"
    microgrid_path = f"config\experiment{exp_number}\microgrid\exp{exp_number}_microgrid.json"
    dataset_path = f"config\experiment{exp_number}\dataset"


    houses = []
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

        
    # MG.generate_dataset(0)
    config_dict = {"batch_size": 3,
          "epochs":300,
          "l2": 1e-6,
          "l1": 1e-10,
          "loss_func": "F.l1_loss",
          "lr": 0.0001,
          "num_hidden_layers":2,
          "num_neurons": 90,
          "optimizer": "Adam", 
          "state_dict" : None,
          "ts": [1,10]
        }

    print('Selected hyperparameters', config_dict)
    model, logs = evaluate_network(
        config_dict, seed=0, MicroGrid_instance=MG, bidder_id=1,
        num_train_data=300, layer_type="CALayerReLUProjected",
        normalize=True,
        normalize_factor=1)
    train_logs = logs['metrics']['train'][config_dict['epochs']]
    val_logs = logs['metrics']['val'][config_dict['epochs']]
    test_logs = logs['metrics']['test'][config_dict['epochs']]
    
    torch.save(model.state_dict(), "model/test.pt")
    print('Train metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(train_logs['r'], train_logs['kendall_tau']))
    print('Valid metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(val_logs['r'], val_logs['kendall_tau']))
    print('Test metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(test_logs['r'], test_logs['kendall_tau']))

    