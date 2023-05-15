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
        num_hidden_units=int(max(1, np.round(cfg['num_neurons'] / cfg['num_hidden_layers']))), l2=cfg['l2'],
        lr=cfg['lr'], normalize_factor=normalize_factor, optimizer=cfg['optimizer'], num_train_data=num_train_data,
        eval_test=True, epochs=cfg['epochs'], loss_func=cfg['loss_func'], normalize=normalize, save_datasets=False, log_path="logs")




if __name__=="__main__":
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
        for i in range(205):
            house.next_data()
        houses.append(house)
    print(f"Loaded {len(houses)} households")
    print("Start compute social welfare")
    print(houses[0].data['consumption'])
    houses[0].data['consumption'].plot(xlabel="Time", ylabel="Consumption (kWh)")
    plt.show()
    microgrid_1 =json.load(open("config\microgrid_profile\default_microgrid.json"))
    MG = Microgrid(houses, microgrid_1)
    # MG.generate_dataset(0)
    config_dict = {"batch_size": 1,
          "epochs":300,
          "l2": 1e-6,
          "loss_func": "F.l1_loss",
          "lr": 0.0001,
          "num_hidden_layers":2,
          "num_neurons": 150,
          "optimizer": "Adam"
        }

    print('Selected hyperparameters', config_dict)
    model, logs = evaluate_network(
        config_dict, seed=0, MicroGrid_instance=MG, bidder_id=1,
        num_train_data=800, layer_type="CALayerReLUProjected",
        normalize=True,
        normalize_factor=1)
    train_logs = logs['metrics']['train'][config_dict['epochs']]
    val_logs = logs['metrics']['val'][config_dict['epochs']]
    test_logs = logs['metrics']['test'][config_dict['epochs']]

    print('Train metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(train_logs['r'], train_logs['kendall_tau']))
    print('Valid metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(val_logs['r'], val_logs['kendall_tau']))
    print('Test metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(test_logs['r'], test_logs['kendall_tau']))

    