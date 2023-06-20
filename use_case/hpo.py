import optuna
import numpy as np
import torch
import random
import json
from mlca_for_elec.env.env import Microgrid, HouseHold
from mlca_for_elec.networks.main import eval_config 
import os 

def evaluate_network(cfg: dict, seed: int, MicroGrid_instance: str, bidder_id: str, num_train_data: int, layer_type: str,
                     normalize: bool, normalize_factor: float, eval_test=False, save_datasets=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return eval_config(
        seed=seed, SAT_instance=MicroGrid_instance, bidder_id=bidder_id,
        layer_type=layer_type, batch_size=cfg['batch_size'], num_hidden_layers=cfg['num_hidden_layers'],
        num_hidden_units=int(max(1, np.round(cfg['num_neurons'] / cfg['num_hidden_layers']))), l2=cfg['l2'], l1 = cfg['l1'],
        lr=cfg['lr'], normalize_factor=normalize_factor, optimizer=cfg['optimizer'], num_train_data=num_train_data,
        eval_test=False, epochs=cfg['epochs'], loss_func=cfg['loss_func'], normalize=normalize, save_datasets=False, log_path="logs", state_dict = cfg["state_dict"])




def objective(trial, MG_instance, num_train_data, bidder_id, layer):
    model = trial.suggest_categorical("model", [layer])
    batch_size = trial.suggest_int("batch_size", 1, 10)
    epochs = trial.suggest_int("epochs", 1, 500)
    l2 = trial.suggest_float("l2", 1e-12, 1e-2, log=True)
    l1= trial.suggest_float("l1", 1e-12, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-8, 1e-3,log = True)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    num_neurons = trial.suggest_int("num_neurons", 1, 100)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])


    config_dict = {"batch_size": batch_size,
          "epochs":epochs,
          "l2": l2,
          "l1":l1,
          "loss_func": "F.l1_loss",
          "lr": lr,
          "num_hidden_layers":  num_hidden_layers,
          "num_neurons": num_neurons,
          "optimizer": optimizer,
          "state_dict" : None
        }
    
    model, logs = evaluate_network(
            config_dict, seed=0, MicroGrid_instance=MG_instance, bidder_id= bidder_id,
            num_train_data=num_train_data, layer_type=model,
            normalize=True,
            normalize_factor=1)
    train_logs = logs['metrics']['train'][config_dict['epochs']]
    val_logs = logs['metrics']['val'][config_dict['epochs']]
    # test_logs = logs['metrics']['test'][config_dict['epochs']]

    return val_logs["mae"] 


if __name__=="__main__":

    exp_number =2
    bidder_id = 1
    num_train_data =200

    layer  = "PlainNN"

    household_path = f"config\experiment{exp_number}\households"
    microgrid_path = f"config\experiment{exp_number}\microgrid\exp{exp_number}_microgrid.json"
    dataset_path = f"config\experiment{exp_number}\dataset"
    
    # Load MicroGrid
    print("Start loading household profiles")
    folder_path = household_path
    houses = []
    for file in os.listdir(folder_path)[:3]:
        if file.endswith(".json"):
            household = json.load(open(folder_path+"/"+ file))
        house = HouseHold(household)

        generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
        consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
        spot_price_path = "data/spot_price/2020.csv"
        fcr_price_path = "data/fcr_price/random_fcr.csv"
        profile_path_train = dataset_path + f"/dataset_{house.ID}.csv"
        profile_path_valtest = dataset_path + f"/test_dataset_{house.ID}.csv"
        house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path, profile_path_train, profile_path_valtest,type = float)
        for i in range(3):
            house.next_data()
        houses.append(house)
        print(f"Loaded {len(houses)} households")
        print("Start compute social welfare")
        print(houses[0].data['consumption'])
        microgrid_1 =json.load(open( microgrid_path))
        MG = Microgrid(houses, microgrid_1)
        for house in MG.households:
            print(house.data['consumption'].sum())



    study = optuna.create_study(storage = "sqlite:///db_sqlite.db", study_name = f"Exp_{exp_number}_bidder{bidder_id}_nitems_{num_train_data}_layer_{layer}", load_if_exists=True)  # Create a new study.
    study.optimize(lambda trial : objective(trial,MG,num_train_data,bidder_id, layer), n_trials=60)  # Invoke optimization of the objective function.e