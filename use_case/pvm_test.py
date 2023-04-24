from mlica_for_elec.env import *
from mlica_for_elec.pvm import *



print("Start loading household profiles")
folder_path = "config\household_profile\\"
houses = []
for file in os.listdir(folder_path)[:10]:
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

pvm