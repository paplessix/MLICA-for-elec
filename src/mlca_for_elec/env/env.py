import os 
import json
import pandas as pd
import itertools as iter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mlca_for_elec.env.util_env import *
from pyomo.environ import *
import pyomo.kernel as pmo
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

class HouseHold():
    def __init__(self, param) -> None:
        self.param = param
        self.ID = self.param["ID"]
        self.node =self.param["grid"]["node"]
        self.data = None
        self.result = None
        self.horizon = 3

    def load_data(self, generation_path, consumption_path, spot_price_path, fcr_price_path):
        
        if self.param["generation"]["type"] == "wind":
            preprocessor_generation = preprocessor_wind
            column_name ="WS10m"
        elif self.param["generation"]["type"] == "solar":
            preprocessor_generation = preprocessor_solar
            column_name ="P"
        else:
            raise Exception("Generation type not supported")
        
        # Load data

        data_generation = preprocessor_generation(pd.read_csv(generation_path)[column_name], self.param["generation"]["max_generation"])
        data_consumption = preprocessor_consumption(pd.read_csv(consumption_path)["Power [kW]"], self.param["consumption"]["max_consumption"])
        data_fcr_price = preprocessor_fcr_price(pd.read_csv(fcr_price_path)["FCR price"])
        data_spot_price = preprocessor_spot_price(pd.read_csv(spot_price_path)["price_euros_mwh"])

        df = pd.concat([data_generation, data_consumption, data_fcr_price, data_spot_price], axis=1)
        
        df["non_served_cost"] = self.param["consumption"]["cost_of_non_served_energy"]
        self.generator_data = ( df.iloc[self.horizon*i:self.horizon*(i+1), : ].reset_index(drop=True) for i in range(365) )
        
        self.data = next(self.generator_data)

    def next_data(self):
        try:
            self.data = next(self.generator_data)
        except StopIteration:
            print("End of data")
            self.data = None
        return self.data

    def display_data(self):
        self.data.plot( kind="bar",subplots=True, figsize=(10,10))
        plt.legend()
        plt.show()
        

    # Get inputs 

    def get_specs(self):
        pass

    def display_inputs(self):
        pass
        
    def __str__(self) -> str:
        return str(self.param)
    
    # Set inputs

    def set_loadlimit(self, loadlimit):
        self.param["grid"]["maxload"] = loadlimit
    # Objective function

    def cost_function(self,model):
        # cost =sum_product(model.SpotPrice, model.Grid_power) + sum_product(model.NonServedCost, model.Non_served_consumption) - sum_product(model.FCRPrice, model.FCR)
        cost = sum_product(model.NonServedCost, model.Non_served_consumption) 
        return cost
    
    #    Run 
    def build_milp(self):
        self.model =   ConcreteModel()

        # defining components of the objective model

        # General Parameters
        self.model.Period = RangeSet(self.data.index[0], self.data.index[-1])

        # Household parameters
        self.model.SpotPrice = Param(self.model.Period,initialize=self.data.spot_price, within=Any)
        self.model.FCRPrice = Param(self.model.Period,initialize=self.data.fcr_price, within=Any)
        self.model.Consumption = Param(self.model.Period, initialize=self.data.consumption, within=Any)
        self.model.Generation = Param(self.model.Period, initialize=self.data.generation, within=Any)
        self.model.NonServedCost = Param(self.model.Period, initialize=self.data.non_served_cost, within=Any)
        
        
        # FCR variables 

        self.model.FCR = Var(self.model.Period,initialize=0, bounds=(0, np.inf)) # Power subscribed on the FCR market
        self.model.FCR_on = Var(self.model.Period, within=Binary) # Binary variable to know if the FCR is on or off
        self.model.FCR_enabled = Param(initialize=self.param["battery"]["fcr_enabled"], within=Binary) # Binary variable to know if the FCR is enabled or not
        
        # battery variables
        self.model.battery_enabled = Param(initialize=self.param["battery"]["enabled"], within=Binary)
        self.model.SoC = Var(self.model.Period)
        self.model.Charge_power = Var(self.model.Period,initialize=0, bounds=( -self.param["battery"]["power"],0))
        self.model.Discharge_power = Var(self.model.Period,initialize=0, bounds=(0, self.param["battery"]["power"]))
        self.model.Grid_power = Var(self.model.Period, bounds=(0, np.inf))
        self.model.Non_served_consumption = Var(self.model.Period, bounds=(0, np.inf))
        self.model.Curtailed_generation = Var(self.model.Period, initialize=0, bounds=(0, np.inf))
        self.model.charge_on = Var(self.model.Period, within=Binary)
        self.model.discharge_on= Var(self.model.Period, within=Binary)

        # Set constraint for the household

        # Kirchoff law
        def houshold_node(model, i):
            return model.Grid_power[i] + model.Discharge_power[i] + model.Generation[i]- model.Curtailed_generation[i]  == -model.Charge_power[i] + model.Consumption[i] - model.Non_served_consumption[i]

        # Grid max load 

        def grid_maxload(model, i):
            return model.Grid_power[i] + self.model.FCR[i] <= self.param["grid"]["maxload"]
        
        # Battery related constraints defintions

        def SoC_definition(model, i):
            # Assigning battery's starting capacity at the beginning
            if i ==model.Period.at(1):
                prev_Soc = self.param["battery"]["init_soc"]
            else : 
                prev_Soc = model.SoC[i-1]

            return model.SoC[i] == (prev_Soc + ( (-model.Charge_power[i] * np.sqrt(self.param["battery"]["rte"])) - (model.Discharge_power[i] / np.sqrt(self.param["battery"]["rte"])))/(self.param["battery"]["duration"]*self.param["battery"]["power"]))

        def SoC_bound_upper(model, i):
            return model.SoC[i] <= self.param["battery"]["max_soc"]*(1-model.FCR_on[i])+ model.FCR_on[i]*self.param["battery"]["soc_fcr"]

        def SoC_bound_lower(model, i):
            return model.SoC[i] >= self.param["battery"]["min_soc"]*(1-model.FCR_on[i])+ model.FCR_on[i]*self.param["battery"]["soc_fcr"]
        

        def SoC_termination(model):
            return model.SoC[model.Period.at(-1)] == self.param["battery"]["init_soc"]
        

        def battery_charge_on(model, i):
            return model.Charge_power[i] +self.model.FCR[i]*1.25 <= model.charge_on[i] * self.param["battery"]["power"]*self.model.battery_enabled
        
        def battery_discharge_on(model, i):
            return model.Discharge_power[i]+self.model.FCR[i]*1.25 <= model.discharge_on[i] * self.param["battery"]["power"]*self.model.battery_enabled
              
        def battery_twosided_constraint(model, i):
            return model.Charge_power[i] + model.Discharge_power[i] <=1


        # FCR related constraints defintions

        def FCR_on(model, i):
            return model.FCR[i] <= model.FCR_on[i] * self.model.FCR_enabled*self.param["battery"]["power"]/1.25
        
        def FCR_4h_bid(model, i):
            if i%4== 0 :
                return Constraint.Feasible
            else : 
                return model.FCR[i] == model.FCR[i-1]
            
        

        # Set constraint and objective for the battery
        # BESS related constraints
        
        self.model.SoC_definition = Constraint(self.model.Period, rule=SoC_definition)
        self.model.SoC_termination = Constraint( rule=SoC_termination)
        self.model.charge_on_cst = Constraint(self.model.Period, rule=battery_charge_on)
        self.model.discharge_on_cst = Constraint(self.model.Period, rule=battery_discharge_on)
        self.model.twosided_constraint = Constraint(self.model.Period, rule=battery_twosided_constraint)
        

        self.model.SoC_bound_upper_cst = Constraint(self.model.Period, rule=SoC_bound_upper)
        self.model.SoC_bound_lower_cst = Constraint(self.model.Period, rule=SoC_bound_lower)

        # FCR related constraints
        self.model.FCR_on_cst = Constraint(self.model.Period, rule=FCR_on)
        self.model.FCR_4h_bid_cst = Constraint(self.model.Period, rule=FCR_4h_bid)

        self.model.household_node_cst = Constraint(self.model.Period, rule=houshold_node)
        self.model.grid_loadmax_cst = Constraint(self.model.Period, rule=grid_maxload)

        self.model.objective = Objective(rule= self.cost_function, sense=minimize)


    def run_milp(self):
        opt = SolverFactory("glpk", executable="solver\glpk\glpsol.exe")
        opt.options['tmlim'] = 5
        opt.solve(self.model)

        # unpack results
        index,charge_power, discharge_power, SoC, Grid_power,non_served, fcr_power= ([] for i in range(7))

        for i in self.model.Period:
            index.append(i)
            charge_power.append(self.model.Charge_power[i].value)
            discharge_power.append(self.model.Discharge_power[i].value)
            SoC.append(self.model.SoC[i].value)
            Grid_power.append(self.model.Grid_power[i].value)
            non_served.append(max(0,self.model.Non_served_consumption[i].value))
            fcr_power.append(self.model.FCR[i].value)

        self.result = pd.concat((self.data,pd.DataFrame({"index":index,'SoC':SoC, 'charge_power':charge_power,
                           'discharge_power':discharge_power, 'Grid_power':Grid_power,"non_served":non_served,"fcr_power":fcr_power}).set_index("index")), axis=1)
        


 
    def get_welfare(self):
        return sum_product(self.model.NonServedCost, self.model.Non_served_consumption)()

    def get_spot_price(self):
        return self.data.spot_price

    def get_planning(self):
        pass

    def get_bids(self):
        self.build_milp()
        self.run_milp()
        bids = []
        for time in self.result.index:
            bids.append({"time":time, "node":self.node, "qtty":self.result.loc[time,"Grid_power"], "price":self.result.loc[time,"non_served_cost"]})
        
        return bids

    def get_value_function(self, range):
        SW =  []
        for capa in np.arange(range[0], range[1]):
            self.set_loadlimit(capa)
            self.build_milp()
            self.run_milp()
            SW.append((capa,self.get_welfare()))
        return SW
    
    def get_optimal_welfare(self):
        self.build_milp()
        self.run_milp()
        return self.model.objective.expr()
    # Display results 

    def display_planning(self):
        fig,ax = plt.subplots(2,2, figsize=(10,10))

        self.result[["charge_power","discharge_power","Grid_power","non_served"]].plot(kind="bar", stacked=True, ax=ax[0,0])  
        
        ax[1,0].axhline(y=self.param["battery"]["max_soc"], color='r', linestyle='-')
        self.result[["SoC"]].plot(kind="line", ax=ax[1,0], grid = True,ylim=(0,1.1))
        ax[1,0].axhline(y=self.param["battery"]["min_soc"], color='r', linestyle='-') 
        self.result[["consumption"]].plot(kind="line", ax=ax[0,1])   
        self.result[["generation"]].plot(kind="line", ax=ax[0,1])  
        self.result[["consumption"]].plot(kind="line", ax=ax[1,1])  
        self.result[["Grid_power","non_served","generation","fcr_power"]].plot(kind="area", ax=ax[1,1])
        ax[1,1].axhline(y=self.param["grid"]["maxload"], color='r', linestyle='-')
        
        plt.tight_layout()
        plt.legend()
        plt.show()


class Microgrid():

    def __init__(self, households, param):
        self.param = param
        self.households = households
        self.congestion_matrix = self.param["congestion_matrix"]
        self.grid_connection = self.param["grid_connection"]
        self.grid_nodes = self.param["nodes"]
        self.N_nodes = len(self.grid_nodes)
        self.horizon = self.households[0].horizon
    
    def create_mg(households, param):
        return Microgrid(households, param)

    def build_model(self):
        self.model = ConcreteModel()
        for i,house in enumerate(self.households):
            house.build_milp()
            self.model.add_component(f"house_{i}", house.model)
            self.model.__getattribute__(f"house_{i}").objective.deactivate()
        
        # add grid variables
        self.model.nodes = Set(initialize=self.grid_nodes)
        self.model.Period = Set(initialize=self.households[0].model.Period)
        self.model.flows = Var(self.model.nodes, self.model.nodes, self.model.Period, domain=NonNegativeReals)
        self.model.external_import = Var(self.model.Period, domain=NonNegativeReals)
        # add kirchoff's law

        def kirchoff_rule(model, node,t):
            s = 0
            for i,house in enumerate(self.households):
                if house.node == node:  
                    s+=self.model.__getattribute__(f"house_{i}").Grid_power[t]
            if node == 0 :
                return (s + sum(model.flows[node,j,t] for j in model.nodes if j != node)
                          == sum(model.flows[j,node,t] for j in model.nodes if j != node)
                         + self.model.external_import[t])
            else:
                return (s + sum(model.flows[node,j,t] for j in model.nodes if j != node)
                         == sum(model.flows[j,node,t] for j in model.nodes if j != node))
        

        def capacity_limitation_rule(model, node1,node2, t):
            return model.flows[node1,node2,t] <= self.congestion_matrix[node1][node2]
            
        def limit_import_rule(model, t):
            return model.external_import[t] <= self.grid_connection

        
        
        
        
        def objective_rule(model):
            return sum(model.__getattribute__(f"house_{i}").objective.expr for i in range(len(self.households)))
        
        self.model.obj = Objective(rule=objective_rule, sense=minimize)
        self.model.node_law_constraint = Constraint(self.model.nodes,self.model.Period, rule=kirchoff_rule)
        self.model.limit_import_constraint = Constraint(self.model.Period, rule=limit_import_rule)
        self.model.capacity_limitation_rule = Constraint(self.model.nodes,self.model.nodes,self.model.Period, rule=capacity_limitation_rule)
    
    def run_model(self):
        opt = SolverFactory('glpk')
        result_obj = opt.solve(self.model, tee=True)
        result_dic = {}
        for i,house in enumerate(self.households):
            result_dic[i]=[]
            result_dic["index"]=[]
            for j in self.model.__getattribute__(f"house_{i}").Period:
                result_dic["index"].append(j)
                result_dic[i].append(self.model.__getattribute__(f"house_{i}").Grid_power[j].value)
        self.consumption = pd.DataFrame(result_dic).set_index("index")
        self.consumption["external_import"] = self.model.external_import.get_values()
        trans_dic = {}

        for node in self.model.nodes:
            for node2 in self.model.nodes:
                if node != node2:
                    trans_dic[(node,node2)] = []
                    trans_dic["index"]=[]
                    for t in self.model.Period:    
                        trans_dic["index"].append(t)
                        trans_dic[(node,node2)].append(self.model.flows[node,node2,t].value)
        self.transmission = pd.DataFrame(trans_dic).set_index("index")

    def get_optimal_welfare():
        pass

    def display_gridflows(self):
        fig, ax = plt.subplots(self.N_nodes,self.N_nodes)

        for node1,node2 in iter.product(self.grid_nodes,self.grid_nodes):
            if node2 > node1:
                self.transmission[(node1,node2)].plot(kind="bar", ax=ax[node1,node2])
                (-self.transmission[(node2,node1)]).plot(kind="bar", ax=ax[node1,node2])
                ax[node1,node2].axhline(y=self.congestion_matrix[node1][node2], color='r', linestyle='-')
                ax[node1,node2].axhline(y=-self.congestion_matrix[node2][node1], color='r', linestyle='-')
                ax[node1,node2].set_title(f"Flow from {node1} to {node2}")
        
        for node1 in self.grid_nodes:
            self.transmission[((node1,node2) for node2 in self.grid_nodes if node1!=node2)].plot(kind="bar", ax=ax[node1,0], stacked=True)
            (-self.transmission[((node2,node1) for node2 in self.grid_nodes if node1!=node2)]).plot(kind="bar", ax=ax[node1,0], stacked=True)
            for i,house in enumerate(self.households):
                if house.node == node1:
                    self.consumption[i].plot(kind="bar", ax=ax[node1,0], label=f"house_{i}", color ="red")
            ax[node1,0].set_title(f"Consumption at node {node1}")
            ax[node1,0].legend()
        ax[0,0].axhline(y=self.grid_connection, color='r', linestyle='-')
        plt.show()

    def get_bidder_ids(self):
        return [house.ID for house in self.households]

    def get_good_ids(self):
        return [i for i in range(self.horizon)]
    
    def get_uniform_random_bids(self, bidder_id,number_of_bids,seed=None):
        if seed is not None:
            np.random.seed(seed)
        bids =[np.random.randint(self.households[bidder_id].param["consumption"]["max_consumption"],size=self.horizon) for i in range(number_of_bids)]
        # bids = [np.random.rand(self.horizon)*self.households[bidder_id].param["consumption"]["max_consumption"] for i in range(number_of_bids)]
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res

    def generate_dataset(self,bidder_id):
        bids = self.get_uniform_random_bids(bidder_id,1000)
        df = pd.DataFrame(bids)
        df.rename(columns ={self.horizon:"value"}, inplace=True)
        df.to_csv(f"data/cost_function/dataset_{bidder_id}.csv")

    def get_random_feasible_bundle_set(self):
        self.build_model()
        self.model.obj.deactivate()
        
        def random_objective_rule(model):
            return sum(sum_product(np.random.rand(len(model.__getattribute__(f"house_{i}").Period))/(-10),model.__getattribute__(f"house_{i}").Grid_power) for i in range(len(self.households))) 
        self.model.objective = Objective(rule = random_objective_rule, sense=minimize)
        opt = SolverFactory('glpk')
        result_obj = opt.solve(self.model)
        opt.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        result_dic = {}  
        for i,house in enumerate(self.households):
            result_dic[i]=[]
            result_dic["index"]=[]
            for j in self.model.__getattribute__(f"house_{i}").Period:
                result_dic["index"].append(j)
                result_dic[i].append(self.model.__getattribute__(f"house_{i}").Grid_power[j].value)
        self.consumption = pd.DataFrame(result_dic).set_index("index")      
        return self.consumption.to_numpy()
    
    def get_random_feasible_bundle(self, bidder_id, number_of_bundles):
        pass
    def get_model_name(self):
        return "GridModel"


    def calculate_value(self,bidder_id,bundle):
        # return np.sum(bundle**3)

        self.households[bidder_id].build_milp()
        def grid_exchange_fix(model, i):
            return model.Grid_power[i] == bundle[i]
        self.households[bidder_id].model.grid_exchange_fix = Constraint(self.households[bidder_id].model.Period, rule=grid_exchange_fix)
        self.households[bidder_id].run_milp()
        self.households[bidder_id].model.objective.expr()
        return self.households[bidder_id].model.objective.expr()
            

if __name__=="__main__":
    print("Start loading household profiles")
    folder_path = "config\household_profile/"
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
    print([house.ID for house in houses])
    microgrid_1 =json.load(open("config\microgrid_profile\default_microgrid.json"))
    MG = Microgrid(houses, microgrid_1)
    MG.build_model()
    MG.run_model()
    MG.display_gridflows()
    print(MG.households[0].get_optimal_welfare())

