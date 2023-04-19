import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mlica_for_elec.util import *
from pyomo.environ import *
import pyomo.kernel as pmo
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

class HouseHold():
    def __init__(self, param) -> None:
        self.param = param
        
        self.data = None
        self.result = None

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
        self.generator_data = ( df.iloc[24*i:24*(i+1), : ] for i in range(365) )
        
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
        cost =sum_product(model.SpotPrice, model.Grid_power) + sum_product(model.NonServedCost, model.Non_served_consumption) - sum_product(model.FCRPrice, model.FCR)
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
        


    # Extract results 
 
    def get_welfare(self):
        
        return sum_product(self.model.NonServedCost, self.model.Non_served_consumption)()

    def get_planning(self):
        pass

    def get_bids(self):
        self.build_milp()
        self.run_milp()
        bids = []
        for time in self.result.index:
            bids.append({"time":time, "qtty":self.result.loc[time,"Grid_power"], "price":self.result.loc[time,"non_served_cost"]})
        
        return bids

    def get_value_function(self, range):
        SW =  []
        for capa in np.arange(range[0], range[1]):
            self.set_loadlimit(capa)
            self.build_milp()
            self.run_milp()
            SW.append((capa,self.get_welfare()))
        return SW
    

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




if __name__=="__main__":

    import json
    household_1 = json.load(open("config\household_profile\default_household.json"))

    house = HouseHold(household_1)
    generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
    consumption_path = "data\consumption\Reference-Residential.csv"
    spot_price_path = "data/spot_price/2020.csv"
    fcr_price_path = "data/fcr_price/random_fcr.csv"

    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path)
    house.next_data()
    house.next_data()
    house.next_data()
    house.build_milp()
    house.run_milp()

    house.display_planning()
