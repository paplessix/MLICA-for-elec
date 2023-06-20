import os 
import json
import pandas as pd
import itertools as iter
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from mlca_for_elec.env.util_env import *
from pyomo.environ import *
from functools import reduce
from scipy import stats
import pyomo.kernel as pmo
import networkx as nx 
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

class HouseHold():  
    def __init__(self, param, horizon = 6 ) -> None:
        self.param = param
        self.ID = self.param["ID"]
        print(self.ID)
        self.node =self.param["grid"]["node"]
        self.data = None
        self.result = None
        self.horizon = horizon
        self.is_build_milp  = None

    def load_data(self, generation_path, consumption_path, spot_price_path, fcr_price_path, profile_path_train= None, profile_path_valtest=None, type = int):

        self.profile_path_train = profile_path_train
        self.profile_path_valtest = profile_path_valtest
        if self.param["generation"]["type"] == "wind":
            preprocessor_generation = preprocessor_wind
            column_name ="WS10m"
        elif self.param["generation"]["type"] == "solar":
            preprocessor_generation = preprocessor_solar
            column_name ="P"
        else:
            raise Exception("Generation type not supported")
        
        # Load data

        data_generation = preprocessor_generation(pd.read_csv(generation_path)[column_name], self.param["generation"]["max_generation"], type )
        data_consumption = preprocessor_consumption(pd.read_csv(consumption_path)["Power [kW]"], self.param["consumption"]["max_consumption"], type)
        data_fcr_price = preprocessor_fcr_price(pd.read_csv(fcr_price_path)["FCR price"])
        data_spot_price = preprocessor_spot_price(pd.read_csv(spot_price_path)["price_euros_mwh"])

        df = pd.concat([data_generation, data_consumption, data_fcr_price, data_spot_price], axis=1)
        
        df["non_served_cost"] = self.param["consumption"]["cost_of_non_served_energy"]
        self.df = df[:2400]
        self.generator_data = ( self.df.iloc[self.horizon*i:self.horizon*(i+1), : ].reset_index(drop=True) for i in range(365) )
        
        self.data = next(self.generator_data)


        # Compute average consumption profile 
        inter = self.df.copy()
        inter["mod"] = inter.index%self.horizon
        self.average_consumption = inter.groupby(["mod"]).mean()["consumption"]


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
        cost = -sum_product(model.NonServedCost, model.Non_served_consumption) + np.dot(model.Consumption,model.NonServedCost)
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
        self.model.battery_enabled = Param(initialize=self.param["battery"]["enabled"], within=Binary, mutable = True)
        self.model.SoC = Var(self.model.Period)
        self.model.Charge_power = Var(self.model.Period,initialize=0, bounds=( -self.param["battery"]["power"],0))
        self.model.Discharge_power = Var(self.model.Period,initialize=0, bounds=(0, self.param["battery"]["power"]))
        self.model.Grid_power = Var(self.model.Period, bounds=(0, np.inf), within= Reals)
        self.model.Non_served_consumption = Var(self.model.Period,initialize= 0, bounds=(0, np.inf))
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
        
        def SoC_initialization(model):
            return model.SoC[model.Period.at(1)] == self.param["battery"]["init_soc"]
        def SoC_termination(model):
            return model.SoC[model.Period.at(-1)] == self.param["battery"]["init_soc"]
        


        def battery_charge_on(model, i):
            return -model.Charge_power[i] + self.model.FCR[i]*1.25 <= model.charge_on[i] * self.param["battery"]["power"]*self.model.battery_enabled
        
        def battery_discharge_on(model, i):
            return model.Discharge_power[i] + self.model.FCR[i]*1.25 <= model.discharge_on[i] * self.param["battery"]["power"]*self.model.battery_enabled
              
        def battery_twosided_constraint(model, i):
            return model.charge_on[i] + model.discharge_on[i] <=1


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
        self.model.SoC_initialization = Constraint( rule=SoC_initialization)
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

        self.model.objective = Objective(rule= self.cost_function, sense=maximize)


        # Declare model as istantiated
        self.is_build_milp = True

    def run_milp(self):
        opt = SolverFactory("glpk", executable="solver\glpk\glpsol.exe")
        # opt = SolverFactory("cplex")
        opt.options['tmlim'] = 5
        opt.solve(self.model)
        self.get_results()


    def get_results(self):
        # unpack results
        index,charge_power, discharge_power, SoC, Grid_power,non_served, fcr_power,charge_on,discharge_on, curtailed_power= ([] for i in range(10))

        for i in self.model.Period:
            index.append(i)
            charge_power.append(self.model.Charge_power[i].value)
            discharge_power.append(self.model.Discharge_power[i].value)
            SoC.append(self.model.SoC[i].value)
            Grid_power.append(self.model.Grid_power[i].value)
            non_served.append(self.model.Non_served_consumption[i].value)
            fcr_power.append(self.model.FCR[i].value)
            charge_on.append(self.model.charge_on[i].value)
            discharge_on.append(self.model.discharge_on[i].value)
            curtailed_power.append(self.model.Curtailed_generation[i].value)

        self.result = pd.concat((self.data,pd.DataFrame({"index":index,'SoC':SoC, 'charge_power':charge_power,
                           'discharge_power':discharge_power, 'Grid_power':Grid_power,"non_served":non_served,"fcr_power":fcr_power,"charge_on":charge_on,"discharge_on":discharge_on,"curtailed_power" : curtailed_power}).set_index("index")), axis=1)
        return self.result
 
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
        self.result["non_served"] = self.result["non_served"].clip(lower=0)
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

        # Grid matrix 

        self.capacity_matrix = self.param["capacity_matrix"]        
        self.susceptance_matrix = 50* (np.array(self.param["capacity_matrix"], dtype=bool))
        self.incidence_graph = nx.bfs_tree(nx.from_numpy_matrix(np.array(self.capacity_matrix)),0)
        self.grid_connection = self.param["grid_connection"]
        self.R = 0.01/1000
        self. X = 0.07/1000

        self.grid_nodes = self.param["nodes"]
        self.map_node_to_household = {node: set(house for house in households if house.node ==node) for node in self.grid_nodes}
        self.N_nodes = len(self.grid_nodes)
        self.horizon = self.households[0].horizon


        self.data= pd.concat((house.data for house in self.households), axis=1,  keys=[house.ID for house in self.households])
    
    def create_mg(households, param):
        return Microgrid(households, param)
    
    
    def add_copperplate_constraints(self):
        self.model.theta = Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.voltage = Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.flows = Var(self.model.nodes, self.model.nodes, self.model.Period, domain=Reals)

        def conservation_rule(model,t):
            s = 0
            for i,house in enumerate(self.households):
                s+=self.model.__getattribute__(f"house_{i}").Grid_power[t]
            return s == model.external_import[t]
        
        self.model.conservation_cst = Constraint(self.model.Period, rule=conservation_rule)


    def add_dcopf_constraints(self):
        # Define specific variables for DCOPF
        self.model.theta = Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.voltage = Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.flows = Var(self.model.nodes, self.model.nodes, self.model.Period, domain=Reals)
        
        # Define flows 

        def flow_rule(model, node1,node2, t):
            return self.susceptance_matrix[node1][node2]*(model.theta[node1,t]-model.theta[node2,t]) == model.flows[node1,node2,t]
        
        # kirchoff law

        def kirchoff_rule(model, node,t):
            s=0
            for i,house in enumerate(self.households):
                if house.node == node:  
                    s+=model.__getattribute__(f"house_{i}").Grid_power[t]
            if node == 0 :
                return (s + sum(model.flows[node,j,t] for j in model.nodes if j != node)
                        == self.model.external_import[t] )
            else:
                return (s + sum(model.flows[node,j,t] for j in model.nodes if j != node) == 0 )
        
        # capacity limitation

        def capacity_limitation_rule_up(model, node1,node2, t):
            if self.susceptance_matrix[node1][node2] != 0:
                return model.flows[node1,node2,t]<= self.capacity_matrix[node1][node2]
            else: 
                return Constraint.Feasible
        def capacity_limitation_rule_down(model, node1,node2, t):
            if self.susceptance_matrix[node1][node2] != 0:
                return model.flows[node1,node2,t] >=  - self.capacity_matrix[node1][node2]
            else: 
                return Constraint.Feasible
        
        def reference_angle(model, t):
            return model.theta[0,t] == 0
        # Implement constraints
        self.model.flows_definition = Constraint(self.model.nodes,self.model.nodes,self.model.Period, rule=flow_rule)
        self.model.node_law_constraint = Constraint(self.model.nodes,self.model.Period, rule=kirchoff_rule)
        self.model.capacity_limitation_rule_up = Constraint(self.model.nodes,self.model.nodes,self.model.Period, rule=capacity_limitation_rule_up)
        self.model.capacity_limitation_rule_down = Constraint(self.model.nodes,self.model.nodes,self.model.Period, rule=capacity_limitation_rule_down)
        self.model.reference_angle =Constraint(self.model.Period, rule=reference_angle)
    
    def add_lindistflow_constraints(self):
        self.model.voltage = Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.theta= Var(self.model.nodes, self.model.Period, domain=Reals)
        self.model.flows = Var(self.model.nodes, self.model.nodes, self.model.Period, domain=Reals)
        self.model.reactive_flows = Var(self.model.nodes, self.model.nodes, self.model.Period, domain=Reals)
        self.model.external_import_reactive = Var(self.model.Period, domain=Reals)

        def voltage_rule(model, node, t):
            if node == 0:
                return model.voltage[node,t] == 220
            else :
                parent_node = self.incidence_graph.predecessors(node).__next__()
                return model.voltage[node,t] == model.voltage[parent_node,t] - 2 *self.R*model.flows[parent_node,node,t] - 2*self.X*model.reactive_flows[parent_node,node,t]

        def flow_rule(model,node,t):
            s = 0
            for i,house in enumerate(self.households):
                if house.node == node:  
                    s+=self.model.__getattribute__(f"house_{i}").Grid_power[t]

            if node == 0:
                return model.external_import[t] == sum(model.reactive_flows[node,child,t] for child in self.incidence_graph.successors(node)) + s
            else:
                parent_node = self.incidence_graph.predecessors(node).__next__()
                return  model.flows[parent_node,node,t] == sum(model.flows[node,child,t] for child in self.incidence_graph.successors(node)) + s
        def reactive_flow_rule(model,node,t):
            if node == 0:
                return model.external_import_reactive[t] == sum(model.reactive_flows[node,child,t] for child in self.incidence_graph.successors(node))

            else:
                parent_node = self.incidence_graph.predecessors(node).__next__()
                return model.reactive_flows[parent_node,node,t] == sum(model.reactive_flows[node,child,t] for child in self.incidence_graph.successors(node))

        def voltage_lim_up(model,node,t):
            return model.voltage[node,t] <= 1.05*220
        def voltage_lim_down(model,node,t):
            return model.voltage[node,t] >= 0.95*220 
               
        
        self.model.flow_rule = Constraint(self.model.nodes,self.model.Period, rule=flow_rule)
        self.model.voltage_rule = Constraint(self.model.nodes,self.model.Period, rule=voltage_rule)
        self.model.reactiver_flow_rule = Constraint(self.model.nodes,self.model.Period, rule=reactive_flow_rule)
        self.model.voltage_lim_up_rule = Constraint(self.model.nodes,self.model.Period, rule=voltage_lim_up)
        self.model.voltage_lim_down_rule = Constraint(self.model.nodes,self.model.Period, rule=voltage_lim_down)

    def is_network_radial(self):
        pass

    def build_model(self):
        self.model = ConcreteModel()
        for i,house in enumerate(self.households):
            house.build_milp()
            self.model.add_component(f"house_{i}", house.model)
            self.model.__getattribute__(f"house_{i}").objective.deactivate()
        
        # add grid variables
        self.model.nodes = Set(initialize=self.grid_nodes)
        self.model.Period = Set(initialize=self.households[0].model.Period)
        self.model.external_import = Var(self.model.Period, domain=NonNegativeReals, initialize = 0)
        self.add_copperplate_constraints()
        #self.add_dcopf_constraints()
        # self.add_lindistflow_constraints()
        
        # add limit on importation
        def limit_import_rule(model, t):
            return model.external_import[t] <= self.grid_connection

        
        
                
        def objective_rule(model):
            return sum(model.__getattribute__(f"house_{i}").objective.expr for i in range(len(self.households))) - sum_product(self.households[0].model.SpotPrice,model.external_import) 
        
        self.model.obj = Objective(rule=objective_rule, sense=maximize)
        self.model.limit_import_constraint = Constraint(self.model.Period, rule=limit_import_rule)
    
    def run_model(self):
        opt = SolverFactory('glpk')
        opt.options['tmlim'] = 20
        result_obj = opt.solve(self.model).write()

        # extract results
        self.results = pd.concat((house.get_results() for house in self.households), axis=1,  keys=[house.ID for house in self.households])
        
        #TODO:
        self.consumption = self.results.loc[:,(slice(None),"Grid_power")].copy()
        self.consumption.columns = self.consumption.columns.droplevel(level=1) # level 1 corresponding to signals
        self.consumption["external_import"] = self.model.external_import.get_values()


        # extract flows 
        self.flows = np.zeros((self.horizon,self.N_nodes,self.N_nodes))
        self.thetas = np.zeros((self.horizon,self.N_nodes))
        self.voltages = np.zeros((self.horizon,self.N_nodes))

        for t in self.model.Period : 
            for node1 in self.model.nodes :
                self.thetas[t,node1] = self.model.theta[node1,t].value 
                self.voltages[t,node1] = self.model.voltage[node1,t].value
                for node2 in self.model.nodes:
                    if node2 > node1:
                        self.flows[t,node1,node2] = self.model.flows[node1,node2,t].value
                        self.flows[t,node2,node1] = self.model.flows[node2,node1,t].value
                    

    def get_efficient_allocation (self) :
        self.build_model()
        self.run_model()
        optimal_welfare = self.model.obj.expr()
        optimal_allocation = self.consumption
        print(optimal_allocation, optimal_welfare)
        return optimal_allocation, optimal_welfare
    

    def get_efficient_allocation_wo_battery (self) :
        self.build_model()
        for house in self.households:   
            house.model.battery_enabled = 0
        self.run_model()
        optimal_welfare = self.model.obj.expr()
        optimal_allocation = self.consumption
        print(optimal_allocation, optimal_welfare)
        return optimal_allocation, optimal_welfare
    
    def display_setup(self):

        inner = [f"House {house.ID}" for house in self.households]
        outer = [inner,(["spot","spot"]+['lower left']*(len(self.households)-2))]

        fig, axd = plt.subplot_mosaic(outer, layout="constrained")
        for house in self.households:
                self.households[house.ID].data.consumption.plot(kind="line", ax=axd[f"House {house.ID}"], label=f"Cons {house.ID}")
                self.households[house.ID].data.generation.plot(kind="line", ax=axd[f"House {house.ID}"], label=f"Gen {house.ID}")
                axd[f"House {house.ID}"].legend()
                axd[f"House {house.ID}"].set_title(f"House {house.ID},\n Battery {house.param['battery']['power']}kW,{ house.param['battery']['duration']}h")
                axd[f"House {house.ID}"].set_ylabel("Power (kW)")
                axd[f"House {house.ID}"].set_xlabel("Time")
        # Plot spot price as a function of time 
        self.get_spot_price().plot(kind="line", ax=axd["spot"], label="Spot price")
        axd["spot"].legend()
        axd["spot"].set_title("Spot price")
        axd["spot"].set_ylabel("Price (€/kWh)")
        axd["spot"].set_xlabel("Time")

        # Plot network graph representation 
        G = nx.from_numpy_matrix(np.array(self.capacity_matrix))
        weights = nx.get_edge_attributes(G,'weight').values()
        print(weights)
        scaling_factor = 1/max(weights)
        alphas = [weight * scaling_factor for weight in weights]
        nx.draw(G, pos=nx.planar_layout(G), with_labels=True, ax=axd["lower left"], width=alphas, node_size=100, font_size=10, font_color="white")
        axd["lower left"].set_title("Network representation")
        axd["lower left"].set_xlabel("Node")
        axd["lower left"].set_ylabel("Node")

        plt.tight_layout()
        plt.suptitle("Initial setup")
        
        plt.show()

    def display_gridflows(self):
        fig, ax = plt.subplots(self.N_nodes,self.N_nodes)

        for node1,node2 in iter.product(self.grid_nodes,self.grid_nodes):
            if node2 > node1:
                self.transmission[(node1,node2)].plot(kind="bar", ax=ax[node1,node2])
                (-self.transmission[(node2,node1)]).plot(kind="bar", ax=ax[node1,node2])
                ax[node1,node2].axhline(y=self.capacity_matrix[node1][node2], color='r', linestyle='-')
                ax[node1,node2].axhline(y=-self.capacity_matrix[node2][node1], color='r', linestyle='-')
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

    def display_angles(self):
        plt.imshow(self.thetas.T, cmap='hot')
        plt.title("Angles at each node")
        plt.xlabel("Time")
        plt.ylabel("Node")
        plt.colorbar()
        plt.show()     


    def display_grid(self):
        
        fig,ax = plt.subplots(self.horizon//4,4, figsize=(40,40))
        for t in range(self.horizon):
            G = nx.from_numpy_matrix(self.flows[t])
            weights = nx.get_edge_attributes(G,'weight').values()
            scaling_factor = 0.1/max(weights)
            alphas = [weight * scaling_factor for weight in weights]
            nx.draw(G, pos=nx.planar_layout(G), with_labels=True, ax=ax[t//4,t%4], edge_color=alphas, edge_cmap=plt.cm.Blues, node_size=100, font_size=10, font_color="white")
        
        plt.show()

    def display_flows(self):
        plt.imshow(self.flows[0], cmap='hot')
        plt.title("Flows at time 0")
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.colorbar()
        plt.show()      


    def display_voltages(self):
        plt.imshow(self.voltages.T, cmap='hot')
        plt.title("Voltages at each node")
        plt.xlabel("Time")
        plt.ylabel("Node")
        plt.colorbar()
        plt.show()     

    def display_results(self):
        (self.data.loc[:,( slice(None) , "consumption")].sum( axis =1)-self.data.loc[:,( slice(None) , "generation")].sum( axis =1)).plot(kind="line", label="Residual Load")

        # self.data.loc[:,( slice(None) , "generation")].sum( axis =1).plot(kind="line", label="Total Production")
        self.results.loc[:,( slice(None) , "non_served")].sum(axis=1).plot(kind="line", label="Total non served")
        self.results.loc[:,( slice(None) , "curtailed_power")].sum(axis=1).plot(kind="line", label="Curtailed")
        self.results.loc[:,( slice(None) , "Grid_power")].sum(axis=1).plot(kind="line", label="Total Consumption from grid")

        self.results.loc[:,( slice(None) , "charge_power")].sum(axis=1).plot(kind="line", label="Total Charge from grid")
        self.results.loc[:,( slice(None) , "discharge_power")].sum(axis=1).plot(kind="line", label="Total Discharge from grid")
        plt.legend()
        plt.axhline(y=self.grid_connection, color='r', linestyle='-')
        self.data.loc[:,( slice(None) , "spot_price")].mean( axis =1).plot(kind="line", label="spot_price", secondary_y=True,alpha=0.5)
        #self.results.loc[:,( slice(None) , "non_served")].sum(axis=1).plot(kind="line", label="Total non served")
        plt.grid()
        plt.show()

        for house in self.households:
            house.display_planning()



    def get_bidder_ids(self):
        return [house.ID for house in self.households]

    def get_good_ids(self):
        return [i for i in range(self.horizon)]
    

    def get_uniform_random_bids(self, bidder_id,number_of_bids,seed=None):

        if seed is not None:
            np.random.seed(seed)
        #bids = [l1_ball_sampling(radius=100, dim=self.horizon) for i in range(number_of_bids)]
        bids = uniform_sampling(number_of_bids, [self.grid_connection for i in range(self.horizon)])
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res
    
    def get_uniformly_spaced_bids(self, bidder_id, number_of_bids, seed= None):
        bids = uniform_spacing_sampling(number_of_bids,np.asarray([10 for i in range(self.horizon)]))
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res
    
    def get_greedy_sampled_bids(self, bidder_id, number_of_bids, seed= None):
        bids = greedy_sampling(self.households[bidder_id].average_consumption, number_of_bids, [10 for i in range(self.horizon)])
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res
    
    def get_metropolis_samped_bids(self, bidder_id, number_of_bids, seed= None):
        min_cons = 50
        max_cons = 66.8
        bids = metropolis_sampling(self.households[bidder_id].average_consumption, number_of_bids, [10 for i in range(self.horizon)], min_cons, max_cons)
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res

    def get_kde_random_bids(self, bidder_id,number_of_bids,seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        data = self.households[bidder_id].df["consumption"].to_numpy().reshape(-1,24).T
        noise = np.random.normal(0,1, size = data.shape)*1.5

        data = data+noise
        kde = stats.gaussian_kde(data)
        bids = kde.resample(number_of_bids).T[:,:self.horizon]

        # bids = greedy_sampling(self.households[bidder_id].average_consumption, number_of_bids, [10 for i in range(self.horizon)])
        # bids = fps_sampling(number_of_bids, [10 for i in range(self.horizon)])
        res = []
        for bid in tqdm(bids):
            val = self.calculate_value(bidder_id,bid)
            bid = np.append(bid,val)
            res.append(bid)
        return res

    def generate_dataset(self,bidder_id, dataset_path, number_of_bids):
        print("-----------------------------------")
        print("Generating dataset")
        print("-----------------------------------")
        print(f"Generating dataset for bidder {bidder_id}")
        print("Optimal allocation is : ")
        print(self.households[bidder_id].get_optimal_welfare())
        # bids = self.get_greedy_sampled_bids(bidder_id, number_of_bids)
        #bids = self.get_uniformly_spaced_bids(bidder_id, number_of_bids)
        # bids = self.get_kde_random_bids(bidder_id,number_of_bids)
        # bids = self.get_metropolis_samped_bids(bidder_id,number_of_bids)
        bids = self.get_uniform_random_bids(bidder_id,number_of_bids)
        df = pd.DataFrame(bids)
        df.rename(columns ={self.horizon:"value"}, inplace=True)
        df.to_csv(dataset_path + f"/dataset_{bidder_id}.csv")

    def generate_test_dataset(self, bidder_id, dataset_path, number_of_bids):
        print("-----------------------------------")
        print("Generating test dataset")
        print("-----------------------------------")
        print(f"Generating test dataset for bidder {bidder_id}")
        print("Optimal allocation is : ")
        print(self.households[bidder_id].get_optimal_welfare())
        # bids = self.get_uniformly_spaced_bids(bidder_id, number_of_bids)
        # bids = self.get_metropolis_samped_bids(bidder_id,number_of_bids)
        bids = self.get_uniform_random_bids(bidder_id,number_of_bids)

        df = pd.DataFrame(bids)
        df.rename(columns ={self.horizon:"value"}, inplace=True)
        df.to_csv(dataset_path + f"/test_dataset_{bidder_id}.csv")

    def get_random_feasible_bundle_set(self):
        self.build_model()
        self.model.obj.deactivate()
        
        def random_objective_rule(model):
            return sum(sum_product(np.random.rand(len(model.__getattribute__(f"house_{i}").Period))/(-10),model.__getattribute__(f"house_{i}").Grid_power) for i in range(len(self.households))) 
        self.model.objective = Objective(rule = random_objective_rule, sense=minimize)
        opt = SolverFactory('glpk')
        result_obj = opt.solve(self.model).write()
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
    
    def get_spot_price(self):
        for i, house in enumerate(self.households): # TODO : fix
            if i == 0: 
                house_0 = house
            assert (house.get_spot_price() == house_0.get_spot_price()).all()

        return house_0.get_spot_price()

    def calculate_value(self,bidder_id,bundle):
        if not self.households[bidder_id].param["battery"]["enabled"] :
            val = np.minimum(bundle, self.households[bidder_id].data["consumption"]).sum()*self.households[bidder_id].param["consumption"]["cost_of_non_served_energy"]
        else : 

            if  not self.households[bidder_id].is_build_milp or self.households[bidder_id].model.nobjectives()==0:
                self.households[bidder_id].build_milp()

            def grid_exchange_fix(model, i):
                return model.Grid_power[i] == bundle[i]
            if hasattr( self.households[bidder_id].model, "grid_exchange_fix"):
                 self.households[bidder_id].model.del_component(self.households[bidder_id].model.grid_exchange_fix)
            self.households[bidder_id].model.grid_exchange_fix = Constraint(self.households[bidder_id].model.Period, rule=grid_exchange_fix)
            self.households[bidder_id].run_milp()
            val = self.households[bidder_id].model.objective.expr()
        return  val
            

if __name__=="__main__":
    print("Start loading household profiles")
    folder_path = "config\experiment1\households"
    houses = []
    for file in os.listdir(folder_path)[:3]:
        if file.endswith(".json"):
            household = json.load(open(folder_path+"/"+ file))
        house = HouseHold(household)
        generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
        consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
        spot_price_path = "data/spot_price/2020.csv"
        fcr_price_path = "data/fcr_price/random_fcr.csv"
        house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path, type = float)
        for i in range(1):
            house.next_data()
        houses.append(house)
    print(f"Loaded {len(houses)} households")
    print("Start compute social welfare")
    print([house.ID for house in houses])
    microgrid_1 =json.load(open("config\experiment1\microgrid\exp1_microgrid.json"))
    MG = Microgrid(houses, microgrid_1)
    # MG.build_model()
    # MG.run_model()
    # MG.get_efficient_allocation()
    # MG.display_results()
    # MG.get_efficient_allocation_wo_battery()
    # MG.display_results()
    MG.households[0].display_planning()
    # MG.display_setup()ex
    # # # MG.display_gridflows()
    # # MG.display_angles()
    # # MG.display_flows()
    # # MG.display_voltages()


