

from typing import List  
from dataclasses import dataclass 
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
import numpy as np
import json 
import os

from pyomo.environ import *

@dataclass
class Order(object):   
    CreatorID: int   
    Side: bool  
    Quantity: int   
    Price: int  
    TimeSlot: int
    Node : int


@dataclass  
class Match(object):   
    Bid: Order   
    Offer: Order   

@dataclass
class Participant(object):
    ID: int
    bid_qtty: int
    awarded_qtty: int
    clearing_price: int

    


class Market(object):
    def __init__(self, grid_path = None):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []
        self.Gate_open = True
        self.grid_path = grid_path
        self.payments = {}


    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)
    
    def close_gate(self):
        self.Offers = sorted(self.Offers, key=lambda x: x.Price, reverse=True)
        self.Participants = set(list(map(lambda x: x.CreatorID, self.Offers)))
        self.Nodes = set(list(map(lambda x: x.Node, self.Offers)))
        if 0 not in self.Nodes:
            self.Nodes.add(0)
        self.TimeSlots = sorted(list(set(list(map(lambda x: x.TimeSlot, self.Offers)))))
        self.Gate_open = False
        self.N_part = len(self.Participants)
        self.accepted_bids = {}
        self.clearing_price = {}
        self.qtty_cleared = {}
        self.sw_global = {}

    
    def plot_orders(self):
        if self.Gate_open:
            print("Gate is open, please close it before plotting")
            raise TypeError

        fig,ax = plt.subplots(6,4)
        color = plt.cm.rainbow(np.linspace(0, 1, len(self.Participants)))
        for time in self.TimeSlots:
            i = time//4
            j = time%4
            offers = list(filter(lambda x: x.TimeSlot == time, self.Offers))
            prices = np.array(list(map(lambda x: x.Price, offers)))
            quantities = np.array(list(map(lambda x: x.Quantity, offers)))
            offerers = np.array(list(map(lambda x: x.CreatorID, offers)))
            colors = np.array(list(map(lambda x: color[x], offerers)))
            ax[i,j].bar(np.cumsum(quantities), 
                height = prices,
                align="edge",
                width = -quantities,
                color = colors,
                fill = True)
            ax[i,j].set_title(f"Time Slot {time}")
        plt.suptitle("Curve preview")
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()    
        plt.tight_layout()
        plt.show()

    # Allocation Rules 
    # Goal Maximizing social welfare

    def dispatch(self, offers, spot_price):
        self.market_model = ConcreteModel()
        self.market_model.dual = Suffix(direction=Suffix.IMPORT)
        N = len(offers)
        K = len(self.Participants)
        self.market_model.Bids = RangeSet(0,N-1)
        self.market_model.participants_set = Set(initialize = sorted(list(self.Participants)))
        self.market_model.nodes_set = Set(initialize = sorted(list(self.Nodes)))
        
        self.market_model.Prices = Param(self.market_model.Bids,initialize = list(map(lambda x: x.Price, offers)))
        self.market_model.Quantities = Param(self.market_model.Bids,initialize = np.array(list(map(lambda x: x.Quantity, offers))))
        self.market_model.Participants = Param(self.market_model.Bids,initialize = np.array(list(map(lambda x: x.CreatorID, offers))))
        self.market_model.Nodes= Param(self.market_model.Bids,initialize = np.array(list(map(lambda x: x.Node, offers))))
        self.market_model.is_awarded_bid = Var(self.market_model.Bids, initialize=0,within=Binary)

        if self.grid_path is not None:
            microgrid =json.load(open(self.grid_path))
            congestion_matrix = microgrid["congestion_matrix"]
            grid_connection = microgrid["grid_connection"]
             # add grid variables
            self.market_model.flows = Var(self.market_model.nodes_set, self.market_model.nodes_set, domain=NonNegativeReals)
            self.market_model.external_import = Var(initialize = 0, domain=NonNegativeReals)
        
            # add kirchoff's law

            def kirchoff_rule(model, node):
                s = 0

                for i in model.Bids:
                    if model.Nodes[i] == node:
                        s+=model.Quantities[i]*model.is_awarded_bid[i]
                if node == 0 :
                    return (s + sum(model.flows[node,j] for j in model.nodes_set if j != node)
                            == sum(model.flows[j,node] for j in model.nodes_set if j != node)
                            + self.market_model.external_import)
                else:
                    return (s + sum(model.flows[node,j] for j in model.nodes_set if j != node)
                            == sum(model.flows[j,node] for j in model.nodes_set if j != node))
            

            def capacity_limitation_rule(model, node1,node2):
                return model.flows[node1,node2] <= congestion_matrix[node1][node2]        

            def limit_import_rule(model):
                return model.external_import <= grid_connection
            

        # Objective rule 
        def objective_rule(model):
            return sum_product(model.Prices, model.is_awarded_bid) - spot_price*model.external_import
        

        # Solve model

        self.market_model.obj = Objective(rule = objective_rule, sense = maximize)
        
        if self.grid_path is not None:
            self.market_model.flows_cons = Constraint(self.market_model.nodes_set,  rule = kirchoff_rule)
            self.market_model.capacity_cons = Constraint(self.market_model.nodes_set, self.market_model.nodes_set, rule = capacity_limitation_rule)
            self.market_model.limit_import_cons = Constraint(rule = limit_import_rule)



        # self.market_model.one_bid_cons = Constraint(self.market_model.participants_set, rule = one_bid_per_participant_rule)
        opt = SolverFactory("glpk", executable="solver\glpk\glpsol.exe")
        opt.options['tmlim'] = 3 
        opt.solve(self.market_model)
        # unpack results

        awarded_bids = []
        cleared_qtty = 0

        for i in range(N):
            if self.market_model.is_awarded_bid[i].value == 1:
                # print(offers[i].CreatorID, offers[i].Price, offers[i].Quantity)
                cleared_qtty += offers[i].Quantity
                awarded_bids.append(offers[i])
        clearing_price = awarded_bids[-1].Price
        return clearing_price, cleared_qtty , awarded_bids
    
    def ClearMarket(self, spot_prices):
        if self.Gate_open:
            print("Gate is open, please close it before clearing")
            raise TypeError
        for time in tqdm(self.TimeSlots):

            filtered_offers = list(filter(lambda x: x.TimeSlot == time, self.Offers))
            self.clearing_price[time], self.qtty_cleared[time], self.accepted_bids[time]= self.dispatch(filtered_offers, spot_prices[time])
            self.sw_global[time] = self.compute_social_welfare( self.accepted_bids[time], self.clearing_price[time])
            self.spot_prices = spot_prices
        return self.clearing_price, self.accepted_bids
    
    def compute_social_welfare(self, accepted_offers, clearing_price):
        return np.dot(np.array(list(map(lambda x: x.Quantity, accepted_offers))), clearing_price - np.array(list(map(lambda x: x.Price,accepted_offers))))

    # Payments rules 


    def LMP_payments(self):
        self.payments["LMP"] = {}
        for time in tqdm(self.TimeSlots):
            self.payments["LMP"][time] = {}
            for part in self.Participants:
                self.payments["LMP"][time][part] = self.clearing_price[time]* sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part and x.TimeSlot == time, self.accepted_bids[time])))
        return self.payments["LMP"]

    def VCG_payments(self): 
        self.payments["VCG"] = {}
        self.SW_without_participant={}
        print("Computing VCG payments")
        for time in tqdm(self.TimeSlots):
            self.payments["VCG"][time] = {}
            self.SW_without_participant[time] = {}
            for part in self.Participants:
                filtered_offers = list(filter(lambda x: x.CreatorID != part and x.TimeSlot == time, self.Offers))
                clearing_price_without_participant,quantity_cleared_wo_participant, accepted_offers_without_participant = self.dispatch(filtered_offers, self.spot_prices[time])
                sw_without_participant= self.compute_social_welfare(accepted_offers_without_participant, clearing_price_without_participant)
                sw_participant= self.compute_social_welfare(list(filter(lambda x: x.CreatorID == part and x.TimeSlot ==time, self.accepted_bids[time])), 0)
                self.payments["VCG"][time][part] = -sw_without_participant + (self.sw_global[time] - sw_participant)
                self.SW_without_participant[time][part] = sw_without_participant
        return self.payments["VCG"]
    
        
    
    def plot_clearing(self):
        fig, ax = plt.subplots(6,4)

        for t in self.TimeSlots:
            offers = list(filter(lambda x: x.TimeSlot == t, self.Offers))
            ax[t//4,t%4].plot( np.cumsum(list(map(lambda x:x.Quantity,offers))),[x.Price for x in offers], color="red", drawstyle = "steps",label="Offers")
            ax[t//4,t%4].plot( np.cumsum(list(map(lambda x:x.Quantity, self.accepted_bids[t]))),[x.Price for x in self.accepted_bids[t]],drawstyle = "steps",label="Accepted Offers")
            ax[t//4,t%4].axvline(self.qtty_cleared[t], color="blue", linestyle="--",label="Quantity Cleared")
            ax[t//4,t%4].axhline(self.clearing_price[t], color="black", linestyle="--", label="Clearing Price")
            ax[t//4,t%4].set_xlabel("Quantity")
            ax[t//4,t%4].set_ylabel("Price")

        plt.suptitle("Clearing")
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()    
        plt.show()

    def plot_clearing_per_participant(self):
        for part in self.Participants:
            plt.plot( np.cumsum(list(map(lambda x:x.Quantity,filter(lambda x:x.CreatorID == part,self.Offers)))),[x.Price for x in filter(lambda x:x.CreatorID == part,self.Offers)], color="red", drawstyle = "steps", alpha = 0.1)
            plt.plot( np.cumsum(list(map(lambda x:x.Quantity,filter(lambda x:x.CreatorID == part,self.accepted_bids)))),[x.Price for x in filter(lambda x:x.CreatorID == part,self.accepted_bids)],drawstyle = "steps",label=part)
        plt.axvline(self.qtty_cleared, color="blue", linestyle="--",label="Quantity Cleared")
        plt.axhline(self.clearing_price, color="black", linestyle="--", label="Clearing Price")
        plt.xlabel("Quantity")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def plot_payments(self):
        fig, ax = plt.subplots(6,4)
        for part in self.Participants:

            ax[part//4,part%4].plot(self.TimeSlots,[self.payments["LMP"][time][part] for time in self.TimeSlots], drawstyle ="steps",label="LMP")
            ax[part//4,part%4].plot(self.TimeSlots,[self.payments["VCG"][time][part] for time in self.TimeSlots], drawstyle ="steps",label="VCG")
            ax[part//4,part%4].set_xlabel("Time")
            ax[part//4,part%4].set_ylabel("Payments")
            ax[part//4,part%4].set_title(f"Participant {part}")
            ax[part//4,part%4].legend()
        plt.suptitle("Payments")
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()

    def report_clearing(self):
        return 0
        print("Clearing Price: ", self.clearing_price)
        print("Quantity Cleared: ", self.qtty_cleared)
        print(f"Social Welfare Global: ", self.sw_global)
        for part in self.Participants :
            print(f"Participant {part} / Awarded Quantity: ", sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids))))
            
            print(f"                  / Social Welfare part: ", self.compute_social_welfare(list(filter(lambda x:x.CreatorID == part,self.accepted_bids)), self.clearing_price))
            print(f"                  / Social Welfare wo/ part: ", self.SW_without_participant[part])
            print(f"                  / VCG Price: ", self.payments["VCG"][part]," / ", self.payments["VCG"][part]/sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids))), " per unit")
            print(f"                  / Pay as clear: ", self.payments["LMP"][part], " / ", self.clearing_price, " per unit")

if __name__ == "__main__":
    # Create market instance and test orders   
    market = Market("config\microgrid_profile/non_constrained_microgrid.json")
    for t in range(24):
        for i in range(3):
            for j in range(20):
                sellOrder = Order(CreatorID=i, Side=True, TimeSlot=t, Quantity=1, Price=  np.random.randint(1,40), Node = np.random.randint(1,4)) 
                market.AddOrder(sellOrder)      

    # Send orders to market   
    spot_prices = [0.004 for _ in range(24)]
    market.AddOrder(sellOrder)  
    market.close_gate()
    market.plot_orders()
    
    market.ClearMarket( spot_prices)
    market.LMP_payments()
    #market.VCG_payments()
    #market.report_clearing()
    market.plot_clearing()
    #market.plot_clearing_per_participant()
    #market.plot_payments()