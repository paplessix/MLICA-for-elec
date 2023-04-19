

from typing import List  
from dataclasses import dataclass 

import matplotlib.pyplot as plt
import numpy as np

from pyomo.environ import *

@dataclass
class Order(object):   
    CreatorID: int   
    Side: bool  
    Quantity: int   
    Price: int  
    TimeSlot: int


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
    def __init__(self):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []
        self.Gate_open = True
        self.payments = {}

    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)
    
    def close_gate(self):
        self.Offers = sorted(self.Offers, key=lambda x: x.Price, reverse=True)
        self.Participants = set(list(map(lambda x: x.CreatorID, self.Offers)))
        self.TimeSlots = sorted(list(set(list(map(lambda x: x.TimeSlot, self.Offers)))))
        self.Gate_open = False
        self.N_part = len(self.Participants)


    
    def plot_orders(self):
        if self.Gate_open:
            print("Gate is open, please close it before plotting")
            raise TypeError

 
        color = plt.cm.rainbow(np.linspace(0, 1, len(self.Participants)))
        prices = np.array(list(map(lambda x: x.Price, self.Offers)))
        quantities = np.array(list(map(lambda x: x.Quantity, self.Offers)))
        offerers = np.array(list(map(lambda x: x.CreatorID, self.Offers)))
        colors = np.array(list(map(lambda x: color[x], offerers)))
        plt.bar(np.cumsum(quantities), 
            height = prices,
            align="edge",
            width = -quantities,
            color = colors,
            fill = True)
        plt.show()

    # Allocation Rules 
    # Goal Maximizing social welfare

    def dispatch(self, offers, grid_constraint):
        
        self.market_model = ConcreteModel()
        self.market_model.dual = Suffix(direction=Suffix.IMPORT)
        N = len(offers)
        K = len(self.Participants)
        self.market_model.Bids = RangeSet(0,N-1)
        self.market_model.participants_set = Set(initialize = sorted(list(self.Participants)))
        
        self.market_model.Prices = Param(self.market_model.Bids,initialize = list(map(lambda x: x.Price, offers)))
        self.market_model.Quantities = Param(self.market_model.Bids,initialize = np.array(list(map(lambda x: x.Quantity, offers))))
        self.market_model.Participants = Param(self.market_model.Bids,initialize = np.array(list(map(lambda x: x.CreatorID, offers))))
        
        self.market_model.is_awarded_bid = Var(self.market_model.Bids, initialize=0,within=Binary)

        # Objective rule 
        def objective_rule(model):
            return sum_product(model.Prices, model.is_awarded_bid) - 0.04*sum_product(model.Quantities,model.is_awarded_bid)
        
        # Constraint rules

        def one_bid_per_participant_rule(model, part):
            return sum([model.is_awarded_bid[i] for i in range(N) if model.Participants[i] == part]) <= 1
        
        def limit_qtty_rule(model):
            return sum_product(model.Quantities, model.is_awarded_bid) <= grid_constraint
        # Solve model

        self.market_model.obj = Objective(rule = objective_rule, sense = maximize)
        # self.market_model.one_bid_cons = Constraint(self.market_model.participants_set, rule = one_bid_per_participant_rule)
        self.market_model.limit_qtty_cons = Constraint(rule = limit_qtty_rule)
        opt = SolverFactory("glpk", executable="solver\glpk\glpsol.exe")
        opt.options['tmlim'] = 3
        opt.solve(self.market_model)
        # unpack results

        awarded_bids = []
        for i in range(N):
            if self.market_model.is_awarded_bid[i].value == 1:
                # print(offers[i].CreatorID, offers[i].Price, offers[i].Quantity)
                awarded_bids.append(offers[i])
        clearing_price = awarded_bids[-1].Price
        return clearing_price , awarded_bids
    
    def ClearMarket(self, qtty_cleared):
        if self.Gate_open:
            print("Gate is open, please close it before clearing")
            raise TypeError
        for time in self.TimeSlots:
            filtered_offers = list(filter(lambda x: x.TimeSlot == time, self.Offers))
            self.clearing_price, self.accepted_bids = self.dispatch(filtered_offers, qtty_cleared)
            self.sw_global = self.compute_social_welfare( self.accepted_bids, self.clearing_price)
            self.qtty_cleared = qtty_cleared
        return self.clearing_price, self.accepted_bids
    
    def compute_social_welfare(self, accepted_offers, clearing_price):
        return np.dot(np.array(list(map(lambda x: x.Quantity, accepted_offers))), clearing_price - np.array(list(map(lambda x: x.Price,accepted_offers))))

    # Payments rules 


    def LMP_payments(self):
        self.payments["LMP"] = {}
        for part in self.Participants:
            self.payments["LMP"][part] = self.clearing_price* sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids)))
        return self.payments["LMP"]

    def VCG_payments(self): 
        self.payments["VCG"] = {}
        self.SW_without_participant={}
        for part in self.Participants:
            filtered_offers = list(filter(lambda x: x.CreatorID != part, self.Offers))
            clearing_price_without_participant, accepted_offers_without_participant = self.dispatch(filtered_offers,self.qtty_cleared)
            sw_without_participant= self.compute_social_welfare(accepted_offers_without_participant, clearing_price_without_participant)
            sw_participant= self.compute_social_welfare(list(filter(lambda x: x.CreatorID == part, self.accepted_bids)), 0)
            self.payments["VCG"][part] = -sw_without_participant + (self.sw_global - sw_participant)
            self.SW_without_participant[part] = sw_without_participant
        return self.payments["VCG"]
    
        
    
    def plot_clearing(self):
        plt.plot( np.cumsum(list(map(lambda x:x.Quantity,self.Offers))),[x.Price for x in self.Offers], color="red", drawstyle = "steps",label="Offers")
        plt.plot( np.cumsum(list(map(lambda x:x.Quantity,self.accepted_bids))),[x.Price for x in self.accepted_bids],drawstyle = "steps",label="Accepted Offers")
        plt.axvline(self.qtty_cleared, color="blue", linestyle="--",label="Quantity Cleared")
        plt.axhline(self.clearing_price, color="black", linestyle="--", label="Clearing Price")
        plt.xlabel("Quantity")
        plt.ylabel("Price")
        plt.legend()
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
        width = 0.3 
        plt.bar(np.arange(self.N_part), list(self.payments["VCG"].values()),width, color="red", label="VCG")
        plt.bar(np.arange(width,self.N_part+width), list(self.payments["LMP"].values()),width, color="blue", label="LMP")
        plt.xlabel("Participant")
        plt.ylabel("Payment")
        plt.legend()
        plt.show()

    def report_clearing(self):
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
    market = Market()
    for i in range(3):
        for j in range(20):
            sellOrder = Order(CreatorID=i, Side=True, TimeSlot=1, Quantity=1, Price=  np.random.randint(1,40)) 
            market.AddOrder(sellOrder)      

    # Send orders to market   

    market.AddOrder(sellOrder)  
    market.close_gate()
    market.plot_orders()
    
    market.ClearMarket(40)
    market.LMP_payments()
    market.VCG_payments()
    market.report_clearing()
    market.plot_clearing()
    market.plot_clearing_per_participant()
    market.plot_payments()