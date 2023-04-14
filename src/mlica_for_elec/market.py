

from typing import List  
from dataclasses import dataclass 

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Order(object):   
    CreatorID: int   
    Side: bool  
    Quantity: int   
    Price: int  


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

    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)
    
    def close_gate(self):
        self.Offers = sorted(self.Offers, key=lambda x: x.Price)
        self.Participants = set(list(map(lambda x: x.CreatorID, self.Offers)))
        self.Gate_open = False


    def plot_orders(self):
        if self.Gate_open:
            print("Gate is open, please close it before plotting")
            raise TypeError

 
        color = plt.cm.rainbow(np.linspace(0, 1, len(self.Participants)))
        prices = np.array(list(map(lambda x: x.Price, self.Offers)))
        quantities = np.array(list(map(lambda x: x.Quantity, self.Offers)))
        offerers = np.array(list(map(lambda x: x.CreatorID, self.Offers)))
        colors = np.array(list(map(lambda x: color[x], offerers)))
        print(np.cumsum(quantities))
        plt.bar(np.cumsum(quantities), 
            height = prices,
            align="edge",
            width = -quantities,
            color = colors,
            fill = True)
        plt.show()


    def maximize_SW(self,ordered_offers,qtty_cleared):
        cumulative_quantity = 0
        clearing_price = ordered_offers[0].Price
        accepted_bids = []
        n=0
        qtty_cleared = qtty_cleared
        while cumulative_quantity < qtty_cleared:
            if ordered_offers[n].Quantity > qtty_cleared - cumulative_quantity:
                accepted_qtty = qtty_cleared - cumulative_quantity
            else:
                accepted_qtty = ordered_offers[n].Quantity
            new_offer = Order(CreatorID=ordered_offers[n].CreatorID, Side=True, Quantity=accepted_qtty, Price=ordered_offers[n].Price)
            cumulative_quantity += accepted_qtty
            clearing_price = ordered_offers[n].Price
            accepted_bids.append(new_offer)
            n+=1
        return clearing_price, accepted_bids


    def ClearMarket(self, qtty_cleared):
        if self.Gate_open:
            print("Gate is open, please close it before clearing")
            raise TypeError
        self.clearing_price, self.accepted_bids = self.maximize_SW(self.Offers,qtty_cleared)
        self.sw_global = self.compute_social_welfare( self.accepted_bids, self.clearing_price)
        self.qtty_cleared = qtty_cleared
        return self.clearing_price, self.accepted_bids
    
    def compute_social_welfare(self, accepted_offers, clearing_price):
        return np.dot(np.array(list(map(lambda x: x.Quantity, accepted_offers))), clearing_price - np.array(list(map(lambda x: x.Price,accepted_offers))))
    
    def compute_VCG_prices(self):
        if self.Gate_open:
            print("Gate is open, please close it before clearing")
            raise TypeError
        self.VCG_payments={}
        self.SW_without_participant={}
        for participant in self.Participants:
            filtered_offers = list(filter(lambda x: x.CreatorID != participant, self.Offers))
            clearing_price_without_participant, accepted_offers_without_participant = self.maximize_SW(filtered_offers,self.qtty_cleared)
            sw_without_participant= self.compute_social_welfare(accepted_offers_without_participant, clearing_price_without_participant)
            sw_participant= self.compute_social_welfare(list(filter(lambda x: x.CreatorID == participant, self.accepted_bids)), self.clearing_price)
            self.VCG_payments[participant] = -sw_without_participant + (self.sw_global - sw_participant)
            self.SW_without_participant[participant] = sw_without_participant

        
    
    def plot_clearing(self):
        plt.plot( np.cumsum(list(map(lambda x:x.Quantity,self.Offers))),[x.Price for x in self.Offers], color="red", drawstyle = "steps",label="Offers")
        plt.plot( np.cumsum(list(map(lambda x:x.Quantity,self.accepted_bids))),[x.Price for x in self.accepted_bids],drawstyle = "steps",label="Accepted Offers")
        plt.axvline(self.qtty_cleared, color="blue", linestyle="--",label="Quantity Cleared")
        plt.axhline(self.clearing_price, color="black", linestyle="--", label="Clearing Price")
        plt.xlabel("Quantity")
        plt.ylabel("Price")
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
            print(f"                  / VCG Price: ", self.VCG_payments[part]," / ", self.VCG_payments[part]/sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids))), " per unit")
            print(f"                  / Pay as clear: ", self.clearing_price* sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids))), " / ", self.clearing_price, " per unit")

if __name__ == "__main__":
    # Create market instance and test orders   
    market = Market()
    for i in range(3):
        for j in range(20):
            sellOrder = Order(CreatorID=i, Side=True, Quantity=np.random.randint(1,20), Price=np.random.randint(1,20)) 
            market.AddOrder(sellOrder)      

    # Send orders to market   

    market.AddOrder(sellOrder)  
    market.close_gate()
    market.plot_orders()
    # Clear Market  
    market.ClearMarket(300)
    market.compute_VCG_prices()
    # Get the clearing price  
    market.report_clearing()
    market.plot_clearing()
    # returns 9  
