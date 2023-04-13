

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


class Market(object):
    def __init__(self):
        self.Bids: List[Order] = []
        self.Offers: List[Order] = []
        self.Matches: List[Match] = []

    def AddOrder(self, order: Order):
        if order.Side:
            self.Offers.append(order)
        else:
            self.Bids.append(order)

    def plot_orders(self):
        self.Offers = sorted(self.Offers, key=lambda x: x.Price)
        self.Participants = set(list(map(lambda x: x.CreatorID, self.Offers)))
 
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

    def ClearMarket(self, qtty_cleared):
        self.Offers = sorted(self.Offers, key=lambda x: x.Price)
        self.Participants = set(list(map(lambda x: x.CreatorID, self.Offers)))
        self.cumulative_quantity = 0
        self.clearing_price = self.Offers[0].Price
        n=0
        self.accepted_bids = []
        self.qtty_cleared = qtty_cleared
        while self.cumulative_quantity < self.qtty_cleared:

            if self.Offers[n].Quantity > self.qtty_cleared - self.cumulative_quantity:
                accepted_qtty = self.qtty_cleared - self.cumulative_quantity
            else:
                accepted_qtty = self.Offers[n].Quantity
            new_offer = Order(CreatorID=self.Offers[n].CreatorID, Side=True, Quantity=accepted_qtty, Price=self.Offers[n].Price)
            self.cumulative_quantity += accepted_qtty
            self.clearing_price = self.Offers[n].Price
            self.accepted_bids.append(new_offer)
            n+=1
        return self.clearing_price, self.accepted_bids
    
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
        for part in self.Participants :
            print(f"Participant {part} / Awarded Quantity: ", sum(map(lambda x: x.Quantity, filter(lambda x: x.CreatorID == part, self.accepted_bids))))
            print(f"                  / Social Welfare: ", np.dot(np.array(list(map(lambda x: x.Quantity,
                                                                                filter(lambda x: x.CreatorID == part, self.accepted_bids)))),
                                                                  self.clearing_price - np.array(list(map(lambda x: x.Price,
                                                                                filter(lambda x: x.CreatorID == part, self.accepted_bids))))))

if __name__ == "__main__":
    # Create market instance and test orders   
    market = Market()
    for i in range(10):
        sellOrder = Order(CreatorID=i, Side=True, Quantity=np.random.randint(1,20), Price=np.random.randint(1,20)) 
        market.AddOrder(sellOrder)      

    # Send orders to market   

    market.AddOrder(sellOrder)  
    market.plot_orders()
    # Clear Market  
    #market.ClearMarket(20)
    # Get the clearing price  
    #market.report_clearing()
    #market.plot_clearing()
    # returns 9  
