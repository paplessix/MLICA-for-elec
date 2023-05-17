import logging
import itertools as iter
import docplex.mp.model as cpx
import numpy as np

def _add_MG_specific_constraints_dcopf(self):
    logging.info("Adding MG specific constraints")
    logging.info("Chosen OPF Modelling method: DCOPF")
    nodes = self.MG_instance.grid_nodes 
    period = np.arange(self.MG_instance.horizon)
    # Add constraints for each node
    # Create variables 

    self.theta = {}
    self.voltage = {}
    self.flows = {}
    self.external_import = {}
    self.demand = {}

    # Update variables

    self.theta.update({(n,t) : self.Mip.continuous_var(name=f"theta_{n}_{t}", lb = -self.Mip.infinity) for n in nodes for t in period})
    self.voltage.update({(n,t) : self.Mip.continuous_var(name=f"voltage_{n}_{t}") for n in nodes for t in period})
    self.flows.update({(n1,n2,t) : self.Mip.continuous_var(name=f"flow_{n1}_to_{n2}_{t}", lb = -self.Mip.infinity) for n1 in nodes for n2 in nodes for t in period})
    self.external_import.update({(t) : self.Mip.continuous_var(name=f"external_import_{t}", lb = -self.Mip.infinity) for t in period})
    
    # Add constraints

    # # 1. Flows defintion
    for node1,node2, t in iter.product(nodes,nodes,period):
        self.Mip.add_constraint(ct = (self.MG_instance.susceptance_matrix[node1][node2]*(self.theta[(node1,t)] - self.theta[(node2,t)]) == self.flows[(node1,node2,t)] ))
    
    # 2. Power balance
    for node,t in iter.product(nodes,period):
        if node == 0:
            self.Mip.add_constraint(ct = ((self.Mip.sum(self.flows[(node,node2,t)] for node2 in nodes )
                                        + self.Mip.sum(self.z[(i, 0, t)] for i,house in enumerate(self.MG_instance.households) if house.node == node) )==  self.external_import[(t)]  ))
        else:
            self.Mip.add_constraint(ct = ((self.Mip.sum(self.flows[(node,node2,t)] for node2 in nodes ) 
                                        + self.Mip.sum(self.z[(i, 0, t)] for i,house in enumerate(self.MG_instance.households) if house.node == node) )==  0  ))
       
    # #3. Capacity constraints
    for node1,node2, t in iter.product(nodes,nodes,period):
        self.Mip.add_constraint(ct = (self.flows[(node1,node2,t)] <= +self.MG_instance.capacity_matrix[node1][node2]))
        self.Mip.add_constraint(ct = (self.flows[(node1,node2,t)] >= -self.MG_instance.capacity_matrix[node1][node2]))
        if node2 > node1 : 
            self.Mip.add_constraint(ct = (self.flows[(node1,node2,t)] == -self.flows[(node2,node1,t)]))
    # 4. Reference node constraints
    for t in period:
        self.Mip.add_constraint(ct = (self.theta[(0,t)] == 0))
        self.Mip.add_constraint(ct= (self.external_import[(t)]  <= self.MG_instance.grid_connection))


