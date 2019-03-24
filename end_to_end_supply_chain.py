import os
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')
from mipcl_py.mipshell.mipshell import *

class supply_chain_optimization(Problem):
    def model(self, g, v, h, cpw, cwd, cdc, sup,\
              D, W, d, x, epsilon, eta, All_list):
        """ 
        Input parameters:
        ------------------
        g[k] = Fixed cost of the operating the wholesale store k (k = 1 to K)
        v[j] = Fixed cost of operating DC j (j = 1 to J)
        h[l][j] = Cost of holding unit amount of product l at DC j
        cpw[l][s][k] = Unit transportation cost for product l from plant s to wholeseler k (s = 1 to S, k = 1 to K)
        cwd[l][k][j] = Unit transportation cost for product l from wholeseler k to DC j (k = 1 to K, j = 1 to J)
        cdc[l][j][i] = Unit transportation cost for product l from DC j to customer i (j = 1 to J, i = 1 to I)
        sup[l][s] = Supply limit of plant s for product l (s = 1 to S)
        D[k] = Capacity of wholesale store k (k = 1 to K))
        W[j] = Capacity of DC j (j = 1 to J)
        d[l][i] = Demand for product l at customer i (i = 1 to I)
        x[j][i] = Binary values indicating whether customer i can be reached from DC j in \tau hours
        epsilon = % service level, e.g. eta = 0.85 for 85% service level
        eta[l][i] = Ratio of ordered amount to the given demand for product l at customer i
        """
        self.I = len(d[0]) # Number of Customers
        self.J = len(v) # Number of DCs
        self.K = len(D) # Number of Wholesale stores
        self.S = len(sup[0]) # Number of Plants
        self.L = len(d) # Number of products
        
        self.g = g 
        self.v = v 
        self.h = h = np.tile(h, (self.L, 1))
        self.cpw = cpw = np.tile(cpw, (self.L, 1, 1)) 
        self.cwd = cwd = np.tile(cwd, (self.L, 1, 1)) 
        self.cdc = cdc = np.tile(cdc, (self.L, 1, 1)) 
        self.sup = sup
        self.D = D
        self.W = W 
        self.d = d 
        self.x = x 
        self.epsilon = epsilon
        self.eta = eta
                
        self.All_list = All_list
        
        '''Decision variables
        ----------------------
        b[l][s][k] = Quantity of product l shipped from plant s to wholesaler k
        f[l][k][j] = Quantity of product l shipped from wholesaler k to DC j
        q[l][j][i] = Quantity of product l shipped from DC j to customer i
        z[j] = 1 if DC j is used, otherwise 0
        y[j][i] = 1 if DC j serves the customer i, otherwise 0
        p[k] = 1, if wholesale store k is open, otherwise 0
        '''
        self.b = b = VarVector([self.L, self.S, self.K],'b', INT, lb = 0.0)
        self.f = f = VarVector([self.L, self.K, self.J],'f', INT, lb = 0.0)
        self.q = q = VarVector([self.L, self.J, self.I],'q', INT, lb = 0.0)
        self.z = z = VarVector([self.J],'z', BIN)
        self.y = y = VarVector([self.J, self.I],'y', BIN)
        self.p = p = VarVector([self.K],'p', BIN)
	
        '''Objective function'''
        Wholesalers_operating_cost = sum_(g[k]*p[k] for k in range(self.K))
        DCs_operating_cost = sum_(v[j]*z[j] for j in range(self.J))
        Tp_cost_plant_to_wholesaler = sum_(cpw[l][s][k]*b[l][s][k] for l in range(self.L)\
                                           for s in range(self.S) for k in range(self.K))
        Tp_cost_wholesaler_to_DC = sum_(cwd[l][k][j]*f[l][k][j] for l in range(self.L)\
                                        for k in range(self.K) for j in range(self.J))
        Tp_cost_DC_to_customer = sum_(eta[l][i]*cdc[l][j][i]*q[l][j][i] for l in range(self.L)\
                                      for j in range(self.J) for i in range(self.I))        
        Inventory_holding_cost = sum_((1 - eta[l][i])*h[l][j]*q[l][j][i] for l in range(self.L)\
                                      for j in range(self.J) for i in range(self.I))
        
        minimize(
            Wholesalers_operating_cost + DCs_operating_cost + Tp_cost_plant_to_wholesaler + \
            Tp_cost_wholesaler_to_DC + Tp_cost_DC_to_customer + Inventory_holding_cost
        )
    
        '''Second objective function reduced to \epsilon-constraint'''
        epsilon*sum_(eta[l][i]*d[l][i] for l in range(self.L) for i in range(self.I)) - \
        sum_(eta[l][i]*x[j][i]*q[l][j][i] for l in range(self.L) for j in range(self.J)\
             for i in range(self.I)) <= 0
        
        '''Unique assignment of a DC to a customer'''
        for i in range(self.I):
            sum_(y[j][i] for j in range(self.J)) <= 1

        '''Outflow from DC <= Capacity constraint for DC'''
        for j in range(self.J):
            sum_(q[l][j][i] for l in range(self.L) for i in range(self.I)) <= W[j]*z[j]
            
        '''Inward flow into a DC <= Capacity constraint for DC'''
        for j in range(self.J):
            sum_(f[l][k][j] for l in range(self.L) for k in range(self.K)) <= W[j]*z[j]

        '''Number of DCs that can be opened'''
        sum_(z[j] for j in range(self.J)) <= self.J
            
        '''Satisfaction of customer demand for the product'''
        for l in range(self.L):
            for j in range(self.J):
                for i in range(self.I):
                    q[l][j][i] == d[l][i]*y[j][i]
                
        '''Ensure that y[j][i] = 0 when z[j] = 0'''
        for j in range(self.J):
            sum_(y[j][i] for i in range(self.I)) <= self.J*z[j]
            
        '''Outflow from a DC <= Inward flow into DC'''
        for l in range(self.L):
            for j in range(self.J):
                sum_(q[l][j][i] for i in range(self.I)) <=\
                sum_(f[l][k][j] for k in range(self.K))
            
        '''Plant supply restriction'''
        for l in range(self.L):
            for s in range(self.S):
                sum_(b[l][s][k] for k in range(self.K)) <= sup[l][s]
            
        '''Inward flow into a wholesale store <= Capacity of the wholesale'''
        for k in range(self.K):
            sum_(b[l][s][k] for l in range(self.L) for s in range(self.S)) <= D[k]*p[k]
            
        '''Outflow from a wholesale store <= Capacity of the wholesale'''
        for k in range(self.K):
            sum_(f[l][k][j] for l in range(self.L) for j in range(self.J)) <= D[k]*p[k]
            
        '''Outflow from a wholesale store <= Inward flow into the wholesale store'''
        for l in range(self.L):
            for k in range(self.K):
                sum_(f[l][k][j] for j in range(self.J)) <=\
                sum_(b[l][s][k] for s in range(self.S))
            
        '''Number of plants that are opened'''
        sum_(p[k] for k in range(self.K)) <= self.K

    def printSolution(self):
        b, f, q, z, y, p = self.b, self.f, self.q, self.z, self.y, self.p
        cpw, cwd, cdc, eta, h = self.cpw, self.cwd, self.cdc, self.eta, self.h

        try:
            _ = self.getObjVal()
    
            for l in range(self.L):
                for s in range(self.S):
                    for k in range(self.K):
                        if b[l][s][k].val != 0:
                            print('b[{}][{}][{}] = {}'.format(l, s, k, b[l][s][k].val))
    
            for l in range(self.L):
                for k in range(self.K):
                    for j in range(self.J):
                        if f[l][k][j].val != 0:
                            print('f[{}][{}][{}] = {}'.format(l, k, j, f[l][k][j].val))
                        
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        if q[l][j][i].val != 0:
                            print('q[{}][{}][{}] = {}'.format(l, j, i, q[l][j][i].val))
                        
            for j in range(self.J):
                if z[j].val != 0:
                    print('z[{}] = {}'.format(j, z[j].val))
                    
            for j in range(self.J):
                for i in range(self.I):
                    if y[j][i].val != 0:
                        print('y[{}][{}] = {}'.format(j, i, y[j][i].val))
                        
            for k in range(self.K):
                if p[k].val != 0:
                    print('p[{}] = {}'.format(k, p[k].val))
                    
            self.wholesalers_operating_cost1 = 0
            for k in range(self.K):
                self.wholesalers_operating_cost1 += g[k]*p[k].val
            print('Operating cost for wholesale stores = {:.2f}'.format(self.wholesalers_operating_cost1))
            
            self.DCs_operating_cost1 = 0
            for j in range(self.J):
                self.DCs_operating_cost1 += v[j]*z[j].val
            print('Operating cost for DCs = {:.2f}'.format(self.DCs_operating_cost1))
            
            self.Tp_cost_wholesaler_to_DC1 = 0
            for l in range(self.L):
                for k in range(self.K):
                    for j in range(self.J):
                        self.Tp_cost_wholesaler_to_DC1 += cwd[l][k][j]*f[l][k][j].val
            print('Transportation cost from wholesale stores to DCs = {:.2f}'.format(self.Tp_cost_wholesaler_to_DC1))
            
            self.Tp_cost_plant_to_wholesaler1 = 0
            for l in range(self.L):
                for s in range(self.S):
                    for k in range(self.K):
                        self.Tp_cost_plant_to_wholesaler1 += cpw[l][s][k]*b[l][s][k].val
            print('Transportation cost from plants to wholesalers = {:.2f}'.format(self.Tp_cost_plant_to_wholesaler1))
            
            self.Tp_cost_DC_to_customer1 = 0
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        self.Tp_cost_DC_to_customer1 += eta[l][i]*cdc[l][j][i]*q[l][j][i].val
            print('Transportation cost from DCs to customers = {:.2f}'.format(self.Tp_cost_DC_to_customer1))
            
            self.Inventory_holding_cost1 = 0
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        self.Inventory_holding_cost1 += (1 - eta[l][i])*h[l][j]*q[l][j][i].val
            print('Inventory holding cost = {:.2f}'.format(self.Inventory_holding_cost1))
            
            self.total_delivered1 = 0
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        self.total_delivered1 += eta[l][i]*x[j][i]*q[l][j][i].val
                        #print('{}, {}, {}'.format(l, j, i))
            self.total_order1 = 0
            for l in range(self.L):
                for i in range(self.I):
                    self.total_order1 += eta[l][i]*d[l][i]
                    
            print('Demand satisfaction = {:.1f}%'.format(self.total_delivered1*100/self.total_order1))
            
            self.values = [self.wholesalers_operating_cost1, self.DCs_operating_cost1, self.Tp_cost_wholesaler_to_DC1,\
                     self.Tp_cost_plant_to_wholesaler1, self.Tp_cost_DC_to_customer1,\
                     self.Inventory_holding_cost1]
            
            print('Total cost ={:.2f}'.format(sum(self.values)))
        except:
            print("No feasible solution found")
            
    def plotSolution(self):
        fig = plt.figure()
#        labels1 = ['CFAs Operating cost', 'SDs Operating cost', 'Transportation cost: \nCFAs'+r'$\rightarrow$'+'SDs',\
#                  'Transportation cost: \nfactories'+r'$\rightarrow$'+'CFAs', 'Transportation cost: \nSDs'+r'$\rightarrow$'+'distributors',\
#                  'Inventory holding cost']
        
        labels2 = ['Wholesale stores \noperating cost \n= {:.2f}'.format(self.wholesalers_operating_cost1),\
                   'DCs Operating cost \n= {:.2f}'.format(self.DCs_operating_cost1), \
                   'Transportation cost: \nwholesales'+r'$\rightarrow$'+'DCs \n= {:.2f}'.format(self.Tp_cost_wholesaler_to_DC1),\
                  'Transportation cost: \nplants'+r'$\rightarrow$'+'wholesalers \n= {:.2f}'.format(self.Tp_cost_plant_to_wholesaler1), \
                  'Transportation cost: \nDCs'+r'$\rightarrow$'+'customers \n= {:.2f}'.format(self.Tp_cost_DC_to_customer1),\
                  'Inventory holding cost \n= {:.2f}'.format(self.Inventory_holding_cost1)]
        
        #explode = (0.1, 0, 0, 0)  # explode 1st slice
        plt.pie(self.values, 
                labels=labels2, 
                autopct='%1.1f%%', 
                shadow=False, 
                textprops=dict(color='k'),
                startangle=140)
        #plt.legend(labels, loc="upper right",)
        plt.axis('equal')
        #plt.tight_layout()
        Total_cost = sum(self.values)
        fig_title = 'Total cost = {0:.2f}'.format(Total_cost)
        fig.suptitle(fig_title, fontsize=12, color = 'k')
        plt.savefig('Results/Piechart_supply_chain_cost.png', dpi = 400)
        plt.show()
        
    def save_result(self):
        b, f, q, z, p = self.b, self.f, self.q, self.z, self.p

        list_plant = self.All_list[0] # List of Factories
        list_wholesaler = self.All_list[1] # List of CFAs
        list_DC = self.All_list[2] # List of SDs
        list_customer = self.All_list[3] # List of Distributors
        
        list_wholesaler_new = [k for k in list_wholesaler if p[list_wholesaler.index(k)].val]
        list_DC_new = [j for j in list_DC if z[list_DC.index(j)].val]
        
        l = 0
        result_dict_b = {}
        writer = pd.ExcelWriter('Results/Plant_to_Wholesaler.xlsx', engine = 'xlsxwriter')
        for l in range(self.L):
            for k in range(len(list_wholesaler)):
                result_list = []
                for s in range(len(list_plant)):
                    #if p[k].val != 0:
                    result_list.append(b[l][s][k].val)
                result_dict_b[list_wholesaler[k]] = result_list
            result_df = pd.DataFrame(result_dict_b, index = list_plant)
            result_df.to_excel(writer, sheet_name='product_{}'.format(l+1))
        writer.save()
            
        result_dict_f = {}
        writer = pd.ExcelWriter('Results/Wholesaler_to_DC.xlsx', engine = 'xlsxwriter')
        for l in range(self.L):
            for j in range(len(list_DC)):
                result_list = []
                for k in range(len(list_wholesaler)):
                    #if z[j].val != 0:
                    result_list.append(f[l][k][j].val)
                result_dict_f[list_DC[j]] = result_list
            result_df = pd.DataFrame(result_dict_f, index = list_wholesaler)
            result_df.to_excel(writer, sheet_name='product_{}'.format(l+1))
        writer.save()
        
        result_dict_q = {}
        writer = pd.ExcelWriter('Results/DC_to_Customer.xlsx', engine = 'xlsxwriter')
        for l in range(self.L):
            for i in range(len(list_customer)):
                result_list = []
                for j in range(len(list_DC)):
                    #if z[j].val != 0:
                    result_list.append(q[l][j][i].val)
                result_dict_q[list_customer[i]] = result_list
            result_df = pd.DataFrame(result_dict_q, index = list_DC)
            result_df.to_excel(writer, sheet_name='product_{}'.format(l+1))
        writer.save()

# Create a directory named 'Results' to save the output results
if not os.path.exists('Results'):
    os.mkdir('Results')  

plant = pd.read_excel('Data.xlsx', sheet_name = 'plant')
sup = np.transpose(plant.values)

wholesaler = pd.read_excel('Data.xlsx', sheet_name = 'wholesaler')
g = wholesaler['operating_cost'].values
D = wholesaler['capacity'].values

DC = pd.read_excel('Data.xlsx', sheet_name = 'DC')
v = DC['operating_cost'].values
W = DC['capacity'].values
h = DC['holding_cost'].values

demand = pd.read_excel('Data.xlsx', sheet_name='customer_demand')
d = np.transpose(demand.values)

list_plant = plant.index.tolist() # List of Plants
list_wholesaler = wholesaler.index.tolist() # List of wholesalers
list_DC = DC.index.tolist() # List of DCs
list_customer = demand.index.tolist() # List of Customers
All_list = [list_plant, list_wholesaler, list_DC, list_customer]

'''Transportation cost'''
Transportation_cost_pw = pd.read_excel('Data.xlsx', sheet_name='Transportation_cost_pw')
cpw = Transportation_cost_pw.values
cpw = np.true_divide(cpw, 5)

Transportation_cost_wd = pd.read_excel('Data.xlsx', sheet_name='Transportation_cost_wd')
cwd = Transportation_cost_wd.values
cwd = np.true_divide(cwd, 5)

Transportation_cost_dc = pd.read_excel('Data.xlsx', sheet_name='Transportation_cost_dc')
cdc = Transportation_cost_dc.values
cdc = np.true_divide(cdc, 5)

distance = pd.read_excel('Data.xlsx', sheet_name='distance')
dist = distance.values

tau = 24 #in hours
velocity = 60 # average velocity of the transport vehicle
threshold_dist = tau*velocity
x = np.zeros((len(v), len(d[0])))
for j1 in range(len(v)):
    for i1 in range(len(d[0])):
        if dist[j1][i1] <= threshold_dist:
            x[j1][i1] = 1

# service level
epsilon = 0.9

# Ratio of ordered amount to the demand amount given at distributor i
eta_value = 0.9 

eta = [eta_value for i in range(len(d[0]))]
No_of_products = len(d)
eta = np.tile(eta, (No_of_products, 1)) 

prob = supply_chain_optimization("Supply Chain Optimization")
prob.model(g, v, h, cpw, cwd, cdc, sup, D, W, d, x, epsilon, eta, All_list)
try:
    prob.optimize()
    prob.printSolution()
    prob.save_result()
    prob.plotSolution()
except:
    print('Problem in optimization')

    

