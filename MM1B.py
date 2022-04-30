# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:19:21 2022

@author: franc
"""

# ******************************************************************************
#
# This is a discrete-event simulator (event scheduling approach) of a queue
# - single-server 
# - finite capacity of the waiting line
# - exponential inter arrival times
# - exponential service times
#
# 
# ******************************************************************************
import random
from queue import PriorityQueue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# ******************************************************************************
# To take the measurements
#
# Collect
# - total number of arrivals, Narr
# - total number of departures, Ndep
# - integral of the number of client in time
# - store the time of the last event (for computing the integral)
# - total delay in the queue 
# - number of lost customers
# ******************************************************************************
class Measure:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay,LostCustomer):
    # , idleStart = [0], idleEnd= [], Ndelay= 0):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay

        self.LostCustomer= LostCustomer
        
# ******************************************************************************
# Client
# 
# Identify the client with
# - type: for future use
# - time of arrival (for computing the delay, i.e., time in the queue)
# ******************************************************************************
class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time


# ******************************************************************************
# ARRIVAL: event function
# 
# Receive in input 
# - the FES, for possibly schedule new events
# - the queue of the clients
# ******************************************************************************
def arrival(time, FES, queue, B):
    global users
    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next arrival
    inter_arrival = random.expovariate(lambd=1.0/ARRIVAL)
    
    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))
    
    # if no. users inferior to the queue capacity
    if users < B:
        # update the state variable, by increasing the no. of clients by 1
        users += 1

    
        # create a record for the client
        client = Client(TYPE1,time)

        # insert the record in the queue
        queue.append(client)


        # if the server is idle start the service
        if users==1:      
            # sample the service time
            service_time = random.expovariate(1.0/SERVICE)
            # schedule the departure of the client
            FES.put((time + service_time, "departure"))

    else:
        # update the number of lost custumers
        data.LostCustomer +=1

        


# ******************************************************************************
def departure(time, FES, queue):
    global users
    

    # get the first element from the queue
    client = queue.pop(0)
        
    # cumulate statistics
    data.dep += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time
    delay = (time-client.arrival_time)
    data.delay += delay
    


    # update the state variable, by decreasing the no. of clients by 1
    users -= 1
    
    # check whether there are more clients to in the queue
    if users >0:
        # sample the service time
        service_time = random.expovariate(1.0/SERVICE)
        # schedule the departure of the client
        FES.put((time + service_time, "departure"))

    

# ******************************************************************************
# Input Variables
# ******************************************************************************


# Initialize the random number generator  
random.seed(42)            




# ******************************************************************************
# Event-loop 
# ******************************************************************************
#

def event_loop(LOAD, B):  
    global users, ARRIVAL, SERVICE, data, TYPE1
    SIM_TIME = 1e6
    SERVICE = 10.0 # av service time
    TYPE1 = 1 # At the beginning all clients are of the same type, TYPE1 

    ARRIVAL = SERVICE/LOAD # av. inter-arrival time
  # Collect measurements
    data = Measure(0,0,0,0,0,0)
    
    
  # State variable: number of users
    users=0  
   
  # Queue of the clients
    queue=[]  
  

  # Future Event Set: the list of events in the form: (time, type)
    FES = PriorityQueue()
    
    
  
  # schedule the first arrival at t=0
    FES.put((0, "arrival"))    
    (time, event_type) = FES.get()
    
  # Actual event loop
    while time < SIM_TIME:
        


    # Call the event functions based on the event type
        if event_type == "arrival":
            arrival(time, FES, queue, B)

        elif event_type == "departure":
            departure(time, FES, queue)
            


    # Extract next event from the FES    
        (time, event_type) = FES.get()
        
    #End of the loop
    return data
    

# ******************************************************************************
# Input Variables
# ******************************************************************************

#Queue capacity values
B_list = range(1, 11)
#Loads values
LOAD_list = [0.2, 0.4, 0.6, 0.8]

# Initialize the random number generator  
random.seed(42)


# ******************************************************************************
# Simulate
# ******************************************************************************
DATA = []

for B in B_list:
    for LOAD in LOAD_list:
        for run in range(10):
            data = event_loop(LOAD, B)
            DATA.append((B, LOAD, data.delay/data.dep, data.LostCustomer/data.arr))
stats = pd.DataFrame(DATA, columns=["B", "load",  "avg delay", "loss prob"])


# averaging stats over runs with different seeds
avg_stats = stats\
.groupby(["B" , "load"])\
.mean()\
.reset_index()

#computing sample std of stats over runs with different seeds
avg_stats_stds = stats\
.groupby(["B" , "load"])\
.std(ddof = 1).reset_index()

# computing confidence intervals semidepths of stats over runs with different seeds
metrics = avg_stats.columns[2:]
avg_stats_ci = avg_stats.copy(deep = True)

for metric in metrics:
    ci_semidepths = []
    for avg, std in zip(avg_stats[metric],  avg_stats_stds[metric]):
        if std !=0:
    #comupting confidence intervals
            CI = t.interval(0.95, 9, avg, std)
        #comupting confidence intervals semidepths
            semidepth = (CI[1] - CI[0])/2
        else:
            semidepth = 0
        ci_semidepths.append(semidepth)
    avg_stats_ci[metric] = ci_semidepths
    
    
# ******************************************************************************
# Plot outputs
# ******************************************************************************

fig, ax = plt.subplots(2, 2, sharey = True, figsize = (12,7))
for i, LOAD in enumerate(LOAD_list):
    # Theoretical Loss probability for B capacity
    Th_loss_prob = list(map(lambda B: LOAD**B*(1-LOAD)/(1-LOAD**(B+1)), B_list))
    
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
    upper = avg_stats.loc[avg_stats["load"] == LOAD, ["loss prob"]] + avg_stats_ci.loc[avg_stats["load"] == LOAD, ["loss prob"]]
    upper = upper.values.squeeze()

    lower = avg_stats.loc[avg_stats["load"] == LOAD, ["loss prob"]] - avg_stats_ci.loc[avg_stats["load"] == LOAD, ["loss prob"]]
    lower = lower.values.squeeze()

    ax[i // 2,  i % 2].plot(B_list, avg_stats.loc[avg_stats["load"] == LOAD, ["loss prob"]], "*-", label = "Simulated")
    ax[i // 2,  i % 2].fill_between(B_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)

    ax[i // 2,  i % 2].plot(B_list, Th_loss_prob, label = "Theoretical")



    ax[i // 2,  i % 2].set_title("Load: {}".format(LOAD))
    ax[i // 2,  i % 2].set_xlabel("B")
    ax[i // 2,  i % 2].set_ylabel("Lost customer prob")
    ax[i // 2,  i % 2].grid()


plt.legend()    
plt.tight_layout()
plt.savefig('Finite B Loss prob.png', bbox_inches='tight')







fig, ax = plt.subplots(2, 2, sharey = True, figsize = (12,7))
for i, LOAD in enumerate(LOAD_list):
    ARRIVAL = SERVICE/LOAD
    Th_avg_delay = []
    #computing average delay for B
    for B in B_list:
        #compute average customer number
        avg_cust_num = sum([j*(1-LOAD)*LOAD**j/(1-LOAD**(B+1)) for j in range(B+1)])
        #probability of being in state B
        PI_b = LOAD**B*(1-LOAD)/(1-LOAD**(B+1))
        #enter rate as difference of arrival rate and loss rate (prob. of being in state B divided by arrival rate)
        enter_rate = 1/ARRIVAL - PI_b/ARRIVAL
        Th_avg_delay.append(avg_cust_num/enter_rate)
        
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth        
    upper = avg_stats.loc[avg_stats["load"] == LOAD, ["avg delay"]] + avg_stats_ci.loc[avg_stats["load"] == LOAD, ["avg delay"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["load"] == LOAD, ["avg delay"]] - avg_stats_ci.loc[avg_stats["load"] == LOAD, ["avg delay"]]
    lower = lower.values.squeeze()
    
    ax[i // 2,  i % 2].plot(B_list, avg_stats.loc[avg_stats["load"] == LOAD, ["avg delay"]], "*-", label = "Simulated")
    ax[i // 2,  i % 2].fill_between(B_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    ax[i // 2,  i % 2].plot(B_list, Th_avg_delay, label = "Theoretical")

    
    
    
    ax[i // 2,  i % 2].set_title("Load: {}".format(LOAD))
    ax[i // 2,  i % 2].set_xlabel("B")
    ax[i // 2,  i % 2].set_ylabel("Avg delay")
    ax[i // 2,  i % 2].grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig('Finite B delays.png', bbox_inches='tight')