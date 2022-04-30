# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:05:56 2022

@author: franc
"""

# -*- coding: utf-8 -*-

# ******************************************************************************
#
# This is a discrete-event simulator (event scheduling approach) of a queue
# - single-server 
# - infinite capacity of the waiting line
# - exponential inter arrival times
# - exponential service times
# - exponential inter failure times (time until the next failure)
# - exponential repair time 
# 
# ******************************************************************************
import random
from queue import PriorityQueue
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
# - total number of lost customers
# - time the server is down
# ******************************************************************************
class Measure:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay,LostCustomer,ftime, f_users):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay
        
        self.LostCustomer = LostCustomer
        self.ftime = ftime
        self.f_users = f_users


        
        
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
#  EVENTS FUNCTIONS
# 
# ******************************************************************************

def arrival(time, FES, queue):
    global users, down
    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next arrival
    inter_arrival = random.expovariate(lambd=1.0/ARRIVAL)
    
    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    
    if down == False:
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
      data.LostCustomer +=1

        
# ******************************************************************************
def departure(time, FES, queue):
    global users, down, repair_time
    
    #allow deparute only if not in failure state
    if down == False:
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
      
    else:
        FES.put((time + repair_time, "departure"))



# ******************************************************************************
def failure(time,FES):
    global users, down, repair_time
    # cumulate statistics
    down = True

    # sample the repair time
    repair_time = random.expovariate(1.0/REPAIR)
    # schedule the repair of the system
    FES.put((time + repair_time, "repair"))

# ******************************************************************************

def repair(time, FES):
    global users, down
    # cumulate statistics
    down = False

    # sample the time until the next failure
    inter_failure = random.expovariate(1.0/FAILURE)
    # schedule the failure of the system
    FES.put((time + inter_failure, "failure"))



             

# ******************************************************************************
# Event-loop 
# ******************************************************************************


def event_loop(LOAD, RISK, SEVERITY):
    global users, ARRIVAL, SERVICE, data, TYPE1, down, REPAIR, FAILURE
    SIM_TIME = 1e6
    SERVICE = 10.0 # av service time
    TYPE1 = 1 # At the beginning all clients are of the same type, TYPE1 

    ARRIVAL = SERVICE/LOAD # av. inter-arrival time
    
    FAILURE = ARRIVAL/RISK # av. failure time

    REPAIR = FAILURE*SEVERITY # av. repair time
  # Collect measurements
    data = Measure(0,0,0,0,0,0,0,0)
    
    down = False  

  # State variable: number of users
    users=0  
   
  # Queue of the clients
    queue=[]  
  

  # Future Event Set: the list of events in the form: (time, type)
    FES = PriorityQueue()
      
  
  # schedule the first arrival at t=0
    FES.put((0, "arrival"))    
    (time, event_type) = FES.get()
    
   # sample the time until the next failure
    inter_failure = random.expovariate(1.0/FAILURE)
    # schedule the failure of the system
    FES.put((time + inter_failure, "failure"))
    
  # Actual event loop
    while time < SIM_TIME:
        

    # Call the event functions based on the event type
        if event_type == "arrival":
            arrival(time, FES, queue)

        elif event_type == "departure":
            departure(time, FES, queue)
            
        elif event_type == "failure":
            failure(time, FES)
            failure_begin = time
            
        elif event_type == "repair":
            repair(time, FES)
            # clients during failure integral 
            data.f_users += (time - failure_begin)*users
            # time the server was in failure
            data.ftime += (time - failure_begin)
            

    # Extract next event from the FES    
        (time, event_type) = FES.get()
        

        
    #End of the loop
    if down:
        data.f_users += (time - failure_begin)*users
        data.ftime += (SIM_TIME - failure_begin)
        
    return data

# ******************************************************************************
# Input Variables
# ******************************************************************************

LOAD = 0.8
RISK_list = [0.5, 1, 2]
SEVERITY_list = list(map(lambda x: x/2, range(1, 11)))  #[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5]




# ******************************************************************************
# Simulate
# ******************************************************************************
DATA = []
for RISK in RISK_list:
  for SEVERITY in SEVERITY_list:
      for run in range(10):
          data = event_loop(LOAD, RISK, SEVERITY)
          DATA.append((RISK, SEVERITY, data.delay/data.dep, data.LostCustomer/data.arr, data.ftime/1e6, data.f_users/data.ftime , data.ut/1e6))
      
stats = pd.DataFrame(DATA, columns=["risk", "severity", "Avg delay", "Lost customer prob", "Down prob", "Down mean no. users",
                                    "Overall mean no. users"])

# averaging stats over runs with different seeds
avg_stats = stats\
.groupby(["risk", "severity"])\
.mean()\
.reset_index()

#computing sample std of stats over runs with different seeds
avg_stats_stds = stats\
.groupby(["risk", "severity"])\
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
    
    
    
    
plt.figure()
for i, RISK in enumerate(RISK_list):
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
    upper = avg_stats.loc[avg_stats["risk"] == RISK, ["Avg delay"]] + avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Avg delay"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["risk"] == RISK, ["Avg delay"]] - avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Avg delay"]]
    lower = lower.values.squeeze()
    
    plt.plot(SEVERITY_list, avg_stats.loc[avg_stats["risk"] == RISK, ["Avg delay"]], "*-", label = "Risk: {}".format(RISK))
    plt.fill_between(SEVERITY_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    
    
plt.xlabel("Severity")
plt.ylabel(metric)
plt.grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig("Task2 Avg delay.png" , bbox_inches='tight')
    
    

    
    
plt.figure()
for i, RISK in enumerate(RISK_list):
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
    upper = avg_stats.loc[avg_stats["risk"] == RISK, ["Lost customer prob"]] + avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Lost customer prob"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["risk"] == RISK, ["Lost customer prob"]] - avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Lost customer prob"]]
    lower = lower.values.squeeze()
    
    plt.plot(SEVERITY_list, avg_stats.loc[avg_stats["risk"] == RISK, ["Lost customer prob"]], "*-", label = "Loss, Risk: {}".format(RISK))
    plt.fill_between(SEVERITY_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    upper = avg_stats.loc[avg_stats["risk"] == RISK, ["Down prob"]] + avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Down prob"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["risk"] == RISK, ["Down prob"]] - avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Down prob"]]
    lower = lower.values.squeeze()
    
    plt.plot(SEVERITY_list, avg_stats.loc[avg_stats["risk"] == RISK, ["Down prob"]], "s-", label = "Down, Risk: {}".format(RISK))
    plt.fill_between(SEVERITY_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    
    
plt.xlabel("Severity")
plt.ylabel("Probability")
plt.grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig("Task2 Probabilities.png" , bbox_inches='tight')
    



fig, ax = plt.subplots(1,2, sharey = True, figsize = (12,7))
MM1_mean_users = [LOAD/(1 - LOAD) for i in range(len(SEVERITY_list))]
for i, RISK in enumerate(RISK_list):
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
    upper = avg_stats.loc[avg_stats["risk"] == RISK, ["Down mean no. users"]] + avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Down mean no. users"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["risk"] == RISK, ["Down mean no. users"]] - avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Down mean no. users"]]
    lower = lower.values.squeeze()
    
    ax[0].plot(SEVERITY_list, avg_stats.loc[avg_stats["risk"] == RISK, ["Down mean no. users"]], "*-", label = "Risk: {}".format(RISK))
    ax[0].fill_between(SEVERITY_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    
    
    upper = avg_stats.loc[avg_stats["risk"] == RISK, ["Overall mean no. users"]] + avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Overall mean no. users"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["risk"] == RISK, ["Overall mean no. users"]] - avg_stats_ci.loc[avg_stats["risk"] == RISK, ["Overall mean no. users"]]
    lower = lower.values.squeeze()
    
    ax[1].plot(SEVERITY_list, avg_stats.loc[avg_stats["risk"] == RISK, ["Overall mean no. users"]], "s-", label = "Risk: {}".format(RISK))
    ax[1].fill_between(SEVERITY_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
       
    
    
    
ax[0].plot(SEVERITY_list, MM1_mean_users, label = "MM1 no failures")
ax[0].legend()
ax[0].grid()

ax[1].plot(SEVERITY_list, MM1_mean_users, label = "MM1 no failures")
ax[1].legend()
ax[1].grid()


ax[0].set_title("Server down")

ax[0].set_xlabel("Severity")
ax[0].set_ylabel("Mean no. users")

ax[1].set_title("Overall")

ax[1].set_xlabel("Severity")
ax[1].set_ylabel("Mean no. users")
    

        
plt.tight_layout()
plt.savefig("Task2 Users.png" , bbox_inches='tight')