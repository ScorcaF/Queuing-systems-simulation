# -*- coding: utf-8 -*-

# ******************************************************************************
#
# This is a discrete-event simulator (event scheduling approach) of a queue
# - single-server 
# - infinite capacity of the waiting line
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
# - total number of times the delay is below a threshold (10, 50, 100)
# ******************************************************************************
class Measure:
    def __init__(self,Narr,Ndep,NAveraegUser,OldTimeEvent,AverageDelay, delay10, delay50, delay100):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay

        self.delay10= delay10
        self.delay50= delay50
        self.delay100= delay100
        
        
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
def arrival(time, FES, queue):
    global users, SERVICE, ARRIVAL, data
    # cumulate statistics
    data.arr += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time

    # sample the time until the next arrival
    inter_arrival = random.expovariate(lambd=1.0/ARRIVAL)
    
    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    # update the state variable, by increasing the no. of clients by 1
    users += 1
    
    # create a record for the client
    client = Client(TYPE1,time)

    # insert the record in the queue
    queue.append(client)


    # if the server is idle start the service
    if users==1:      
        # data.idleEnd.append(time)
        # sample the service time
        service_time = random.expovariate(1.0/SERVICE)
        # schedule the departure of the client
        FES.put((time + service_time, "departure"))



        


# ******************************************************************************
def departure(time, FES, queue):
    global users, SERVICE, data
    
    # simulated thresholds
    DELAY_THR_list = [10, 50, 100]
    # get the first element from the queue
    client = queue.pop(0)
        
    # cumulate statistics
    data.dep += 1
    data.ut += users*(time-data.oldT)
    data.oldT = time
    delay = (time-client.arrival_time)
    data.delay += delay
    
    # Verifing if delay is below the simulated thresholds
    if delay < DELAY_THR_list[0]:
        data.delay10 +=1
    if delay < DELAY_THR_list[1]:
        data.delay50 +=1
    if delay < DELAY_THR_list[2]:
        data.delay100 +=1


    # update the state variable, by decreasing the no. of clients by 1
    users -= 1
    
    # check whether there are more clients to in the queue
    if users >0:
        # sample the service time
        service_time = random.expovariate(1.0/SERVICE)
        # schedule the departure of the client
        FES.put((time + service_time, "departure"))

    


             

# ******************************************************************************
# Event-loop 
# ******************************************************************************
#

def event_loop(LOAD, SIM_TIME, compute_distr = False):
    global users, ARRIVAL, SERVICE, data, TYPE1
    SERVICE = 10.0 # av service time
    TYPE1 = 1 # At the beginning all clients are of the same type, TYPE1 

    ARRIVAL = SERVICE/LOAD # av. inter-arrival time
  # Collect measurements
    data = Measure(0,0,0,0,0,0, 0, 0)
    
    
  # State variable: number of users
    users=0  
   
  # Queue of the clients
    queue=[]  
  

  # Future Event Set: the list of events in the form: (time, type)
    FES = PriorityQueue()
    
    if compute_distr == True:
        #set num of users to track their distribution, if the users in the system are more than this, consider this value as the maximum one
         max_users = 10
         # list of empty arrays, each one with two entries:
        # the first entry represent the initial time, the second the final time (at which we have no. users = index in the list)
        # while the index in the list corresponds to the number of clients 
         users_times = [np.array([]).reshape(0,2) for i in range(max_users+1)]
         
         #insert record for first array for the first client
         #when inserting this value we do not know the final time, hence we fix it as simulation time
         #the real final time will be updated once known
         users_times[0] = np.vstack((users_times[0], np.array([0,SIM_TIME])))

    
  
  # schedule the first arrival at t=0
    FES.put((0, "arrival"))    
    (time, event_type) = FES.get()
    
  # Actual event loop
    while time < SIM_TIME:
        
        if compute_distr == True:
            #The number of users is about to change
            #Update the final time of the final entry at the array corresponding to the old number of users 
            
            if users < max_users:
                users_times[users][-1, 1] = time
            else:
                users_times[max_users][-1, 1] = time

    # Call the event functions based on the event type
        if event_type == "arrival":
            arrival(time, FES, queue)

        elif event_type == "departure":
            departure(time, FES, queue)
            
        if compute_distr == True:
            #The number of users is changed
            #Insert a new record at the array corresponding to the new number of users
            
            #As before: when inserting this value we do not know the final time, hence we fix it as simulation time
            #the real final time will be updated once known
            if users < max_users:
                users_times[users] = np.vstack((users_times[users], np.array([time, SIM_TIME])))
            else:
                users_times[max_users] = np.vstack((users_times[max_users], np.array([time, SIM_TIME])))

    # Extract next event from the FES    
        (time, event_type) = FES.get()
        
    #End of the loop
        
    if compute_distr == False:
        return data
    
    else:
        #Compute all the time intervals for each no. of users, sum them and normalize to obtain a distribution
        users_distribution = []
        #for each no. of users
        for times in users_times:
            #compute interval length
            intervals = times[:,1]- times[:,0]
            #sum intervals and compute ratio with total sim time
            users_distribution.append(intervals.sum()/SIM_TIME)
            
        return data, users_distribution
    
# ******************************************************************************
# Input Variables
# ******************************************************************************

LOAD_list= list(map(lambda x: x/100, list(range(9, 97, 3)))) # load of the queue: from 0.09 to 0.96 with step 0.03
SIM_TIME_list = [100000, 250000, 500000, 1000000, 1500000, 2000000] # condition to stop the simulation

# Initialize the random number generator  
random.seed(42)

# ******************************************************************************
# Simulate
# ******************************************************************************

DATA = []

for SIM_TIME in SIM_TIME_list:
    for LOAD in LOAD_list:
        for run in range(10):
            data = event_loop(LOAD, SIM_TIME)
            DATA.append((SIM_TIME, LOAD, data.delay/data.dep, data.arr - data.dep,
                          data.arr/SIM_TIME, data.dep/SIM_TIME, data.delay10/data.dep, data.delay50/data.dep, data.delay100/data.dep ))
stats = pd.DataFrame(DATA, columns=["sim time", "load", "avg delay", "remaining clients", 
                                    "arrival", "departure", "below10 prob", "below50 prob", "below100 prob"])

# averaging stats over runs with different seeds
avg_stats = stats\
.groupby(["sim time" , "load"])\
.mean()\
.reset_index()

#computing sample std of stats over runs with different seeds
avg_stats_stds = stats\
.groupby(["sim time" , "load"])\
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
    

# Computing theoretical delay for each load, from Little's Law
Theoretical_delays = list(map(lambda LOAD: 1.0/(1.0/SERVICE - LOAD/SERVICE), LOAD_list))



# ******************************************************************************
# Plot outputs
# ******************************************************************************

fig, ax = plt.subplots(2, 3, sharey = True, figsize = (12,7))
for i, SIM_TIME in enumerate(SIM_TIME_list):
    #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
    upper = avg_stats.loc[avg_stats["sim time"] == SIM_TIME, ["avg delay"]] + avg_stats_ci.loc[avg_stats["sim time"] == SIM_TIME, ["avg delay"]]
    upper = upper.values.squeeze()
    
    lower = avg_stats.loc[avg_stats["sim time"] == SIM_TIME, ["avg delay"]] - avg_stats_ci.loc[avg_stats["sim time"] == SIM_TIME, ["avg delay"]]
    lower = lower.values.squeeze()
    
    ax[i // 3,  i % 3].plot(LOAD_list, avg_stats.loc[avg_stats["sim time"] == SIM_TIME, ["avg delay"]], label = "Simulated")
    ax[i // 3,  i % 3].fill_between(LOAD_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    ax[i // 3,  i % 3].plot(LOAD_list, Theoretical_delays, c = "red", alpha = 0.6, label = "Theoretical")

    
    
    ax[i // 3,  i % 3].set_title("Sim time: {}".format(SIM_TIME))
    ax[i // 3,  i % 3].set_xlabel("Load")
    ax[i // 3,  i % 3].set_ylabel("Avg delay")
    ax[i // 3,  i % 3].grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig('Average delay.png', bbox_inches='tight')


fig, ax = plt.subplots(2, 3, sharey = True, figsize = (12,7))
for i, SIM_TIME in enumerate(SIM_TIME_list):
    for thr in metrics[-3:]:
        #computing extremes of confidence intervals as sum (difference) of average and inverval semidepth
        upper = avg_stats.loc[avg_stats["sim time"] == SIM_TIME, [thr]] + avg_stats_ci.loc[avg_stats["sim time"] == SIM_TIME, [thr]]
        upper = upper.values.squeeze()
    
        lower = avg_stats.loc[avg_stats["sim time"] == SIM_TIME, [thr]] - avg_stats_ci.loc[avg_stats["sim time"] == SIM_TIME, [thr]]
        lower = lower.values.squeeze()
    
        ax[i // 3,  i % 3].plot(LOAD_list, avg_stats.loc[avg_stats["sim time"] == SIM_TIME, [thr]], label = thr)
        ax[i // 3,  i % 3].fill_between(LOAD_list, 
                                    upper,
                                    lower,
                                    alpha = 0.4)
    
    
    
    ax[i // 3,  i % 3].set_title("Sim time: {}".format(SIM_TIME))
    ax[i // 3,  i % 3].set_xlabel("Load")
    ax[i // 3,  i % 3].set_ylabel("Below Thr Prob")
    ax[i // 3,  i % 3].grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig('Below Threshold probs.png', bbox_inches='tight')


# ******************************************************************************
# Input Variables
# ******************************************************************************

LOAD_list = [0.2, 0.4, 0.6, 0.8]


# ******************************************************************************
# Simulate
# ******************************************************************************

DISTRIBUTIONS = []
for LOAD in LOAD_list:
        _, users_distribution = event_loop(LOAD, 1000000, True)
        DISTRIBUTIONS.append(users_distribution)
        
# ******************************************************************************
# Plot outputs
# ******************************************************************************

sim_ticks = list(map(lambda x: x - 0.4, range(11)))
fig, ax = plt.subplots(2, 2, sharey = True, figsize = (12,7))
for i, dist in enumerate(DISTRIBUTIONS):
    LOAD = LOAD_list[i]
    #Theoretical (geometric) distribution
    Th_dist = [ (1 - LOAD)*LOAD**j for j in range(11)]
        
    ax[i // 2,  i % 2].bar(sim_ticks, dist, width = 0.4, label = "Simulated")
    ax[i // 2,  i % 2].bar(range(11), Th_dist, width = 0.4, label = "Theoretical")

    
    
    ax[i // 2,  i % 2].set_title("Load: {}".format(LOAD))
    ax[i // 2,  i % 2].set_xlabel("Num users")
    ax[i // 2,  i % 2].set_ylabel("Probability")
    ax[i // 2,  i % 2].grid()
    

plt.legend()    
plt.tight_layout()
plt.savefig('Users distributions.png', bbox_inches='tight')

