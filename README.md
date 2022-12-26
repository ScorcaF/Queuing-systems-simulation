# Queuing-systems-simulation

The work involves the analyses of different types of queues to collect some performance metrics,
compared to the ones from the analytical models when possible. 

The considered queues types are the following:
- **MM1**:
  - The inter-arrivals times are independent and exponentially distributed with average frequency λ.
  - The service times are independent and exponentially
distributed with average frequency µ = 0.1.
  - The queue capacity "B" is infinite.
  - There is exactly 1 server.
![mm1](https://user-images.githubusercontent.com/70110839/209571666-2d99bc66-8ef0-4e55-9063-5f0dad4de99a.png)
- **MM1B**:
  - The inter-arrivals times are independent and exponentially distributed with average frequency λ.
  -  The service times are independent and exponentially
distributed with average frequency µ.
  - B from 1 to 10.
  - ρ = λ/µ from 0.2 to 0.8, with step 0.2.
![mm1B](https://user-images.githubusercontent.com/70110839/209571668-85200a9a-c8a4-4139-b9ed-5df2f41f3855.png)
- **MM1 with failure and reparation possibility**:
 - The assumptions are the same of the MM1 queue, with the
   addition of the fact that inter-failure time and the repair time are independent and exponentially distributed with
   average frequencies respectively γ and σ. Notice that since the introduction of these additional events the queue can not be defined anymore as "MM1".
   
Different input parameters are tested to perform analysis on "average delay", "users distribution", "probability of loosing a customer", etc. .
The work is summarized in https://github.com/ScorcaF/Queuing-systems-simulation/blob/12d118193123c177dffb032760a6dbd610e207be/LABQ_Scorca_Francesco.pdf 
