# Temporal difference methods

&nbsp;&nbsp;&nbsp;In this section, we're going to see a new family of methods that learn from experience. These methods are the basis of many advanced algorithms, and we'll see the concepts that we'll learn here throughout the rest of the course. They are called `temporal difference` methods. They are a family of methods that learn the optimal values based on experience. Just like Monte Carlo methods.

![](../Assets/photos/Temporal%20difference_1.PNG)

&nbsp;&nbsp;&nbsp;In fact, temporal difference methods combine features of Monte Carlo. Methods with features of dynamic programming.

![](../Assets/photos/Temporal%20difference_2.PNG)

&nbsp;&nbsp;&nbsp;As in Monte Carlo methods, the agent will face the environment generating a trajectory with the states visited, the actions taken and the rewards obtained at the end of the episode. We'll use that experience to update the Q-value estimates and the policy. Also, as in Monte Carlo methods, we do not have a model of the environment dynamics.

![](../Assets/photos/Temporal%20difference_3.PNG)

&nbsp;&nbsp;&nbsp;Since we don't have a model, we can't use the state values to guide the policy because for that we'd need the environment dynamics. Instead, we'll use the Q values to guide the policy. Because the Q values implicitly estimate the environment dynamics. In that way, the policy will simply choose the action with the highest Q value.

![](../Assets/photos/Temporal%20difference_4.PNG)

&nbsp;&nbsp;&nbsp;Also, as in dynamic programming, we'll use `bootstrapping`. That is, we will rely on estimates of the Q values to produce new estimates that will be more accurate. This is the update rule of the first method that we are going to learn called `SARSA`. As you see, we update a value. The estimate of a Q value using the estimate of another Q value. The value representing the next action taken in the next state achieved. And this technique is known as `bootstrapping`.

![](../Assets/photos/Temporal%20difference_5.PNG)

&nbsp;&nbsp;&nbsp;As in the Monte Carlo and dynamic programming methods will follow the generalized policy iteration template in which policy evaluation and policy improvement will alternate. Remember that these two processes compete with each other, but they also drive each other towards the optimal q-values and optimal policy.

![](../Assets/photos/Temporal%20difference_6.PNG)

&nbsp;&nbsp;&nbsp;Monte Carlo methods wait until the end of the episode to update the Q values. They wait because they need to compute the return for each moment in time, and for that we need all the rewards obtained after that instant in time. Unlike Monte Carlo methods, temporal difference methods perform this cycle of policy evaluation and improvement. Every time that we take an action during the episode without waiting until the end. In this way, the learning process is constant and uniform. That's a big advantage because the learning that we do at the beginning of the episode influences the policy during the rest of the episode, improving its decision making, and that is simply not possible with Monte Carlo methods.

![](../Assets/photos/Temporal%20difference_7.PNG)



# Solving control tasks with temporal difference methods

&nbsp;&nbsp;&nbsp;We are going to see how `temporal difference` methods solve control tasks. As we said before, we are going to have a value table with an entry for each combination of state and action. Each entry will contain the estimated Q value for that combination.

![](../Assets/photos/Temporal%20difference_8.PNG)

&nbsp;&nbsp;&nbsp;Now remember the bellman equations. The Q value is the expected return of a trajectory that starts in state's in the current state and taking the current action. This expected return can be written as the probability of reaching each successor state. After taking the action times the reward achieved when we reach that state, plus the discounted return starting from that state, which is the value of that state. The value of the next state can be written as the probability of taking each action by the policy times the Q value of that action, which is that actions expected return.

![](../Assets/photos/Temporal%20difference_9.PNG)

&nbsp;&nbsp;&nbsp;Thanks to this expression, we can use values that the agent collects interacting with the environment to estimate the Q values. Every time that the agent takes an action, we'll use the reward that it obtains the next state that it reaches, the action that the policy chooses for that next state. And the Q value table with the Q value estimates. In order to produce an estimate of that Q value.

![](../Assets/photos/Temporal%20difference_10.PNG)

&nbsp;&nbsp;&nbsp;With those elements, we can estimate the Q value this way as the reward obtained, plus a discounted estimate of the next Q value.

![](../Assets/photos/Temporal%20difference_11.PNG)

&nbsp;&nbsp;&nbsp;But now we have two separate estimates. The old one and a new one that incorporates real information from the environment. We'll compare these two estimates, and the difference will be called the temporal difference error.

![](../Assets/photos/Temporal%20difference_12.PNG)

&nbsp;&nbsp;&nbsp;Temporal difference methods use this error, this difference to update the Q value estimates, and they'll do it according to this formula.

![](../Assets/photos/Temporal%20difference_13.PNG)

&nbsp;&nbsp;&nbsp;If you're having deja vu. That's because this is how constant Alpha Monte Carlo works. We push the estimate in the direction of the new return observed by a certain percentage Alpha. The difference is that now we're estimating the return. As this expression, the reward obtained immediately after executing the action, plus the discounted value of the next action in the next state. And this will allow us to update the estimate for the Q value right after we take the action at the next moment in time.

![](../Assets/photos/Temporal%20difference_14.PNG)

&nbsp;&nbsp;&nbsp;In this slide, we can see more clearly what happens if we distribute Alpha and regroup the terms. We can express the update rule as follows. Now we can see that the new estimate is a weighted average between the old estimate and the new one. The new estimate of the Q value will be alpha percent. The new estimate plus one minus alpha percent. The old estimate. If Alpha is 20%, the previous estimate will represent 80% of the new estimate. And the estimate based on experience? 20%.

![](../Assets/photos/Temporal%20difference_15.PNG)



# Monte Carlo vs temporal difference methods

&nbsp;&nbsp;&nbsp;We're going to take a practical look at the difference between `Monte Carlo` and `temporal difference` methods. We made an agent face an episode of the five by five Maze using a monte Carlo method and a temporal difference method with the Monte Carlo method.  
After performing the first move, the Q value table remains unchanged after performing five actions. The table remains the same. Even after taking 120 actions. The table remains the same as at the beginning, because Monte Carlo methods need to wait until the end of the episode to update the Q values.

![](../Assets/photos/Temporal%20difference_16.PNG)
![](../Assets/photos/Temporal%20difference_17.PNG)
![](../Assets/photos/Temporal%20difference_18.PNG)
![](../Assets/photos/Temporal%20difference_19.PNG)

&nbsp;&nbsp;&nbsp;On the other hand, temporal difference methods can start to update the Q-value table immediately after taking the first action. What this means is that the actions taken at the beginning of the episode start influencing the behavior of the agent immediately. After 20 actions. The algorithm has visited many states and has updated all these Q values, and after 120 moves, it will have updated all these values. The estimated values at this point are not perfect. They might not be even good, but the optimization process has begun without waiting until the end of the episode to perform the updates all at once.

![](../Assets/photos/Temporal%20difference_20.PNG)
![](../Assets/photos/Temporal%20difference_21.PNG)
![](../Assets/photos/Temporal%20difference_22.PNG)
![](../Assets/photos/Temporal%20difference_23.PNG)


# SARSA

&nbsp;&nbsp;&nbsp;We're going to introduce the first temporal difference method that we are going to implement called `SARSA`. It follows an `on policy` exploration strategy. What this means is that we'll keep a single policy that will be responsible for both exploring the environment and taking part in the optimization process. Remember that in order to find the optimal actions, we need to keep exploring the effect of all the actions.

![](../Assets/photos/Temporal%20difference_24.PNG)

&nbsp;&nbsp;&nbsp;This algorithm will perform the exploration by sometimes picking a random action. Whenever it's time to pick an action, we'll flip a coin with probability. Epsilon will pick a random action, and with probability, one minus Epsilon will pick the action with the highest estimated Q value.

![](../Assets/photos/Temporal%20difference_25.PNG)

&nbsp;&nbsp;&nbsp;The name `SARSA` comes from the five elements involved in the update rule. The State at time t the action taken in that state. The reward obtained immediately after taking that action the next state achieved after taking the action and also the action that the policy would choose for that successor state. These five elements form the acronym `SARSA`. Based on these five elements, the temporal difference error is calculated. And used to refine the estimate of the Q value in our table. We'll do that by moving our estimate of the Q value in the direction of the new estimate collected using experience from the environment in a certain percent alpha.

![](../Assets/photos/Temporal%20difference_26.PNG)

&nbsp;&nbsp;&nbsp;Notice again that the same policy that explores the environment is the policy that picks the next action in the update rule. This will separate from the next algorithm that we'll learn.

![](../Assets/photos/Temporal%20difference_27.PNG)

&nbsp;&nbsp;&nbsp;This is the algorithm.  
First, we'll initialize our policy, which will be an epsilon greedy policy, as we mentioned before. And we'll also initialize the q-value table with the estimates of the q-values for each action in each state. Then we'll enter the main loop that will repeat for a number of episodes. In each episode, we'll initialize the task and observe the initial state that we call S0. Next, we'll pick an action for that state following the policy, the policy that we initialized here, and then we'll enter an inner loop that will execute for every moment in time until the episode finishes. And at each moment in time, we'll execute the action that we picked for the present state and we'll observe the next state achieved and the reward obtained. Then for the next state achieved, we'll pick an action according to the policy. And right after we'll update the Q value estimate of the original state and the action taken. This action here using the update rule that we saw before. The new estimate will be the old estimate plus alpha times the temporal difference error. And that's all.  
&nbsp;&nbsp;&nbsp;When the algorithm finishes, we'll have a near optimal policy and Q value table of estimates near optimal. Because our policy sometimes picks a random action. So our policy will never get to be the optimal policy, but it can approximate it very closely.

![](../Assets/photos/Temporal%20difference_28.PNG)

https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_5_sarsa_complete.ipynb#scrollTo=mZKL4RXvqBFT



# Q-Learning

&nbsp;&nbsp;&nbsp;We're going to see the second algorithm that learns from temporal differences called `q-learning`. This method follows an `off policy` learning strategy. What this means is that we'll have two separate policies. One of them to participate in the optimization process and the other to explore the environment. The policy that participates in the optimization process is called Pi, and it will be a greedy policy, which means that it will always select the action with the highest estimated Q value. On the other hand, the exploratory policy will be called B, and we'll use it to face the environment and collect experience samples.

![](../Assets/photos/Temporal%20difference_29.PNG)

&nbsp;&nbsp;&nbsp;This is the update rule. As you can see, it's pretty much identical to SAS, except that now we have two policies. We'll choose the next action based on the target policy. Since the exploratory policy has only the role of collecting the experience and interacting with the environment, and as you know, the target policy chooses the action in a greedy manner. It selects the action with the highest estimated Q value.

![](../Assets/photos/Temporal%20difference_30.PNG)

&nbsp;&nbsp;&nbsp;The rest of the algorithm is virtually identical to `SARSA`. As I said, we initialized two separate policies. The target policy, which is greedy and the exploratory policy as well as the Q value table. And then we'll enter the main loop, the loop that will repeat for a number of episodes. In each episode, we'll restart the task and observe the initial state. Then we enter an inner loop that will execute for every moment in time until the end of the episode. At each moment in time, we'll pick an action using the exploratory policy. We'll execute that action in the environment and we'll observe the next state and the reward obtained, and then we'll update the Q value estimates according to the update rule that we saw before. And that's it. When the algorithm finishes, we'll have the optimal policy and the optimal Q value estimates.

![](../Assets/photos/Temporal%20difference_31.PNG)

https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_5_qlearning_complete.ipynb#scrollTo=5jMXrtsQrioA



# Advantages of temporal difference methods

&nbsp;&nbsp;&nbsp;We're going to consider the advantages that `temporal difference learning` offers versus `Monte Carlo` methods and `dynamic programming`.  
Compared to Monte Carlo methods. Temporal difference learning allows us to update the Q values while the experience is being collected. This means that the decision making of the agent can be improved during the episode without having to wait until the end. In practice, methods that use temporal difference learning find the optimal values and Q tables faster.

![](../Assets/photos/Temporal%20difference_32.PNG)

&nbsp;&nbsp;&nbsp;Compared to dynamic programming. Temporal difference learning is more efficient because we can focus the effort on the states that lead to the goals. And also temporal difference methods are applicable to a wider range of control tasks as they don't need a model of the environment dynamics. They learn based on samples that they collect from the environment.

![](../Assets/photos/Temporal%20difference_33.PNG)










































































