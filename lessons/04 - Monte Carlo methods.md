# Monte Carlo methods

&nbsp;&nbsp;&nbsp;We're going to see the first family of algorithms that will learn based on experience known as `Monte carlo methods`. This is a family of methods that learn the optimal state-values or q-values based on samples of experience collected by the agent while interacting with the environment.  
At the beginning of the learning process, the agent will follow an arbitrary policy. Then the agent will try to perform the task using that policy until the end of the episode, generating these trace of experience. The trace contains the states visited. The actions taken in those states and the rewards obtained as a consequence.

![](../Assets/photos/monte%20carlo_1.PNG)


&nbsp;&nbsp;&nbsp;At the end of the episode will compute the return from every state visited. Which, as you know, is the sum of discounted rewards from the time we visit that state until the end of the episode. The value of a state is the expected return following the present policy. So every time that we observe a new return for a specific state, we'll update the estimated value for that state as the average of all the returns that the agent has collected and that start in that state.

![](../Assets/photos/monte%20carlo_2.PNG)


&nbsp;&nbsp;&nbsp;The same process can be done using q-values, except now we must average the returns produced after taking a specific action in that state.

![](../Assets/photos/monte%20carlo_3.PNG)

&nbsp;&nbsp;&nbsp;By the law of large numbers, the more returns we observe for the state or action, the more we approach its expected value. Simply put, the more experience, the more accurate our estimates.

![](../Assets/photos/monte%20carlo_4.PNG)

&nbsp;&nbsp;&nbsp;Monte carlo methods have some advantages over dynamic programming.  
First, the estimate of one state does not depend on the rest. What that means is that the complexity of estimating the value of a state doesn't depend on the number of states in the task. In the case of dynamic programming to estimate the value of a state, the algorithm bootstraps the value of other states. That is, they use one estimate to produce another estimate. Therefore, the complexity of the algorithms grows exponentially with the number of states.

![](../Assets/photos/monte%20carlo_5.PNG)

&nbsp;&nbsp;&nbsp;Another advantage is that we can focus our efforts on estimating correctly the value of the states that lead to the goal. Dynamic programming sweeps through the state space and updates every single state whether they are important or not.  
In the Maze problem, the states shaded in green are more important than the rest because they form the optimal path and therefore we want to focus our learning on them. Finally, as we said before, Monte carlo methods don't require a model of the environment, the dynamics of the environment will be implicit in our estimates.

![](../Assets/photos/monte%20carlo_6.PNG)  
![](../Assets/photos/monte%20carlo_7.PNG)


# Solving control tasks with Monte Carlo methods

&nbsp;&nbsp;&nbsp;We're going to see how to solve a control task using Monte carlo methods. To do so, we will use the template that we saw in the previous section called Generalized Policy Iteration. Remember that following this template, two processes take turns to evaluate and improve the policy, eventually leading to the optimal policy.

![](../Assets/photos/monte%20carlo_8.PNG)

&nbsp;&nbsp;&nbsp;We'll begin with any arbitrary policy. The agent will face the environment using the initial policy for one whole episode from start to finish. This will generate a trajectory from the initial state until the final reward with the rewards that we obtain in the trajectory will compute the return at each instant of time, as you see in these formulas. The return at an instant of time will be the discounted sum of rewards starting at that moment in time.

![](../Assets/photos/monte%20carlo_9.PNG)

&nbsp;&nbsp;&nbsp;Well, our strategy is to use those returns to evaluate the policy. And based on the estimated value function, improve the policy.

![](../Assets/photos/monte%20carlo_10.PNG)

&nbsp;&nbsp;&nbsp;However, we have a problem: with dynamic programming, we would update the policy using the rule below. We would search for the action that leads to the highest return using the state transition probabilities and expected rewards. We could get away with doing this because we owned a model of the environment where we could look up these values.  
&nbsp;&nbsp;&nbsp;With Monte carlo methods, we don't have a model and therefore we don't have access to those dynamics, it would be great if instead of keeping an estimate of the value of the states, we get an estimate of the expected return from taking each individual action in that state. Then it would be enough to compare those estimates and choose the action with the highest estimated return.

![](../Assets/photos/monte%20carlo_11.PNG)

&nbsp;&nbsp;&nbsp;And that's exactly what q-values do. They estimate the value of taking an action in a state, and that estimate is nothing more than the expected return from taking that action. The dynamics of the environment are implicitly captured in this estimate.

![](../Assets/photos/monte%20carlo_12.PNG)

&nbsp;&nbsp;&nbsp;Now we can change the policy so that it takes the action with the highest q-value. This means that instead of keeping a table of values for each state. We'll keep a table where the entries are, the estimates of each q-value.

![](../Assets/photos/monte%20carlo_13.PNG)

&nbsp;&nbsp;&nbsp;And then our Generalized Policy Iteration diagram will look like this: in the policy evaluation phase, we'll update our estimates of the q-value function and in our policy improvement phase, we'll use those estimates to improve the policy. This process will continue improving the policy and the q-value function until we both approach their optimal values.

![](../Assets/photos/monte%20carlo_14.PNG)

&nbsp;&nbsp;&nbsp;However, for this strategy to work, we have to keep something in mind:  
We are now going to improve our policy based on the experience that the agent collects when interacting with the environment. The experience that the agent collects depends on the actions that it takes, and those actions depend on the policy that the agent is using at the time. Therefore, we'll have a policy that will select actions based on estimates, and those estimates can be accurate, which normally occurs at the end of the learning process, or they can be inaccurate, which is normally the case at the beginning of the learning process.

![](../Assets/photos/monte%20carlo_15.PNG)

&nbsp;&nbsp;&nbsp;Imagine that there is an action that is optimal, but our estimate of its q-value is very bad. Then the policy will never choose it because the estimate is low, so we'll never get the opportunity to correct the estimate. The only way to prevent this from happening is to make sure that all actions are chosen from time to time so that we do not leave possible optimal actions undiscovered.

![](../Assets/photos/monte%20carlo_16.PNG)

&nbsp;&nbsp;&nbsp;So how can we maintain the exploration?  
Well, basically, we have two options, the first one is called exploring starts, and in it, every time the agent faces the environment to collect the experience, it starts in a random initial state and it will take an initial random action. That way, all the q-values will be updated at some point when the state and the action are chosen to start the episode. However, this is not very realistic.  
There is a lot of tasks where we simply don't have this option. That leaves us with option number two, stochastic policies. And what this means is that our policies will have a probability of choosing every action greater than 0. This ensures that from time to time, it takes an action that it doesn't consider optimal to improve its understanding of the task. Perhaps the actions that we consider to be bad turn out to be better than the ones that we consider optimal. This is perfectly possible because the policy works with estimated values and those estimates are refined during the learning process. But they don't need to be perfect from the get go. And by the way, the second option is more realistic and easier to incorporate in our methods.

![](../Assets/photos/monte%20carlo_17.PNG)

&nbsp;&nbsp;&nbsp;In fact, the use of a stochastic policies can be implemented in two different ways through the `on-policy` learning strategy and the `off-policy` learning strategy.  
The `on-policy` strategy generates the experience with the same policy that we are going to optimize while the `off-policy` strategy uses two separate policies, one to explore the environment and the other to be optimized.

![](../Assets/photos/monte%20carlo_18.PNG)



# On-policy Monte Carlo control

&nbsp;&nbsp;&nbsp;we're going to present the first Monte carlo method that we are going to implement, this method follows an `on-policy strategy` to maintain the exploration of the environment. Following this strategy we are going to define a policy that sometimes executes a random action. This policy is called `epsilon-greedy`.  
&nbsp;&nbsp;&nbsp;In this policy, every action will have a probability of being chosen greater than 0. When it's time to choose an action, we will flip a coin. With probability epsilon we will choose the action at random and with probability 1 minus epsilon, we will choose the action with the highest estimated q-value. Therefore, the probability of choosing an action that we consider suboptimal based on our estimate of its value, will be epsilon divided by the number of available actions. And the probability of choosing the action that we estimate is optimal will be one minus epsilon plus the probability of being selected randomly. With this policy, all actions will be picked from time to time, and they'll have a chance to prove that they are better than we expected.

![](../Assets/photos/monte%20carlo_19.PNG)

&nbsp;&nbsp;&nbsp;Let's look at this with an example. Suppose that epsilon equals 0.2 and there's four actions. Then one minus epsilon is 0.8, which is the probability of choosing the action with the highest estimated value. When we choose an action at random, each action has the same probability of being chosen 0.05. Therefore, the probability of choosing the optimal action is 0.85 and the probability of choosing each of the other three actions is 0.05.

![](../Assets/photos/monte%20carlo_20.PNG)

&nbsp;&nbsp;&nbsp;Here's the complete algorithm.  
It takes as input a value for epsilon, the probability of taking a random action and for gamma the discount factor to compute the returns. The policy is going to be epsilon-greedy at all times. Every time we finish a policy evaluation cycle, the probability of choosing each action will be updated accordingly. We will also keep a table with an entry for each combination of state and action. In each entry of this table, we will keep a list with the returns that the agent computes from the experience it gathers. To update the q-values, we will average these returns. When all this is set up, we will enter the main loop of the algorithm, that we'll repeat for a number of episodes. We'll make the agent interact with the environment following the present policy until the end of the episode. Once the episode is over, we'll initialize the value of the return as 0. Then for each state visited, we will compute its return. That is the sum of discounted the rewards obtained after visiting that state, and we will append that return to the list corresponding to the reference state and the action taken. Then, will update the q-value of that state and action as the average of those returns. In order to calculate the returns efficiently, we will update the q-values backwards starting at the last state visit and ending at the first one.  
&nbsp;&nbsp;&nbsp;This way we can update the return at each moment in time with this update rule as the reward obtained in that moment in time, plus the discounted cumulative return. This is an efficient way to compute the return for each state without having to add up all the rewards for each one, but the result is the same. When the process finishes, we'll have a policy and q-values close to the optimal ones. Not exactly optimal because the policy occasionally takes a random action.

![](../Assets/photos/monte%20carlo_21.PNG)

https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_4_on_policy_control_complete.ipynb#scrollTo=aMm5uC9x8Paw

![](../Assets/photos/monte%20carlo_22.PNG)



# Constant alpha Monte Carlo  

&nbsp;&nbsp;&nbsp;We are going to make a small modification to the `on-policy Monte carlo` algorithm that will allow us to update the value estimates more efficiently.  
The first difference with the original algorithm is that we are not going to keep track of the returns observed by the agent. On the contrary, when it's time to update the q-value estimates, we are going to push the estimate in the direction of the new return that we observe by a percentage alpha. It'll be like computing a weighted average between the new return observed based on experience. And the old estimate. So that as we observe new returns, we slowly push our estimates in their direction. To do that, the first thing that we need to do is declare a parameter `alpha`. That will measure the speed at which we push the estimate in the direction of the new returns.

![](../Assets/photos/monte%20carlo_23.PNG)



# Off-policy Monte Carlo control

&nbsp;&nbsp;&nbsp;we're going to see the second strategy to maintain exploration called `off-policy` learning. We want to find the optimal actions but in order to do that, from time to time, we have to take a suboptimal action to explore the environment. Otherwise, we won't know for sure which actions are the best.  
In `off-policy` learning, we separate the exploration from the optimization process and we use different policies for each one.  

![](../Assets/photos/monte%20carlo_24.PNG)

&nbsp;&nbsp;&nbsp;One of them, the exploratory policy, will perform the task and collect the experience. This experience will be the trajectory containing the states visited, the actions taken and the rewards achieved after taking those actions.

![](../Assets/photos/monte%20carlo_25.PNG)

&nbsp;&nbsp;&nbsp;That experience will be used by the other policy called the target policy. This policy is the one that will improve in the policy improvement phase to find the optimal policy. We will update the target policy based on samples of experience collected by the other policy, that is the exploratory policy. That's the reason why this strategy is called off-policy.

![](../Assets/photos/monte%20carlo_26.PNG)

&nbsp;&nbsp;&nbsp;For this strategy to work, the exploratory policy has to cover all the actions that the target policy can take. What this means is that if the target policy can take an action, meaning that its probability of choosing it is greater than 0, then the exploratory policy also has to have a probability of choosing it greater than 0. Otherwise, there would be actions that the target policy can choose and will never get updated because the exploratory policy wouldn't choose it.

![](../Assets/photos/monte%20carlo_27.PNG)

&nbsp;&nbsp;&nbsp;The problem with this strategy is that we collect the returns using the exploratory policy and if we average them, we will approximate the q-values for this exploratory policy and not for the target policy, which is the one that we really want. Something has to be done to these returns so that their average represents the q-value following the target policy.  
How can we do this?

![](../Assets/photos/monte%20carlo_28.PNG)

&nbsp;&nbsp;&nbsp;Well, we're going to apply a statistical technique known as important sampling. We'll multiply the return at time 't' by a value that we are going to call 'Wt', and this value is the probability of generating the trajectory that produced that return following the target policy, divided by the probability of generating that return, following the exploratory policy. That is, on the numerator, will have a product of all the probabilities of choosing the actions taken by the target policy and in the denominator will have the probabilities of being chosen by the exploratory policy, and that's the value that we will multiply to the returns to correct them.

![](../Assets/photos/monte%20carlo_29.PNG)

&nbsp;&nbsp;&nbsp;With this transformation, multiplying the importance sampling ratio by the return, their average will approximate the value under the targeted policy. Problem solved!

![](../Assets/photos/monte%20carlo_30.PNG)

&nbsp;&nbsp;&nbsp;The last thing we need to look at is the update rule for the q-values. As you know, our estimate of a q-value is the average of the returns observed by the agent that start from taking that action in that state. We can compute the average in two ways.  
The first one is to store the returns in the list as we observe them. And then when it's time to update the q-values, we'll compute the average of that list. This way of averaging returns, works, but it's inefficient as it needs a large amount of memory to store the returns and it repeats useless operations every time that we need to compute the average.

![](../Assets/photos/monte%20carlo_31.PNG)

&nbsp;&nbsp;&nbsp;Instead, we're going to do it in the same way as in the constant alpha Monte carlo. Every time that we observe a new return we'll nudge our current estimate of the q-value in proportion to a certain value in the direction of the new return. Only this time, instead of using a constant alpha value, as we did in the previous method, will use the important sampling ratio that we saw in the previous slides. To keep this ratio from amplifying or dampening the update too much, we will normalize the ratio by the sum of all importance sampling ratios observed for that state and action. That will keep the updates between 0 and 1 and smooth out the learning process.

![](../Assets/photos/monte%20carlo_32.PNG)

&nbsp;&nbsp;&nbsp;Finally, we are now ready to see the whole algorithm. It's similar to the previous algorithms, but it uses two policies. As we described, innovation will keep a value stable called C of S, and they will keep the sum of important sampling ratios. For each state and action so that we can use them to normalize the important sampling ratio. To start, the algorithm will enter the main loop that will repeat for a number of episodes, will use the exploratory policy to generate the trajectory by interacting with the environment and based on it, will calculate the returns. Will iterate through all of the states from the last to the first and on each one of them will calculate its return. Then we'll update the some of important sampling ratios. And used the update rule to improve the estimates of the q-value.  
If these changes alter the action taken by the target policy, then we stop the process and we move on to the next episode. Otherwise, we update the value of the important sampling ratio and we continue iterating back through time. And that's it, when the process ends will obtain the optimal policy and quality function.

![](../Assets/photos/monte%20carlo_33.PNG)


https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_4_off_policy_control_complete.ipynb#scrollTo=u450wuAjC41D













































































































