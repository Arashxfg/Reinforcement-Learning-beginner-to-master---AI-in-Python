#  Deep SARSA

&nbsp;&nbsp;&nbsp;In this section, we're going to extend the temporal difference method to estimate the Q values using a neural network. The resulting algorithm will be known as deep SARSA. This method is the combination of the SARSA algorithm with neural networks and the techniques used to train them which are called deep learning. In this version of the algorithm, the estimates of the Q values will not be stored in a table, but they will be produced by feeding the state as input into a neural network. And the output of that neural network will be a vector where each element is the Q value, the estimated Q value of a specific action for that state.

![](../Assets/photos/Deep%20SARSA_1.PNG)

&nbsp;&nbsp;&nbsp;The resulting algorithm will look like this. Let's take a quick look at it and then we'll explain in detail the differences. As we said, we are going to use a neural network to estimate the Q values. So instead of initializing a table of Q values as before, we'll initialize the parameters of that neural network. As in. The policy that we are going to use will be an epsilon greedy policy. That is, the policy will select random actions with a certain probability defined by the value of epsilon and with probability one minus epsilon. The policy will choose the action with the highest estimated Q value. The algorithm learns from the experience collected by the agent interacting with the environment. Just like the original SARSA algorithm. So the main part of the algorithm is the execution of this loop, which will be repeated for a number of episodes.  
This loop consists of two parts. The first one is the part where the agent interacts with the environment to collect experiences, which it will then use to refine the neural network estimates in order to improve the policy. The second part of the algorithm is where we'll update the parameters of the neural network. We'll do that based on the experiences that the agent collected before, and we'll update the parameters of the neural network so that the estimates of the Q values that it produces are more accurate every time we update it. This update is produced by the Stochastic Gradient Descent algorithm.

![](../Assets/photos/Deep%20SARSA_2.PNG)



# Neural Network optimization (Deep Q-Network)

&nbsp;&nbsp;&nbsp;We are going to see how to optimize our neural network to produce increasingly accurate q-value estimates. What we'll do is compare the estimates made by the neural network with the correct value. The difference between these two values is the estimation error. With this error, we'll define a cost function that is a function that represents the size of the errors and that we are going to minimize in the deep SARSA algorithm. We are going to minimize the cost function called mean squared error, which consists of the mean of the square of the errors.

![](../Assets/photos/Deep%20SARSA_3.PNG)

&nbsp;&nbsp;&nbsp;In our case, the estimates will be the Q values produced by the neural network for a specific state and a specific action. The target value, as in the original SARSA algorithm, will be the return estimated in one step. That is, the reward obtained by executing the action and the discounted Q value of the action chosen by the policy in the next state. It is this difference that will have to minimize by defining and minimizing the cost function. To do that, we'll compute the value of the cost function based on a batch of experiences taken from the replay memory. Based on them, we'll compute the estimated Q values and the target values for each state transition, and based on them we'll compute the cost function.

![](../Assets/photos/Deep%20SARSA_4.PNG)

&nbsp;&nbsp;&nbsp;Once we have the value of the cost function, we'll compute its gradient with respect to each of the parameters of the neural network. This vector will indicate the direction in which to modify the neural network parameters to maximize the cost function. But since we are performing, Stochastic gradient descent will be interested in minimizing the loss function. Therefore, we'll move the parameters of the neural network in the direction opposite to the gradient vector of the loss function by an amount proportional to alpha. As you can see, the update rule is very similar to the one that we saw in our tabular algorithms, except that now, instead of modifying the values of the tables, we are modifying the parameters of the neural network. That is the process known as stochastic gradient descent.

![](../Assets/photos/Deep%20SARSA_5.PNG)

&nbsp;&nbsp;&nbsp;We'll perform this update at every moment in time for the duration of the control task right after the agent interacts with the environment.

![](../Assets/photos/Deep%20SARSA_6.PNG)



# Experience Replay

&nbsp;&nbsp;&nbsp;We are going to see what a replay memory is and how we are going to use it. In the Deep SARSA algorithm, a replay memory is a database in which we are going to store the state transitions that the agent observes when facing the control task. Each time the agent executes an action, we are going to record in the memory the state that the agent was in, the action taken by the agent. The reward obtained as a consequence of taking that action and the next state reached by the agent.

![](../Assets/photos/Deep%20SARSA_7.PNG)

&nbsp;&nbsp;&nbsp;The memory has a maximum number of entries and when this number is exceeded, it will begin to delete the oldest entries and replace them with the latest transitions that the agent observes. In this way, the replay memory will keep a fresh pool of state transitions.

![](../Assets/photos/Deep%20SARSA_8.PNG)

&nbsp;&nbsp;&nbsp;Then, when it's time to update the neural network, we'll use the experience that we have stored in the memory to do so. From all the state. Transitions stored in the memory will randomly choose a batch of transitions. The size of the batch is chosen by us.

![](../Assets/photos/Deep%20SARSA_9.PNG)

&nbsp;&nbsp;&nbsp;Once we have that batch, we'll use it to compute the cost function. And based on that estimate of the cost function, we'll update the neural network.

![](../Assets/photos/Deep%20SARSA_10.PNG)

&nbsp;&nbsp;&nbsp;The replay memory is involved in these three lines of the algorithm. First, before entering the main loop, we'll create an empty replay memory. At each moment in time, during the episode, we'll store the state transition immediately after the agent experiences it. And also at this moment in time, during the episode, we'll sample a batch of transitions to update the neural network. For that. We'll take the batch from the memory. We'll compute the estimate of the cost function, and based on it, we'll perform a stochastic gradient descent step.

![](../Assets/photos/Deep%20SARSA_11.PNG)



# Target Network

&nbsp;&nbsp;&nbsp;We're going to see the last change that will make to the algorithm to adapt it to the use of neural networks In the deep SARSA algorithm, we are going to combine two techniques that can lead to an unstable learning process. On the one hand, we're going to do bootstrapping. That is, we are going to update an estimate by taking another estimate as reference. As you can see, the target towards which we take the Q-value estimates contains an estimate of the value of the next state and the next action. On the other hand, we are going to use a function approximator, which in this case will be a neural network.

![](../Assets/photos/Deep%20SARSA_12.PNG)

&nbsp;&nbsp;&nbsp;But why can these two techniques combined generate an unstable learning process? Well, look at the graph on the right, which represents the Q-value estimates computed by a neural network for a particular action in each state of this embedded control task. When we update the estimate for a state via gradient descent, we not only modify the estimate for that state, but for all states near it. In the graph, the blue function represents the Q-value estimates before performing the gradient descent step and the red one the estimates after. When the agent performs an action, the next state is usually very similar to the previous one. What this means is that when we update the estimated Q value that we want to optimize, we'll also be shifting its target. That is the value that we want our estimates to follow. Since our estimates of the Q values will be approaching a moving target, this process will be unstable.

![](../Assets/photos/Deep%20SARSA_13.PNG)

&nbsp;&nbsp;&nbsp;For the learning process to be stable, the targets must also be stable since they represent the correct values that our estimates must move to.

![](../Assets/photos/Deep%20SARSA_14.PNG)

&nbsp;&nbsp;&nbsp;But how can we achieve this? Well, what we are going to do is create an exact replica of the neural network that estimates the Q values. And we are going to use that copy only to estimate the target values. This neural network is known as the target network. The difference is that when we do gradient descent to minimize the error of the estimates, this target network won't be modified. Its parameters will remain the same. This means that the target estimates will remain stable during the learning process.

![](../Assets/photos/Deep%20SARSA_15.PNG)

&nbsp;&nbsp;&nbsp;For this reason, in the expression of the cost function, we write theta target as the neural network parameters because these estimates are made using the target network.

![](../Assets/photos/Deep%20SARSA_16.PNG)

&nbsp;&nbsp;&nbsp;Then every few episodes will make a new copy of the main neural network, and we'll use that fresh copy to estimate the target values so that those target value estimates also become more accurate throughout the learning process.

![](../Assets/photos/Deep%20SARSA_17.PNG)

&nbsp;&nbsp;&nbsp;The target network will intervene in these three lines of code. First, before we enter the main loop, we'll create the replica of the neural network. Then when we compute the loss function, we'll use the copy of the neural network to produce the estimates included in the target value. And then every K episodes of the environment will synchronize the neural network's.

![](../Assets/photos/Deep%20SARSA_18.PNG)


https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_8_deep_sarsa_complete.ipynb#scrollTo=55v03cjFI_3e






























































