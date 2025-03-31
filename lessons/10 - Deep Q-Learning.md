# Deep Q-Learning

&nbsp;&nbsp;&nbsp;In this section, we are going to familiarize ourselves with the Deep Q-Learning algorithm. It is the combination of the temporal difference algorithm Q-Learning with neural networks. In the same way as the Deep SARSA algorithm we will have a neural network that will estimate the q-values of each action, taking a state as input.

![](../Assets/photos/Deep%20Q_1.PNG)

&nbsp;&nbsp;&nbsp;As this is an extension of the Q-learning algorithm we will follow and off-policy learning strategy. That means that we'll explore the environment using an exploratory policy that will be epsilon greedy with respect to the estimated q-values. And to update the neural network we will use a separate policy that will be greedy with respect to the estimated q-values.

![](../Assets/photos/Deep%20Q_2.PNG)

&nbsp;&nbsp;&nbsp;Therefore, we will not compute the target of the cost function based on the next action explored, but on the one chosen by the target policy, which is the policy to be optimized. This is the big difference with the deep SARSA method, which followed an on-policy exploration strategy and therefore, updates the neural network using the action chosen by the same policy that explores the environment.

![](../Assets/photos/Deep%20Q_3.PNG)

&nbsp;&nbsp;&nbsp;Now we are going to make a small change to the algorithm in order to simplify it. Instead of declaring the target policy explicitly as a function, will only declare the exploratory policy and then in the cost function, when we have to choose the q-value of the next action taken, we will use the q-value of the action that the target policy would choose. That is, the action with the maximum q-value. This is just a small simplification that we will be able to do things to the max function of the PyTorch library, but the result will be identical to what we get if we create the target policy separately and we use it in the cost function.

![](../Assets/photos/Deep%20Q_4.PNG)

&nbsp;&nbsp;&nbsp;On the other hand, in this algorithm, we are also going to use a replay memory that is going to store the experiences observed by the agent.

![](../Assets/photos/Deep%20Q_5.PNG)

&nbsp;&nbsp;&nbsp;And based on randomly chosen batches of that experience, we are going to update the neural network. Finally, we also have to emphasize that we are going to use a target network, as we did with the Deep SARSA algorithm. This network is going to give stability to the parameter updates and will allow us to obtain accurate estimates of those q-values much earlier.

![](../Assets/photos/Deep%20Q_6.PNG)  
![](../Assets/photos/Deep%20Q_7.PNG)


https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_9_deep_q_learning_complete.ipynb









